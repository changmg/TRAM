import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
from torch.autograd import Function

import self_ops

from quant.quantizer import UniformAffineQuantizer
from quant.app_mult import error_of_discard_3_columns as get_error
from quant.app_mult import get_ith_bit
from utils.common import tensor_to_str
from conf import settings
print_log = settings.LOGGER.info


def ste_clamp01(x):
    y = x.clamp(0.0, 1.0)              # forward value in [0, 1]
    return x + (y - x).detach()        # backward grad is 1


class QuantLayer(nn.Module):
    """
    Quantized layer that can perform quantized convolution/linear or normal convolution/linear operation.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super(QuantLayer, self).__init__()
        # save original module parameters
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding, dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
            self.app_fwd_func = self_ops.approx_conv2d_baseline_op
            self.view_shape = (1, org_module.out_channels, 1, 1)
            self._extra_repr = org_module.extra_repr() + '\nfunction: conv2d'
            self.kernel_size = org_module.kernel_size
            self.stride = org_module.stride
            self.padding = org_module.padding
            self.dilation = org_module.dilation
            self.out_channels = org_module.out_channels
            assert org_module.groups == 1, 'Grouped conv2d is not supported'
        elif isinstance(org_module, nn.Linear):
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
            # self.app_fwd_func = self_ops.approx_linear_op
            self.app_fwd_func = F.linear # do not approximate linear layer as it has small number of MACs
            self.view_shape = (1, org_module.out_features)
            self._extra_repr = org_module.extra_repr() + '\nfunction: linear'
            self.in_features = org_module.in_features
            self.out_features = org_module.out_features
        else:
            raise ValueError('Unknown module type: {}'.format(type(org_module)))
        self.weight = org_module.weight
        self.bias = org_module.bias
        # status parameters
        self.register_buffer('use_weight_quant', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('use_act_quant', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('use_appmult', torch.tensor(False, dtype=torch.bool))
        self.use_fixed_appmult = True
        self.cali_appmult = False
        self.layer_id = -1
        # initialize quantizer
        if isinstance(org_module, nn.Conv2d):
            weight_quantizer_shape = (self.weight.shape[0], 1, 1, 1) if weight_quant_params['channel_wise'] else ()
        elif isinstance(org_module, nn.Linear):
            weight_quantizer_shape = (self.weight.shape[0], 1) if weight_quant_params['channel_wise'] else ()
        else:
            raise ValueError('Unknown module type: {}'.format(type(org_module)))
        assert not act_quant_params['channel_wise']
        act_quantizer_shape = ()
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params, quantizer_shape=weight_quantizer_shape, observer='minmax')
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params, quantizer_shape=act_quantizer_shape, observer='minmax')
        # initialize number of MACs
        self.num_macs = None # will be calculated later
        self.hook_registered = False
        # save number of quantization bits
        self.num_bits = weight_quant_params['n_bits']
        assert self.num_bits == act_quant_params['n_bits'], 'Weight and activation quantization bits must be the same'
        # maximum number of partial product columns that can be discarded
        self.num_max_discard_cols = None #  will be initialized later
        

    def compute_macs(self):
        def hook_fn(module, input, output):
            x = input[0]
            if self.fwd_func is F.conv2d:
                # x shape: (N, C_in, H_in, W_in)
                N, C_in, H_in, W_in = x.shape
                K_h, K_w = self.kernel_size
                S_h, S_w = self.stride
                P_h, P_w = self.padding
                D_h, D_w = self.dilation
                H_out = (H_in + 2 * P_h - D_h * (K_h - 1) - 1) // S_h + 1
                W_out = (W_in + 2 * P_w - D_w * (K_w - 1) - 1) // S_w + 1
                self.num_macs = C_in * K_h * K_w * self.out_channels * H_out * W_out
                print_log(f'layer_id = {self.layer_id}, H_in = {H_in}, W_in = {W_in}, C_in = {C_in}, K_h = {K_h}, K_w = {K_w}, C_out = {self.out_channels}, H_out = {H_out}, W_out = {W_out}, #MACs = {self.num_macs}')
            elif self.fwd_func is F.linear:
                # x shape: (N, in_features)
                self.num_macs = self.in_features * self.out_features
                print_log(f'layer_id = {self.layer_id}, in_features = {self.in_features}, out_features = {self.out_features}, #MACs = {self.num_macs}')
            else:
                raise ValueError('Unknown function: {}'.format(self.fwd_func))
            # Unregister the hook after first use
            self._macs_hook.remove()

        if not self.hook_registered:
            self._macs_hook = self.register_forward_hook(hook_fn)
            self.hook_registered = True

    # forward function for w8a8 DSE (save memory)
    def forward(self, input: torch.Tensor):
        input_int, input_fq = self.act_quantizer(input) if self.use_act_quant else (None, input)
        weight_int, weight_fq = self.weight_quantizer(self.weight) if self.use_weight_quant else (None, self.weight)
        out = self.fwd_func(input_fq, weight_fq, bias=self.bias, **self.fwd_kwargs)
        if self.use_appmult and self.fwd_func is not F.linear: # approximate convolution only
            assert self.use_act_quant and self.use_weight_quant, 'AppMult requires both weight and activation quantization'
            assert self.num_bits == 8, 'This AppMult implementation only supports W8A8'
            # extract bits
            needed_bits = min(self.num_max_discard_cols, self.num_bits)
            w = [get_ith_bit(weight_int, i) for i in range(needed_bits)]
            x = [get_ith_bit(input_int, i) for i in range(needed_bits)]
            # discard columns
            error = 0.0
            for col_id in range(self.num_max_discard_cols):
                error_col = 0.0
                min_i = max(0, col_id - (self.num_bits - 1))
                max_i = min(col_id + 1, self.num_bits) # exclusive
                for i in range(min_i, max_i):
                    j = col_id - i
                    error_col += self.fwd_func(x[i], w[j], bias=None, **self.fwd_kwargs)
                error += ste_clamp01(self.gamma_free[col_id]) * error_col
            out -= self.act_quantizer.scale * self.weight_quantizer.scale.view(self.view_shape) * error
        return out

    def set_app_state(self, weight_quant: bool, act_quant: bool, use_appmult: bool):
        self.use_weight_quant = torch.tensor(weight_quant, dtype=torch.bool)
        self.use_act_quant = torch.tensor(act_quant, dtype=torch.bool)
        self.use_appmult = torch.tensor(use_appmult, dtype=torch.bool)

    def get_app_state(self):
        return self.use_weight_quant.item(), self.use_act_quant.item(), self.use_appmult.item()

    def set_observers_status(self, enable_observers: bool):
        self.weight_quantizer.set_observers_status(enable_observers)
        self.act_quantizer.set_observers_status(enable_observers)

    def use_new_observer(self, observer_str: str):
        self.weight_quantizer.use_new_observer(observer_str)
        self.act_quantizer.use_new_observer(observer_str)
    
    def set_fixed_appmult(self, use_fixed_appmult: bool):
        self.use_fixed_appmult = use_fixed_appmult
        # freeze the gradient of indicators
        if hasattr(self, 'indicators'):
            self.indicators.requires_grad = not use_fixed_appmult
        if hasattr(self, 'gamma_free'):
            self.gamma_free.requires_grad = not use_fixed_appmult

    def set_layer_id(self, layer_id: int):
        self.layer_id = layer_id
    
    # for W8A8 DSE
    # def add_trainable_indicators(self):
    #     ind_init_value = -1e-3
    #     self.indicators = torch.nn.Parameter(torch.ones((8,), dtype=torch.float32, device='cuda') * ind_init_value) # self.indicators[i] denotes whether to keep or discard the i-th column of partial products
    def add_trainable_indicators(self, init_value):
        assert init_value.dim() == 1
        self.num_max_discard_cols = init_value.shape[0]
        assert self.num_max_discard_cols <= 2 * self.num_bits - 1 and self.num_max_discard_cols > 0, 'num_max_discard_cols must be in [1, 2*n_bits-1]'
        # ind_init_value = 0.0
        # self.gamma[i] denotes how "strongly" we want to discard the i-th column of partial products
        # 0 (keep the column) <= gamma <= 1 (discard whole column)
        self.gamma_free = torch.nn.Parameter(init_value.clone())
        print_log(f'adding indicators to layer_id = {self.layer_id}: num_bits = {self.num_bits}, num_max_discard_cols = {self.num_max_discard_cols}, address of self.gamma_free = {hex(id(self.gamma_free))}')

    def add_trainable_indicators_homogeneous(self, gamma_free):
        assert gamma_free.dim() == 1
        self.num_max_discard_cols = gamma_free.shape[0]
        assert self.num_max_discard_cols <= 2 * self.num_bits - 1 and self.num_max_discard_cols > 0, 'num_max_discard_cols must be in [1, 2*n_bits-1]'
        self.gamma_free = gamma_free
        print_log(f'adding homogeneous indicators to layer_id = {self.layer_id}: num_bits = {self.num_bits}, num_max_discard_cols = {self.num_max_discard_cols}, address of self.gamma_free = {hex(id(self.gamma_free))}')

    # def print_indicators(self, _binary: bool = False):
    #     if hasattr(self, 'indicators'):
    #         if _binary:
    #             print_log(f'layer_id = {self.layer_id:2.0f}, ind = {[f"{1 if (indicator >= 0) else 0}" for indicator in self.indicators]}')
    #         else:
    #             print_log(f'layer_id = {self.layer_id:2.0f}, ind = {[f"{indicator.item():5.3f}" for indicator in self.indicators]}')
    def print_indicators(self, _binary: bool = False):
        if hasattr(self, 'gamma_free'):
            print_log(f'layer_id = {self.layer_id:2.0f}, gamma_free = {[f"{gamma_free.item():10.6f}" for gamma_free in self.gamma_free]}')
            print_log(f'layer_id = {self.layer_id:2.0f}, gamma      = {[f"{gamma.item():10.6f}" for gamma in ste_clamp01(self.gamma_free)]}')

    # def initialize_power_model(self):
    #     self.power_MLP = MLP_w8a8()
    #     self.power_MLP.load_state_dict(torch.load('./power_model/power_model_w8a8.pth', map_location='cuda', weights_only=True))
    #     self.power_MLP.eval() # set to eval mode
    #     for p in self.power_MLP.parameters(): # freeze MLP parameters
    #         p.requires_grad = False
    def initialize_power_model(self):
        pass

    # def compute_power_per_mul(self):
    #     if hasattr(self, 'power_MLP'):
    #         return self.power_MLP(mask_func(self.indicators))
    #     else:
    #         raise ValueError('power_MLP not found')
    # def compute_power_per_mul(self):
    #     if hasattr(self, 'gamma_free'):
    #         # obtain total cost
    #         # self.num_bits-1 rows and self.num_bits columns of compressors; with an additional 0.5 for the partial product w0x0
    #         total_cost = (self.num_bits - 1) * self.num_bits + 0.5
    #         # obtain fixed-part cost
    #         # estimate power consumption based on gamma values
    #         assert self.gamma_free.shape == (self.num_max_discard_cols,)
    #         num_compressors_per_col = [min(col_id, self.num_bits - 1, 2*self.num_bits - col_id - 1) for col_id in range(self.num_max_discard_cols)]
    #         num_compressors_per_col[0] = 0.5 # first column uses a logic AND gate instead of a compressor
    #         num_compressors_tensor = torch.tensor(num_compressors_per_col, dtype=torch.float32, device='cuda')
    #         gamma = ste_clamp01(self.gamma_free)
    #         cost = torch.sum((1.0 - gamma) * num_compressors_tensor)
    #         # print_log(f'layer_id = {self.layer_id}: num_compressors_tensor = {tensor_to_str(num_compressors_tensor, 20)}, gamma_free = {tensor_to_str(self.gamma_free)}, sum_cost = {sum_cost.item():.4f}')
    #         # fixed-part cost
    #         dummy_gamma = torch.zeros((self.num_max_discard_cols,), dtype=torch.float32, device='cuda') # all columns kept
    #         fixed_cost = total_cost - torch.sum((1.0 - dummy_gamma) * num_compressors_tensor)
    #         return (fixed_cost + cost) / total_cost
    #     else:
    #         raise ValueError('power_MLP not found')


    def init_hardware_cost(self):
        # hardware cost model parameters
        # AND power: 5.28e-05; HA power: 9.62e-05; FA power: 2.36e-04
        C_and = 1.000
        C_ha  = 1.822
        C_fa  = 4.470
        print_log(f'Initializing hardware cost for layer_id = {self.layer_id}: C_and = {C_and}, C_ha = {C_ha}, C_fa = {C_fa}')
        N = self.num_bits
        num_cols = 2 * N - 1
        cost = []
        total_ha = 0
        total_fa = 0
        # refer to array multiplier structure
        for c in range(num_cols):
            # number of partial products in column c
            n_pp = min(c + 1, N, 2 * N - 1 - c)
            n_and = n_pp
            if c == 0:
                # only one AND gate, no compressor tree
                n_ha = 0
                n_fa = 0
            elif c < N:
                # simple split of n_pp - 1 additions into 1 HA and the rest FAs
                assert n_pp >= 2
                n_ha = 1
                n_fa = max(n_pp - 2, 0)
            elif c == N:
                n_ha = 1
                n_fa = n_pp - 1
            else:
                n_ha = 0
                n_fa = n_pp
            col_cost = C_and * n_and + C_ha * n_ha + C_fa * n_fa
            total_ha += n_ha
            total_fa += n_fa
            cost.append(col_cost)
            print_log(f'col {c}: n_pp = {n_pp}, n_and = {n_and}, n_ha = {n_ha}, n_fa = {n_fa}, col_cost = {col_cost}')
        print_log(f'Total compressors: HA = {total_ha}, FA = {total_fa}')
        # [2*num_bits-1]
        device = self.weight.device
        self.col_cost = torch.tensor(cost, dtype=torch.float32).to(device)
        # precompute totals
        assert self.num_max_discard_cols is not None, 'num_max_discard_cols must be initialized before init_hardware_cost'
        self.droppable_cost_total = self.col_cost[:self.num_max_discard_cols].sum()
        self.total_cost = self.col_cost.sum()
        self.fixed_cost = self.total_cost - self.droppable_cost_total # fixed part, including non droppable columns
    def compute_power_per_mul(self):
        if not hasattr(self, 'gamma_free'):
            raise ValueError('gamma_free not found')
        if not hasattr(self, 'col_cost'):
            self.init_hardware_cost()
        # γ in [0,1], γ=1 means fully dropped
        gamma = ste_clamp01(self.gamma_free)  # [num_max_discard_cols]
        col_cost_droppable = self.col_cost[:self.num_max_discard_cols]  # same device as layer
        # variable part
        active_cost = torch.sum((1.0 - gamma) * col_cost_droppable)
        # get power per multiplication
        power_per_mul = (self.fixed_cost + active_cost) / self.total_cost
        return power_per_mul


    def extra_repr(self):
        return f'{self._extra_repr}, layer_id: {self.layer_id}, use_weight_quant: {self.use_weight_quant}, use_act_quant: {self.use_act_quant}, use_appmult: {self.use_appmult}, use_fixed_appmult: {self.use_fixed_appmult}'