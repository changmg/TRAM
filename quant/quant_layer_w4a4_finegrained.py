import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
from torch.autograd import Function

import self_ops

from quant.quantizer import UniformAffineQuantizer
from quant.app_mult import error_of_discard_3_columns as get_error
from quant.app_mult import get_ith_bit
from power_model.main_power_model import MLP as MLP_w8a8
from power_model.main_power_model_w4a4 import MLP as MLP_w4a4
from conf import settings
print_log = settings.LOGGER.info


class MaskFunc(Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return (input >= 0).float()

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        mask = ((input >= -1) & (input <= 1)).float()
        grad_input = grad_output * mask
        return grad_input
mask_func = MaskFunc.apply


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
            self.app_fwd_func = self_ops.approx_linear_op
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

    # # forward function for removing fixed number of partial product columns
    # def forward(self, input: torch.Tensor):
    #     input_int, input_fq = self.act_quantizer(input) if self.use_act_quant else (None, input)
    #     weight_int, weight_fq = self.weight_quantizer(self.weight) if self.use_weight_quant else (None, self.weight)
    #     if self.use_appmult:
    #         assert self.use_act_quant and self.use_weight_quant, 'AppMult requires both weight and activation quantization'
    #         out = self.fwd_func(input_fq, weight_fq, bias=self.bias, **self.fwd_kwargs)
    #         error = get_error(act_scale=self.act_quantizer.scale, weight_scale=self.weight_quantizer.scale, input_int=input_int, weight_int=weight_int, fwd_func=self.fwd_func, fwd_kwargs=self.fwd_kwargs)
    #         out += -error
    #     else:
    #         out = self.fwd_func(input_fq, weight_fq, bias=self.bias, **self.fwd_kwargs)
    #     return out

    # general forward function for arbitrary AppMults (LUT-based)
    def forward(self, input: torch.Tensor):
        input_int, input_fq = self.act_quantizer(input) if self.use_act_quant else (None, input)
        weight_int, weight_fq = self.weight_quantizer(self.weight) if self.use_weight_quant else (None, self.weight)
        if not self.use_appmult:
            out = self.fwd_func(input_fq, weight_fq, bias=self.bias, **self.fwd_kwargs)
        else:
            assert self.use_act_quant and self.use_weight_quant, 'AppMult requires both weight and activation quantization'
            scale_a, zero_point_a = self.act_quantizer.scale, self.act_quantizer.zero_point
            scale_w, zero_point_w = self.weight_quantizer.scale, self.weight_quantizer.zero_point
            out = scale_a * scale_w.view(self.view_shape) * (
                self.app_fwd_func(input_int, weight_int, bias=None, **self.fwd_kwargs)
                - self.fwd_func(zero_point_a.expand_as(input_int), weight_int, bias=None, **self.fwd_kwargs)
                - self.fwd_func(input_int, zero_point_w.expand_as(weight_int), bias=None, **self.fwd_kwargs)
                + self.fwd_func(zero_point_a.expand_as(input_int), zero_point_w.expand_as(weight_int), bias=None, **self.fwd_kwargs)
            )
            if self.bias is not None:
                out += self.bias.view(self.view_shape)
        return out

    # # forward function for w8a8 DSE
    # def forward(self, input: torch.Tensor):
    #     input_int, input_fq = self.act_quantizer(input) if self.use_act_quant else (None, input)
    #     weight_int, weight_fq = self.weight_quantizer(self.weight) if self.use_weight_quant else (None, self.weight)
    #     out = self.fwd_func(input_fq, weight_fq, bias=self.bias, **self.fwd_kwargs)
    #     if self.use_appmult:
    #         assert self.use_act_quant and self.use_weight_quant, 'AppMult requires both weight and activation quantization'
    #         # extract bits
    #         wi = [get_ith_bit(weight_int, i) for i in range(8)]
    #         xi = [get_ith_bit(input_int, i) for i in range(8)]
    #         # compute sum of partial product columns
    #         sum_col = torch.zeros((8, *out.shape), dtype=torch.float32, device=out.device)
    #         sum_col[0] = self.fwd_func(xi[0], wi[0], bias=None, **self.fwd_kwargs)
    #         sum_col[1] = self.fwd_func(xi[0], wi[1], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[1], wi[0], bias=None, **self.fwd_kwargs)
    #         sum_col[2] = self.fwd_func(xi[0], wi[2], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[1], wi[1], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[2], wi[0], bias=None, **self.fwd_kwargs)
    #         sum_col[3] = self.fwd_func(xi[0], wi[3], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[1], wi[2], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[2], wi[1], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[3], wi[0], bias=None, **self.fwd_kwargs)
    #         sum_col[4] = self.fwd_func(xi[0], wi[4], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[1], wi[3], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[2], wi[2], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[3], wi[1], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[4], wi[0], bias=None, **self.fwd_kwargs)
    #         sum_col[5] = self.fwd_func(xi[0], wi[5], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[1], wi[4], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[2], wi[3], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[3], wi[2], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[4], wi[1], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[5], wi[0], bias=None, **self.fwd_kwargs)
    #         sum_col[6] = self.fwd_func(xi[0], wi[6], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[1], wi[5], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[2], wi[4], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[3], wi[3], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[4], wi[2], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[5], wi[1], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[6], wi[0], bias=None, **self.fwd_kwargs)
    #         sum_col[7] = self.fwd_func(xi[1], wi[6], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[2], wi[5], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[3], wi[4], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[4], wi[3], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[5], wi[2], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[6], wi[1], bias=None, **self.fwd_kwargs) + self.fwd_func(xi[7], wi[0], bias=None, **self.fwd_kwargs)
    #         # discard columns
    #         out -= self.act_quantizer.scale * self.weight_quantizer.scale.view(self.view_shape) * (
    #             mask_func(self.indicators[0]) * sum_col[0] +
    #             mask_func(self.indicators[1]) * sum_col[1] +
    #             mask_func(self.indicators[2]) * sum_col[2] +
    #             mask_func(self.indicators[3]) * sum_col[3] +
    #             mask_func(self.indicators[4]) * sum_col[4] +
    #             mask_func(self.indicators[5]) * sum_col[5] +
    #             mask_func(self.indicators[6]) * sum_col[6] +
    #             mask_func(self.indicators[7]) * sum_col[7]
    #         )
    #     return out

    # # forward function for w4a4 DSE
    # def forward(self, input: torch.Tensor):
    #     input_int, input_fq = self.act_quantizer(input) if self.use_act_quant else (None, input)
    #     weight_int, weight_fq = self.weight_quantizer(self.weight) if self.use_weight_quant else (None, self.weight)
    #     out = self.fwd_func(input_fq, weight_fq, bias=self.bias, **self.fwd_kwargs)
    #     if self.use_appmult:
    #         assert self.use_act_quant and self.use_weight_quant, 'AppMult requires both weight and activation quantization'
    #         # extract bits
    #         wi = [get_ith_bit(weight_int, i) for i in range(3)]
    #         xi = [get_ith_bit(input_int, i) for i in range(3)]
    #         # discard columns
    #         out -= self.act_quantizer.scale * self.weight_quantizer.scale.view(self.view_shape) * (
    #             mask_func(self.indicators[0]) * self.fwd_func(xi[0], wi[0], bias=None, **self.fwd_kwargs) +
    #             mask_func(self.indicators[1]) * self.fwd_func(xi[1], wi[0], bias=None, **self.fwd_kwargs) + 
    #             mask_func(self.indicators[2]) * self.fwd_func(xi[0], wi[1], bias=None, **self.fwd_kwargs) + 
    #             mask_func(self.indicators[3]) * self.fwd_func(xi[2], wi[0], bias=None, **self.fwd_kwargs) + 
    #             mask_func(self.indicators[4]) * self.fwd_func(xi[1], wi[1], bias=None, **self.fwd_kwargs) + 
    #             mask_func(self.indicators[5]) * self.fwd_func(xi[0], wi[2], bias=None, **self.fwd_kwargs)
    #         )
    #     return out

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

    def set_layer_id(self, layer_id: int):
        self.layer_id = layer_id
    
    # # for W8A8 DSE
    # def add_trainable_indicators(self):
    #     ind_init_value = -1e-3
    #     self.indicators = torch.nn.Parameter(torch.ones((8,), dtype=torch.float32, device='cuda') * ind_init_value) # self.indicators[i] denotes whether to keep or discard the i-th column of partial products

    # for W4A4 DSE
    def add_trainable_indicators(self):
        ind_init_value = -1e-3
        # am_configs[i] = 1 means removing a certain partial product from the multiplier
        # am_configs[0] corresponds to w0x0, am_configs[1] to w0x1, am_configs[2] to w1x0,
        # am_configs[3] corresponds to w0x2, am_configs[4] to w1x1, am_configs[5] to w2x0
        self.indicators = torch.nn.Parameter(torch.ones((6,), dtype=torch.float32, device='cuda') * ind_init_value)

    def print_indicators(self, _binary: bool = False):
        if hasattr(self, 'indicators'):
            if _binary:
                print_log(f'layer_id = {self.layer_id:2.0f}, ind = {[f"{1 if (indicator >= 0) else 0}" for indicator in self.indicators]}')
            else:
                print_log(f'layer_id = {self.layer_id:2.0f}, ind = {[f"{indicator.item():5.3f}" for indicator in self.indicators]}')

    def initialize_power_model(self):
        # self.power_MLP = MLP_w8a8()
        # self.power_MLP.load_state_dict(torch.load('./power_model/power_model_w8a8.pth', map_location='cuda', weights_only=True))
        self.power_MLP = MLP_w4a4()
        self.power_MLP.load_state_dict(torch.load('./power_model/power_model_w4a4.pth', map_location='cuda', weights_only=True))
        self.power_MLP.eval() # set to eval mode
        for p in self.power_MLP.parameters(): # freeze MLP parameters
            p.requires_grad = False

    def compute_power_per_mul(self):
        if hasattr(self, 'power_MLP'):
            return self.power_MLP(mask_func(self.indicators))
            # masked_indicators = mask_func(self.indicators[:6])  # apply mask to first 6 indicators
            # padded_input = torch.cat([masked_indicators, torch.zeros(2, device=self.indicators.device)]) # do not use the last 2 indicators (do not discard)
            # return self.power_MLP(padded_input)
        else:
            raise ValueError('power_MLP not found')
        
    def extra_repr(self):
        return f'{self._extra_repr}, layer_id: {self.layer_id}, use_weight_quant: {self.use_weight_quant}, use_act_quant: {self.use_act_quant}, use_appmult: {self.use_appmult}, use_fixed_appmult: {self.use_fixed_appmult}'