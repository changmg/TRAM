import torch
import torch.nn as nn

from quant.quant_layer_lut import QuantLayer
from quant.quantizer import LearnedClippingObserver, MovingAverageLearnedClippingObserver
from conf import settings
print_log = settings.LOGGER.info


class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, dataset_name: str, weight_quant_params: dict = {'n_bits': 8, 'channel_wise': True}, act_quant_params: dict = {'n_bits': 8, 'channel_wise': False}):
        super().__init__()
        self.model = model
        self.dataset_name = dataset_name
        self.quant_module_replace(self.model, weight_quant_params, act_quant_params)
        self.collect_quant_layers()
        self.rename_quant_layers()
        self.num_bits = weight_quant_params['n_bits']
        assert self.num_bits == act_quant_params['n_bits'], 'weight and activation quantization bitwidth must be the same'

    def quant_module_replace(self, module: nn.Module, weight_quant_params: dict, act_quant_params: dict):
        """
        Recursively replace the normal conv2d and Linear layer to QuantLayer
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        import models.cifar10.inception as inception
        if self.dataset_name == 'cifar10':
            from models.cifar10.resnet import BasicBlock, Bottleneck
        elif self.dataset_name == 'imagenet':
            from models.imagenet.resnet import BasicBlock, Bottleneck
        else:
            raise ValueError(f'Unknown dataset: {self.dataset_name}')
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Linear)): # quantize conv2d and linear layers
                setattr(module, name, QuantLayer(child_module, weight_quant_params, act_quant_params))
            elif isinstance(child_module, (nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.BatchNorm2d, nn.ReLU, nn.Dropout)): # skip these layers; batchnorm, relu, & dropout will be folded later
                continue
            elif isinstance(child_module, (
                nn.Sequential, BasicBlock, Bottleneck, 
                inception.BasicConv2d, inception.InceptionA, inception.InceptionB, inception.InceptionC, inception.InceptionD, inception.InceptionE
            )): # recursively quantize the children of these layers
                self.quant_module_replace(child_module, weight_quant_params, act_quant_params)
            else:
                raise ValueError(f'Unknown module for quantization: {child_module}')

    def collect_quant_layers(self):
        self.quant_layers = []
        def collect_quant_layers_rec(self, curr_module: nn.Module):
            for _, child_module in curr_module.named_children():
                if isinstance(child_module, QuantLayer):
                    self.quant_layers += [child_module]
                else:
                    collect_quant_layers_rec(self, child_module)
        collect_quant_layers_rec(self, curr_module=self.model)

    def rename_quant_layers(self):
        assert self.quant_layers is not None
        for i, quant_layer in enumerate(self.quant_layers):
            quant_layer.set_layer_id(i)

    def forward(self, input):
        return self.model(input)

    def set_app_state(self, weight_quant: bool, act_quant: bool, use_appmult: bool):
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                m.set_app_state(weight_quant, act_quant, use_appmult)

    def get_app_state(self):
        weight_quantized = None
        act_quantized = None
        use_appmult = None
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                _weight_quantized, _act_quantized, _use_appmult = m.get_app_state()
                if weight_quantized is None:
                    weight_quantized = _weight_quantized
                else:
                    assert weight_quantized == _weight_quantized, 'inconsistent weight quantization state'
                if act_quantized is None:
                    act_quantized = _act_quantized
                else:
                    assert act_quantized == _act_quantized, 'inconsistent act quantization state'
                if use_appmult is None:
                    use_appmult = _use_appmult
                else:
                    assert use_appmult == _use_appmult, 'inconsistent use_appmult state'
        return weight_quantized, act_quantized, use_appmult

    def set_observers_status(self, enable_observer: bool):
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                m.set_observers_status(enable_observer)

    def set_fixed_appmult(self, use_fixed_appmult: bool):
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                m.set_fixed_appmult(use_fixed_appmult)

    def print_indicators(self):
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                m.print_indicators()

    def compute_hardware_loss(self, _print=False):
        loss = 0
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                power_per_mul = m.compute_power_per_mul()
                if _print:
                    print_log(f"Layer {m.layer_id}: power_per_mul = {power_per_mul.item()}")
                    # print_log(f"Corresponding indicators: ")
                    m.print_indicators(_binary=False)
                    # m.print_indicators(_binary=True)
                loss += power_per_mul * m.num_macs / self.total_macs # power per multiplication * (fixed latency, treat as 1) * #MACs of this layer / total #MACs
        return loss

    def compute_macs(self, dummy_input):
        # register hook
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                m.compute_macs()
        # Run a dummy forward pass to activate hooks
        self.eval()
        with torch.no_grad():
            _ = self.forward(dummy_input)
        # Print results
        self.total_macs = 0
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                if hasattr(m, "num_macs"):
                    self.total_macs += m.num_macs
                else:
                    raise ValueError(f"no num_macs attribute in {m}")
        print_log(f"Total MACs: {self.total_macs}")

    def switch_train_eval_mode(self, train_mode: bool):
        if train_mode:
            self.model.train()
            for m in self.model.modules():
                if isinstance(m, QuantLayer):
                    m.set_observers_status(enable_observers=True)
        else:
            self.model.eval()
            for m in self.model.modules():
                if isinstance(m, QuantLayer):
                    m.set_observers_status(enable_observers=False)

    def set_cali_appmult_mode(self, cali_appmult_mode: bool):
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                m.cali_appmult = cali_appmult_mode

    def prepare_post_training_quantization(self):
        self.set_app_state(weight_quant=True, act_quant=True, use_appmult=False)
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                m.weight_quantizer.use_new_observer('mse')
                m.act_quantizer.use_new_observer('mse')

    def prepare_quantization_aware_training(self):
        self.set_app_state(weight_quant=True, act_quant=True, use_appmult=False)
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                m.weight_quantizer.use_new_observer('learned_clipping')
                m.act_quantizer.use_new_observer('moving_average_learned_clipping')

    def prepare_trainappmult(self, use_homogeneous_appmult, num_max_discard_cols, num_init_discard_cols):
        assert num_max_discard_cols <= 2 * self.num_bits - 1 and num_max_discard_cols > 0, 'num_max_discard_cols must be in [1, 2*n_bits-1]'
        assert num_init_discard_cols <= num_max_discard_cols and num_init_discard_cols >= 0, 'num_init_discard_cols must be in [0, num_max_discard_cols]'
        if use_homogeneous_appmult:
            # self.gamma[i] denotes how "strongly" we want to discard the i-th column of partial products
            # 0 (keep the column) <= gamma <= 1 (discard whole column)
            init_value = torch.zeros((num_max_discard_cols,), dtype=torch.float32, device='cuda')
            assert num_init_discard_cols <= num_max_discard_cols, 'initial discard cols should not exceed max discard cols'
            init_value[:num_init_discard_cols] = 1.0
            self.gamma_free = torch.nn.Parameter(init_value)
        self.set_app_state(weight_quant=True, act_quant=True, use_appmult=True)
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                assert isinstance(m.weight_quantizer.observer, LearnedClippingObserver)
                assert isinstance(m.act_quantizer.observer, MovingAverageLearnedClippingObserver)
                if use_homogeneous_appmult:
                    m.add_trainable_indicators_homogeneous(self.gamma_free)
                else:
                    m.add_trainable_indicators()
                m.initialize_power_model()

    def set_first_layer_to_8bit(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                module_list += [m]
        module_list[0].weight_quantizer.bitwidth_refactor(8)
        module_list[0].act_quantizer.bitwidth_refactor(8)
        settings.LOGGER.info('set first layer to 8-bit')
        settings.LOGGER.info(f'first layer: {module_list[0]}')

    def set_first_and_last_layers_to_8bit(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                module_list += [m]
        module_list[0].weight_quantizer.bitwidth_refactor(8)
        module_list[0].act_quantizer.bitwidth_refactor(8)
        module_list[-1].weight_quantizer.bitwidth_refactor(8)
        module_list[-1].act_quantizer.bitwidth_refactor(8)
        settings.LOGGER.info('set first and last layer to 8-bit')

    def extra_repr(self):
        return self.model.extra_repr()