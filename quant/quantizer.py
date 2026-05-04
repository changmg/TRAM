import torch
import math

from torch import nn

from utils.common import tensor_to_str
from conf import settings
print_log = settings.LOGGER.info


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()

    
def fake_quant(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, n_levels: int):
    """
    Fake quantization function.
    """
    x_int = round_ste(x / scale) + zero_point
    x_int = torch.clamp(x_int, 0, n_levels - 1)
    x_float_q = (x_int - zero_point) * scale
    return x_int, x_float_q


class ObserverBase(nn.Module):
    """
    Base class for observer.
    Assume that asymmetric quantization is used.
    """
    def __init__(self, n_bits: int, channel_wise: bool, data_shape: tuple):
        super(ObserverBase, self).__init__()
        self.register_buffer('n_bits', torch.tensor(n_bits, dtype=torch.int))
        self.register_buffer('n_levels', torch.tensor(2 ** self.n_bits.item(), dtype=torch.int))
        self.register_buffer('channel_wise', torch.tensor(channel_wise, dtype=torch.bool))
        self.register_buffer('eps', torch.tensor(torch.finfo(torch.float32).eps, dtype=torch.float32))
        self.register_buffer("min_val", torch.zeros(data_shape, dtype=torch.float32))
        self.register_buffer("max_val", torch.zeros(data_shape, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        raise NotImplementedError
    
    def get_scale_zp(self, max: torch.Tensor, min: torch.Tensor):
        """
        Get scale and zero point for quantization.
        """
        scale = (max - min) / (self.n_levels - 1)
        scale = torch.max(scale, self.eps) # avoid zero division
        zero_point = (-min / scale).round()
        return scale.detach(), zero_point.detach()

    def get_max_min(self, x: torch.Tensor):
        if self.channel_wise: # channel-wise quantization can be only applied to weights
            if x.dim() == 4: # for conv2d, weight format: out_channels, in_channels, kernel_size, kernel_size
                # Compute min/max across in_channels and kernel dimensions
                max_val = x.amax(dim=(1, 2, 3))  # [out_c]
                min_val = x.amin(dim=(1, 2, 3))  # [out_c]
                # Reshape for broadcasting: [out_c, 1, 1, 1]
                max_val = max_val.view(-1, 1, 1, 1)
                min_val = min_val.view(-1, 1, 1, 1)
            elif x.dim() == 2: # for linear, weight format: out_features, in_features
                # Compute min/max across input features
                max_val = x.amax(dim=1)  # [out_f]
                min_val = x.amin(dim=1)  # [out_f]
                # Reshape for broadcasting: [out_f, 1]
                max_val = max_val.view(-1, 1)
                min_val = min_val.view(-1, 1)
            else:
                raise ValueError('Unsupported input shape')
        else:
            max_val = x.max()
            min_val = x.min()
        return max_val, min_val

    def extra_repr(self):
        return 'n_bits={}, n_levels={}, channel_wise={}, data_shape={}'.format(
            self.n_bits.item(), self.n_levels.item(), self.channel_wise.item(), self.min_val.shape)


class MSEObserver(ObserverBase):
    """
    Mean Square Error observer.
    """
    def __init__(self, org_observer: ObserverBase):
        super(MSEObserver, self).__init__(n_bits=org_observer.n_bits.item(), channel_wise=org_observer.channel_wise.item(), data_shape=org_observer.min_val.shape)
        # inherit the initial value from org_observer
        self.max_val = org_observer.max_val
        self.min_val = org_observer.min_val
        self.best_clip_factor = 1.0

    def forward(self, x: torch.Tensor):
        self.max_val, self.min_val = self.get_max_min(x)
        best_score = 1e+10
        for i in range(80):
            new_max = self.max_val * (1.0 - (i * 0.01))
            new_min = self.min_val * (1.0 - (i * 0.01))
            new_scale, new_zero_point = self.get_scale_zp(new_max, new_min)
            _, x_q = fake_quant(x, new_scale, new_zero_point, self.n_levels)
            # L_p norm minimization as described in LAPQ
            # https://arxiv.org/abs/1911.07190
            score = lp_loss(x, x_q, p=2.4, reduction='all')
            if score < best_score:
                best_score = score
                self.best_clip_factor = 1.0 - (i * 0.01)
                scale, zero_point = new_scale, new_zero_point
        return scale.detach(), zero_point.detach()

    def extra_repr(self):
        base_str = super(MSEObserver, self).extra_repr()
        return f'{base_str}, max_val={tensor_to_str(self.max_val)}, min_val={tensor_to_str(self.min_val)}, best_clip_factor={self.best_clip_factor}'


class MinMaxObserver(ObserverBase):
    """
    Min-Max observer.
    """
    def __init__(self, n_bits: int, channel_wise: bool, data_shape: tuple):
        super(MinMaxObserver, self).__init__(n_bits=n_bits, channel_wise=channel_wise, data_shape=data_shape)

    def forward(self, x: torch.Tensor):
        self.max_val, self.min_val = self.get_max_min(x)
        return self.get_scale_zp(self.max_val, self.min_val)


class MovingAverageMinMaxObserver(ObserverBase):
    """
    Moving Average Min-Max observer (for retraining)
    """
    # def __init__(self, n_bits: int, channel_wise: bool, data_shape: tuple, momentum: float=0.1):
    #     super(MovingAverageMinMaxObserver, self).__init__(n_bits=n_bits, channel_wise=channel_wise, data_shape=data_shape)
    #     self.momentum = momentum
    #     self.num_flag = 0

    def __init__(self, org_observer: ObserverBase, momentum: float=0.1):
        super(MovingAverageMinMaxObserver, self).__init__(n_bits=org_observer.n_bits.item(), channel_wise=org_observer.channel_wise.item(), data_shape=org_observer.min_val.shape)
        self.momentum = momentum
        # inherit the initial value from org_observer
        self.num_flag = 1
        self.max_val = org_observer.max_val
        self.min_val = org_observer.min_val

    def forward(self, x: torch.Tensor):
        max_val_cur, min_val_cur = self.get_max_min(x)
        # record the moving average of min/max values
        if self.num_flag == 0:
            self.num_flag += 1
            self.max_val, self.min_val = max_val_cur, min_val_cur
        else:
            self.max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
            self.min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
        return self.get_scale_zp(self.max_val, self.min_val)


class LearnedClippingObserver(ObserverBase):
    """
    Learned clipping observer.
    Reference: OmniQuant: Omnidirectionally calibrated quantization for large language models, ICLR 2024
    """
    def __init__(self, org_observer: ObserverBase):
        super(LearnedClippingObserver, self).__init__(n_bits=org_observer.n_bits.item(), channel_wise=org_observer.channel_wise.item(), data_shape=org_observer.min_val.shape)
        self.init_const()
        # inherit the initial value from org_observer
        self.max_val = org_observer.max_val
        self.min_val = org_observer.min_val
        if isinstance(org_observer, MSEObserver):
            init_factor = org_observer.best_clip_factor
            assert init_factor <= 1.0 and init_factor > 0.0
            init_value = math.log(init_factor / (1 - init_factor + self.EPS))
        else:
            init_value = 0.0
        self.upper_bound_factor = nn.Parameter(torch.ones(org_observer.min_val.shape, dtype=torch.float32) * init_value)
        self.lower_bound_factor = nn.Parameter(torch.ones(org_observer.min_val.shape, dtype=torch.float32) * init_value)
        
    def init_const(self):
        self.CLIP_MIN = 1e-5
        self.CLIP_MAX = 1e4
        self.EPS = 1e-5

    def forward(self, x: torch.Tensor):
        self.max_val, self.min_val = self.get_max_min(x)
        new_max = torch.sigmoid(self.upper_bound_factor) * self.max_val
        new_min = torch.sigmoid(self.lower_bound_factor) * self.min_val
        scale = (new_max - new_min) / (self.n_levels - 1)
        scale = scale.clamp(min=self.CLIP_MIN, max=self.CLIP_MAX)
        zero_point = -new_min / scale
        zero_point = zero_point.clamp(min=-self.CLIP_MAX, max=self.CLIP_MAX).round()
        return scale, zero_point

    def extra_repr(self):
        base_str = super(LearnedClippingObserver, self).extra_repr()
        return f'{base_str}, lower_bound_factor={tensor_to_str(self.lower_bound_factor)}, upper_bound_factor={tensor_to_str(self.upper_bound_factor)}'


class MovingAverageLearnedClippingObserver(ObserverBase):
    """
    Moving Average Learned Clipping observer (for retraining)
    """
    def __init__(self, org_observer: ObserverBase, momentum: float=0.1):
        super(MovingAverageLearnedClippingObserver, self).__init__(n_bits=org_observer.n_bits.item(), channel_wise=org_observer.channel_wise.item(), data_shape=org_observer.min_val.shape)
        self.init_const()
        self.momentum = momentum
        self.num_flag = 1
        # inherit the initial value from org_observer
        self.max_val = org_observer.max_val
        self.min_val = org_observer.min_val
        if isinstance(org_observer, MSEObserver):
            init_factor = org_observer.best_clip_factor
            assert init_factor <= 1.0 and init_factor > 0.0
            init_value = math.log(init_factor / (1 - init_factor + self.EPS))
        else:
            init_value = 0.0
        self.upper_bound_factor = nn.Parameter(torch.ones(org_observer.min_val.shape, dtype=torch.float32) * init_value)
        self.lower_bound_factor = nn.Parameter(torch.ones(org_observer.min_val.shape, dtype=torch.float32) * init_value)

    def init_const(self):
        self.CLIP_MIN = 1e-5
        self.CLIP_MAX = 1e4
        self.EPS = 1e-5

    def forward(self, x: torch.Tensor):
        # record the moving average of min/max values (without gradient)
        with torch.no_grad():
            max_val_cur, min_val_cur = self.get_max_min(x)
            if self.num_flag == 0:
                self.num_flag += 1
                self.max_val, self.min_val = max_val_cur, min_val_cur
            else:
                self.max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
                self.min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
        # compute dynamic range
        new_max = torch.sigmoid(self.upper_bound_factor) * self.max_val
        new_min = torch.sigmoid(self.lower_bound_factor) * self.min_val
        # compute scale and zero point
        scale = (new_max - new_min) / (self.n_levels - 1)
        scale = scale.clamp(min=self.CLIP_MIN, max=self.CLIP_MAX)
        zero_point = -new_min / scale
        zero_point = zero_point.clamp(min=-self.CLIP_MAX, max=self.CLIP_MAX).round()
        return scale, zero_point

    def extra_repr(self):
        base_str = super(MovingAverageLearnedClippingObserver, self).extra_repr()
        return f'{base_str}, lower_bound_factor={tensor_to_str(self.lower_bound_factor)}, upper_bound_factor={tensor_to_str(self.upper_bound_factor)}'


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int, channel_wise: bool, quantizer_shape: tuple, observer: str):
        super(UniformAffineQuantizer, self).__init__()
        self.register_buffer('scale', torch.zeros(quantizer_shape, dtype=torch.float32))
        self.register_buffer('zero_point', torch.zeros(quantizer_shape, dtype=torch.float32))
        self.enable_observers = False
        if observer == 'minmax':
            self.observer = MinMaxObserver(n_bits=n_bits, channel_wise=channel_wise, data_shape=quantizer_shape)
        elif observer == 'mse':
            self.observer = MSEObserver(n_bits=n_bits, channel_wise=channel_wise, data_shape=quantizer_shape)
        elif observer == 'moving_average_minmax':
            self.observer = MovingAverageMinMaxObserver(n_bits=n_bits, channel_wise=channel_wise, data_shape=quantizer_shape)
        else:
            raise ValueError('Unknown observer: {}'.format(observer))

    def forward(self, x: torch.Tensor):
        if self.enable_observers:
            self.scale, self.zero_point = self.observer.forward(x)
        return fake_quant(x, self.scale, self.zero_point, self.observer.n_levels)

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.observer.n_bits = torch.tensor(refactored_bit, dtype=torch.int)
        self.observer.n_levels = torch.tensor(2 ** refactored_bit, dtype=torch.int) 

    def use_new_observer(self, observer_str: str):
        if observer_str == 'minmax':
            if not isinstance(self.observer, MinMaxObserver):
                self.observer = MinMaxObserver(org_observer=self.observer)
        elif observer_str == 'mse':
            if not isinstance(self.observer, MSEObserver):
                self.observer = MSEObserver(org_observer=self.observer)
        elif observer_str == 'moving_average_minmax':
            if not isinstance(self.observer, MovingAverageMinMaxObserver):
                self.observer = MovingAverageMinMaxObserver(org_observer=self.observer)
        elif observer_str == 'learned_clipping':
            if not isinstance(self.observer, LearnedClippingObserver):
                self.observer = LearnedClippingObserver(org_observer=self.observer)
        elif observer_str == 'moving_average_learned_clipping':
            if not isinstance(self.observer, MovingAverageLearnedClippingObserver):
                self.observer = MovingAverageLearnedClippingObserver(org_observer=self.observer)
        else:
            raise ValueError('Unknown observer: {}'.format(observer_str))

    def set_observers_status(self, enable_observers: bool):
        self.enable_observers = enable_observers

    def extra_repr(self):
        return f'enable_observers={self.enable_observers}, scale={tensor_to_str(self.scale)}, zero_point={tensor_to_str(self.zero_point)}'