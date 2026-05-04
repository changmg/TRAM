import torch
from torch.autograd import Function
import approx_ops
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm as tqdm


# global variables
# lut_appmult (torch.Tensor): lookup table for approximate multiplication
# lut_grad_a (torch.Tensor): lookup table for gradient of approximate multiplication w.r.t. the first operand
# lut_grad_b (torch.Tensor): lookup table for gradient of approximate multiplication w.r.t. the second operand
lut_appmult = None
lut_appmult_for_layer = None
lut_grad_a = None
lut_grad_b = None


def init_lookup_tables(file_name, quantization_bit):
    # lookup table parameters
    # must match the CUDA code settings, do not touch!!!!!!!!
    LUT_MAXVAL = 2**quantization_bit - 1
    if quantization_bit == 7:
        # LUT_MAXVAL = 127                               # maximum value of input operands 
        LUT_ROW_NUM = LUT_MAXVAL + 2                     # padding to mitigate bank conflict
        LUT_COL_NUM = LUT_MAXVAL + 3                     # padding to mitigate bank conflict
        LUT_ELEM_NUM = LUT_ROW_NUM * LUT_COL_NUM + 638   # padding for multiple-thread 4-element loading
    elif quantization_bit == 8:
        # LUT_MAXVAL = 255                               # maximum value of input operands 
        LUT_ROW_NUM = LUT_MAXVAL + 2                     # padding to mitigate bank conflict
        LUT_COL_NUM = LUT_MAXVAL + 3                     # padding to mitigate bank conflict
        LUT_ELEM_NUM = LUT_ROW_NUM * LUT_COL_NUM
    elif quantization_bit == 4:
        LUT_ROW_NUM = LUT_MAXVAL + 2                     # padding to mitigate bank conflict
        LUT_COL_NUM = LUT_MAXVAL + 3                     # padding to mitigate bank conflict
        LUT_ELEM_NUM = LUT_ROW_NUM * LUT_COL_NUM + 718   # padding for multiple-thread 4-element loading
    elif quantization_bit == 6:
        LUT_ROW_NUM = LUT_MAXVAL + 2                     # padding to mitigate bank conflict
        LUT_COL_NUM = LUT_MAXVAL + 3                     # padding to mitigate bank conflict
        LUT_ELEM_NUM = LUT_ROW_NUM * LUT_COL_NUM + 830   # padding for multiple-thread 4-element loading
    else:
        raise NotImplementedError(f'quantization_bit = {quantization_bit} is not supported')

    # load lookup tables from file
    print(f'Initializing lookup tables from {file_name}...')

    # global variables
    global lut_appmult, lut_grad_a, lut_grad_b

    # parse file
    state = 'init'
    lut_appmult = torch.zeros(LUT_ELEM_NUM, dtype=torch.uint16).cuda()
    lut_grad_a = torch.zeros(LUT_ELEM_NUM, dtype=torch.int16).cuda()
    lut_grad_b = torch.zeros(LUT_ELEM_NUM, dtype=torch.int16).cuda()
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('LUT for approximate multiplier:'):
                state = 'load_appmult'
                continue
            elif line.startswith('LUT for the gradient of the approximate multiplier w.r.t. the first operand:'):
                state = 'load_grad_a'
                continue
            elif line.startswith('LUT for the gradient of the approximate multiplier w.r.t. the second operand:'):
                state = 'load_grad_b'
                continue
            else:
                if state == 'init': # skip header
                    continue
            # parse lines; in each line, the first element is operand a, the second element is operand b, and the third element is the result
            a, b, res = line.split()
            a, b, res = int(a), int(b), int(res)
            if state == 'load_appmult':
                lut_appmult[a * LUT_COL_NUM + b] = res
            elif state == 'load_grad_a':
                lut_grad_a[a * LUT_COL_NUM + b] = res
            elif state == 'load_grad_b':
                lut_grad_b[a * LUT_COL_NUM + b] = res
            else:
                raise ValueError('Unknown state')


def init_lookup_tables_for_layer(file_name, quantization_bit, layer_id):
    # lookup table parameters
    # must match the CUDA code settings, do not touch!!!!!!!!
    LUT_MAXVAL = 2**quantization_bit - 1
    if quantization_bit == 7:
        # LUT_MAXVAL = 127                               # maximum value of input operands 
        LUT_ROW_NUM = LUT_MAXVAL + 2                     # padding to mitigate bank conflict
        LUT_COL_NUM = LUT_MAXVAL + 3                     # padding to mitigate bank conflict
        LUT_ELEM_NUM = LUT_ROW_NUM * LUT_COL_NUM + 638   # padding for multiple-thread 4-element loading
    elif quantization_bit == 8:
        # LUT_MAXVAL = 255                               # maximum value of input operands 
        LUT_ROW_NUM = LUT_MAXVAL + 2                     # padding to mitigate bank conflict
        LUT_COL_NUM = LUT_MAXVAL + 3                     # padding to mitigate bank conflict
        LUT_ELEM_NUM = LUT_ROW_NUM * LUT_COL_NUM
    elif quantization_bit == 4:
        LUT_ROW_NUM = LUT_MAXVAL + 2                     # padding to mitigate bank conflict
        LUT_COL_NUM = LUT_MAXVAL + 3                     # padding to mitigate bank conflict
        LUT_ELEM_NUM = LUT_ROW_NUM * LUT_COL_NUM + 718   # padding for multiple-thread 4-element loading
    elif quantization_bit == 6:
        LUT_ROW_NUM = LUT_MAXVAL + 2                     # padding to mitigate bank conflict
        LUT_COL_NUM = LUT_MAXVAL + 3                     # padding to mitigate bank conflict
        LUT_ELEM_NUM = LUT_ROW_NUM * LUT_COL_NUM + 830   # padding for multiple-thread 4-element loading
    else:
        raise NotImplementedError(f'quantization_bit = {quantization_bit} is not supported')

    # load lookup tables from file
    print(f'Initializing lookup tables from {file_name}...')

    # parse file
    state = 'init'
    lut_appmult_temp= torch.zeros(LUT_ELEM_NUM, dtype=torch.uint16).cuda()
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('LUT for approximate multiplier:'):
                state = 'load_appmult'
                continue
            elif line.startswith('LUT for the gradient of the approximate multiplier w.r.t. the first operand:'):
                state = 'load_grad_a'
                continue
            elif line.startswith('LUT for the gradient of the approximate multiplier w.r.t. the second operand:'):
                state = 'load_grad_b'
                continue
            else:
                if state == 'init': # skip header
                    continue
            # parse lines; in each line, the first element is operand a, the second element is operand b, and the third element is the result
            a, b, res = line.split()
            a, b, res = int(a), int(b), int(res)
            if state == 'load_appmult':
                lut_appmult_temp[a * LUT_COL_NUM + b] = res
            else:
                raise ValueError('Unknown state')

    # global variables
    global lut_appmult_for_layer # dict that maps layer_id to lut_appmult
    # store lut_appmult for the given layer_id
    if lut_appmult_for_layer is None:
        lut_appmult_for_layer = dict()
    assert layer_id not in lut_appmult_for_layer, f'lut_appmult for layer_id {layer_id} already exists'
    lut_appmult_for_layer[layer_id] = lut_appmult_temp.clone()
    # assert layer_id in lut_appmult_for_layer, f'failed to store lut_appmult for layer_id {layer_id}'


class AccGemm(Function):
    @staticmethod
    def forward(ctx, a, b):
        """gemm function forward.
        Args:
            a (torch.Tensor): [M, K]
            b (torch.Tensor): [K, N]
        
        Returns:
            c (torch.Tensor): [M, N]
        """
        # prepare input & output tensors 
        # a_cont, b_cont = a.contiguous(), b.contiguous()
        a_cont, b_cont = a.contiguous().to(torch.float32), b.contiguous().to(torch.float32)
        M, K, N = a.shape[0], a.shape[1], b.shape[1]
        # c = a.new_zeros(M, N).to(torch.uint32)
        c = a.new_zeros(M, N).to(torch.float32)

        # call gemm forward function
        approx_ops.acc_gemm_forward_fp32(a_cont, b_cont, c)

        # convert output dtype 
        c = c.to(torch.float32)

        # save for backward
        ctx.save_for_backward(a_cont, b_cont)
        ctx.ori_shape = (M, K, N)

        return c

    @staticmethod
    def backward(ctx, g_c):
        """gemm function backward.
        Args:
            g_c (torch.Tensor): [M, N], float32
        
        Returns:
            g_a (torch.Tensor): [M, K], float32
            g_b (torch.Tensor): [K, N], float32
        """
        # check input dtype
        assert g_c.dtype == torch.float32

        # get saved tensors
        a_cont, b_cont, = ctx.saved_tensors
        M, K, N = ctx.ori_shape

        # prepare output tensors
        g_a, g_b = g_c.new_zeros(M, K), g_c.new_zeros(K, N)

        # call gemm backward function
        g_a = torch.matmul(g_c, b_cont.t())
        g_b = torch.matmul(a_cont.t(), g_c)

        return g_a, g_b

acc_gemm_op = AccGemm.apply

class ApproxGemmBaseline(Function):
    @staticmethod
    def forward(ctx, a, b):
        """gemm function forward.
        Args:
            a (torch.Tensor): [M, K]
            b (torch.Tensor): [K, N]
        
        Returns:
            c (torch.Tensor): [M, N]
        """
        # save for backward
        ctx.save_for_backward(a, b)

        # check input dtype
        # assert a.dtype == torch.uint8 and b.dtype == torch.uint8
        a, b = a.to(torch.uint8), b.to(torch.uint8)

        # prepare input & output tensors 
        a_cont, b_cont = a.contiguous(), b.contiguous()
        M, K, N = a.shape[0], a.shape[1], b.shape[1]
        c = a.new_zeros(M, N).to(torch.uint32)

        # call gemm forward function
        global lut_appmult
        approx_ops.approx_gemm_forward(a_cont, b_cont, c, lut_appmult)

        # convert output dtype 
        c = c.to(torch.float32)

        return c

    @staticmethod
    def backward(ctx, g_c):
        """gemm function backward.
        Args:
            g_c (torch.Tensor): [M, N], float32
        
        Returns:
            g_a (torch.Tensor): [M, K], float32
            g_b (torch.Tensor): [K, N], float32
        """
        # check input dtype
        assert g_c.dtype == torch.float32

        # get saved tensors
        a, b, = ctx.saved_tensors

        # call straight-through estimator
        g_a = torch.matmul(g_c.contiguous(), b.t().contiguous())
        g_b = torch.matmul(a.t().contiguous(), g_c.contiguous())

        return g_a, g_b

approx_gemm_baseline_op = ApproxGemmBaseline.apply


def approx_linear_op(input, weight, bias):
    """approximate linear forward.
    Args:
        input (torch.Tensor): [batch_size, in_features]
        weight (torch.Tensor): [out_features, in_features]
        bias (torch.Tensor): [out_features]
    
    Returns:
        output (torch.Tensor): [batch_size, out_features]
    """
    assert bias is None, 'bias is not supported in approximate linear'
    output = approx_gemm_baseline_op(input, weight.t())
    return output


def approx_conv2d_baseline_op(input, weight, bias, stride, padding, dilation, groups):
    """conv2d function forward.
    Args:
        input (torch.Tensor): [batch_size, in_channels, in_height, in_width]
        weight (torch.Tensor): [out_channels, in_channels, kernel_height, kernel_width]
        bias (torch.Tensor): [out_channels]
        stride (int)
        padding (int)
        dilation (int)
        groups (int)
        
    Returns:
        output (torch.Tensor): [batch_size, out_channels, out_height, out_width]
    """
    # check
    assert bias is None, 'bias is not supported in approximate convolution'
    assert groups == 1, 'groups is not supported in approximate convolution'
    stride, padding, dilation = stride[0], padding[0], dilation[0]

    # prepare input & output tensors
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1

    # pad the input (zero padding)
    input_padded = F.pad(input, (padding, padding, padding, padding), mode='constant', value=0)        

    # Use unfold to create sliding windows
    input_unfolded = F.unfold(input_padded, kernel_size=(kernel_height, kernel_width), dilation=(dilation, dilation), stride=(stride, stride))

    # Reshape for matrix multiplication
    input_unfolded = input_unfolded.transpose(1, 2)  # Shape: (batch_size, num_windows, in_channels * kernel_height * kernel_width)
    input_unfolded = input_unfolded.reshape(-1, in_channels * kernel_height * kernel_width)  # Shape: (batch_size * num_windows, in_channels * kernel_height * kernel_width)
    weight_reshaped = weight.view(out_channels, -1).t()  # Shape: (in_channels * kernel_height * kernel_width, out_channels)

    # Perform the matrix multiplication
    output = approx_gemm_baseline_op(input_unfolded, weight_reshaped) # Shape: (batch_size * num_windows, out_channels)

    # Reshape the output to (batch_size, out_channels, out_height, out_width)
    output = output.view(batch_size, out_height * out_width, out_channels).transpose(1, 2)  # Shape: (batch_size, out_channels, num_windows)
    output = output.view(batch_size, out_channels, out_height, out_width)  # Shape: (batch_size, out_channels, out_height, out_width)

    return output


class ApproxGemmBaselineForLayer(Function):
    @staticmethod
    def forward(ctx, a, b, layer_id):
        """gemm function forward.
        Args:
            a (torch.Tensor): [M, K]
            b (torch.Tensor): [K, N]
        
        Returns:
            c (torch.Tensor): [M, N]
        """
        # save for backward
        ctx.save_for_backward(a, b)

        # check input dtype
        # assert a.dtype == torch.uint8 and b.dtype == torch.uint8
        a, b = a.to(torch.uint8), b.to(torch.uint8)

        # prepare input & output tensors 
        a_cont, b_cont = a.contiguous(), b.contiguous()
        M, K, N = a.shape[0], a.shape[1], b.shape[1]
        c = a.new_zeros(M, N).to(torch.uint32)

        # call gemm forward function
        global lut_appmult_for_layer
        if layer_id not in lut_appmult_for_layer:
            raise ValueError(f'lut_appmult for layer_id {layer_id} does not exist')
        approx_ops.approx_gemm_forward(a_cont, b_cont, c, lut_appmult_for_layer[layer_id])

        # convert output dtype 
        c = c.to(torch.float32)

        return c

    @staticmethod
    def backward(ctx, g_c):
        """gemm function backward.
        Args:
            g_c (torch.Tensor): [M, N], float32
        
        Returns:
            g_a (torch.Tensor): [M, K], float32
            g_b (torch.Tensor): [K, N], float32
        """
        # check input dtype
        assert g_c.dtype == torch.float32

        # get saved tensors
        a, b, = ctx.saved_tensors

        # call straight-through estimator
        g_a = torch.matmul(g_c.contiguous(), b.t().contiguous())
        g_b = torch.matmul(a.t().contiguous(), g_c.contiguous())

        return g_a, g_b, None

approx_gemm_baseline_op_for_layer = ApproxGemmBaselineForLayer.apply


def approx_conv2d_baseline_op_for_layer(input, weight, bias, stride, padding, dilation, groups, layer_id):
    """conv2d function forward.
    Args:
        input (torch.Tensor): [batch_size, in_channels, in_height, in_width]
        weight (torch.Tensor): [out_channels, in_channels, kernel_height, kernel_width]
        bias (torch.Tensor): [out_channels]
        stride (int)
        padding (int)
        dilation (int)
        groups (int)
        
    Returns:
        output (torch.Tensor): [batch_size, out_channels, out_height, out_width]
    """
    # check
    assert bias is None, 'bias is not supported in approximate convolution'
    assert groups == 1, 'groups is not supported in approximate convolution'
    stride, padding, dilation = stride[0], padding[0], dilation[0]

    # prepare input & output tensors
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride + 1
    out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1

    # pad the input (zero padding)
    input_padded = F.pad(input, (padding, padding, padding, padding), mode='constant', value=0)        

    # Use unfold to create sliding windows
    input_unfolded = F.unfold(input_padded, kernel_size=(kernel_height, kernel_width), dilation=(dilation, dilation), stride=(stride, stride))

    # Reshape for matrix multiplication
    input_unfolded = input_unfolded.transpose(1, 2)  # Shape: (batch_size, num_windows, in_channels * kernel_height * kernel_width)
    input_unfolded = input_unfolded.reshape(-1, in_channels * kernel_height * kernel_width)  # Shape: (batch_size * num_windows, in_channels * kernel_height * kernel_width)
    weight_reshaped = weight.view(out_channels, -1).t()  # Shape: (in_channels * kernel_height * kernel_width, out_channels)

    # Perform the matrix multiplication
    output = approx_gemm_baseline_op_for_layer(input_unfolded, weight_reshaped, layer_id) # Shape: (batch_size * num_windows, out_channels)

    # Reshape the output to (batch_size, out_channels, out_height, out_width)
    output = output.view(batch_size, out_height * out_width, out_channels).transpose(1, 2)  # Shape: (batch_size, out_channels, num_windows)
    output = output.view(batch_size, out_channels, out_height, out_width)  # Shape: (batch_size, out_channels, out_height, out_width)

    return output