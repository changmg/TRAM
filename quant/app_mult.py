import torch


def get_bit(x: torch.Tensor, mask: int) -> torch.Tensor:
    return torch.bitwise_and(x.int(), mask).float()

    
# def get_ith_bit(x: torch.Tensor, i: int) -> torch.Tensor:
#     return get_bit(x, 1 << i)
def get_ith_bit(x: torch.Tensor, i: int) -> torch.Tensor:
    return (x.int() & (1 << i)).float()
# def get_ith_bit(x: torch.Tensor, i: int) -> torch.Tensor:
#     # no extra mask tensor, no extra int() call each time
#     # x is already integer (e.g., int8 or int32)
#     return (x & (1 << i)).to(torch.bool)

    
def get_last_n_bits(x: torch.Tensor, n: int) -> torch.Tensor:
    return get_bit(x, (1 << n) - 1)

    
def error_of_discard_1_columns(act_scale: torch.Tensor, weight_scale: torch.Tensor, input_int: torch.Tensor, weight_int: torch.Tensor, fwd_func: torch.nn.functional, fwd_kwargs: dict) -> torch.Tensor:
    """
    error computation (discard 1 column)
    """
    w0 = get_ith_bit(weight_int, i=0)
    x0 = get_last_n_bits(input_int, n=1)
    _as, _ws = act_scale, weight_scale
    error = fwd_func(_as * x0,   _ws * w0, bias=None, **fwd_kwargs)
    return error


def error_of_discard_2_columns(act_scale: torch.Tensor, weight_scale: torch.Tensor, input_int: torch.Tensor, weight_int: torch.Tensor, fwd_func: torch.nn.functional, fwd_kwargs: dict) -> torch.Tensor:
    """
    error computation (discard 2 columns)
    """
    w0, w1 = get_ith_bit(weight_int, i=0), get_ith_bit(weight_int, i=1)
    x1_0, x0 = get_last_n_bits(input_int, n=2), get_last_n_bits(input_int, n=1)
    _as, _ws = act_scale, weight_scale
    t0 = fwd_func(_as * x1_0, _ws * w0, bias=None, **fwd_kwargs)
    t1 = fwd_func(_as * x0,   _ws * w1, bias=None, **fwd_kwargs)
    error = t0 + t1
    # error = t0 + t1 - 1.25
    return error


def int_error_of_discard_2_columns(input_int: torch.Tensor, weight_int: torch.Tensor, fwd_func: torch.nn.functional, fwd_kwargs: dict) -> torch.Tensor:
    """
    error computation (discard 2 columns)
    """
    w0, w1 = get_ith_bit(weight_int, i=0), get_ith_bit(weight_int, i=1)
    x1_0, x0 = get_last_n_bits(input_int, n=2), get_last_n_bits(input_int, n=1)
    t0 = fwd_func(x1_0, w0, bias=None, **fwd_kwargs)
    t1 = fwd_func(x0,   w1, bias=None, **fwd_kwargs)
    error = t0 + t1
    return error


def error_of_discard_3_columns(act_scale: torch.Tensor, weight_scale: torch.Tensor, input_int: torch.Tensor, weight_int: torch.Tensor, fwd_func: torch.nn.functional, fwd_kwargs: dict) -> torch.Tensor:
    """
    error computation (discard 3 columns)
    """
    w0, w1, w2 = get_ith_bit(weight_int, i=0), get_ith_bit(weight_int, i=1), get_ith_bit(weight_int, i=2)
    x2_0, x1_0, x0 = get_last_n_bits(input_int, n=3), get_last_n_bits(input_int, n=2), get_last_n_bits(input_int, n=1)
    _as, _ws = act_scale, weight_scale
    t0 = fwd_func(_as * x2_0, _ws * w0, bias=None, **fwd_kwargs)
    t1 = fwd_func(_as * x1_0, _ws * w1, bias=None, **fwd_kwargs)
    t2 = fwd_func(_as * x0,   _ws * w2, bias=None, **fwd_kwargs)
    error = t0 + t1 + t2
    return error


def error_of_discard_4_columns(act_scale: torch.Tensor, weight_scale: torch.Tensor, input_int: torch.Tensor, weight_int: torch.Tensor, fwd_func: torch.nn.functional, fwd_kwargs: dict) -> torch.Tensor:
    """
    error computation (discard 4 columns)
    """
    w0, w1, w2, w3 = get_ith_bit(weight_int, i=0), get_ith_bit(weight_int, i=1), get_ith_bit(weight_int, i=2), get_ith_bit(weight_int, i=3)
    x3_0, x2_0, x1_0, x0 = get_last_n_bits(input_int, n=4), get_last_n_bits(input_int, n=3), get_last_n_bits(input_int, n=2), get_last_n_bits(input_int, n=1)
    _as, _ws = act_scale, weight_scale
    t0 = fwd_func(_as * x3_0, _ws * w0, bias=None, **fwd_kwargs)
    t1 = fwd_func(_as * x2_0, _ws * w1, bias=None, **fwd_kwargs)
    t2 = fwd_func(_as * x1_0, _ws * w2, bias=None, **fwd_kwargs)
    t3 = fwd_func(_as * x0,   _ws * w3, bias=None, **fwd_kwargs)
    error = t0 + t1 + t2 + t3
    return error


def int_error_of_discard_4_columns(input_int: torch.Tensor, weight_int: torch.Tensor, fwd_func: torch.nn.functional, fwd_kwargs: dict) -> torch.Tensor:
    """
    error computation (discard 4 columns)
    """
    w0, w1, w2, w3 = get_ith_bit(weight_int, i=0), get_ith_bit(weight_int, i=1), get_ith_bit(weight_int, i=2), get_ith_bit(weight_int, i=3)
    x3_0, x2_0, x1_0, x0 = get_last_n_bits(input_int, n=4), get_last_n_bits(input_int, n=3), get_last_n_bits(input_int, n=2), get_last_n_bits(input_int, n=1)
    t0 = fwd_func(x3_0, w0, bias=None, **fwd_kwargs)
    t1 = fwd_func(x2_0, w1, bias=None, **fwd_kwargs)
    t2 = fwd_func(x1_0, w2, bias=None, **fwd_kwargs)
    t3 = fwd_func(x0  , w3, bias=None, **fwd_kwargs)
    error = t0 + t1 + t2 + t3
    return error


def error_of_discard_5_columns(act_scale: torch.Tensor, weight_scale: torch.Tensor, input_int: torch.Tensor, weight_int: torch.Tensor, fwd_func: torch.nn.functional, fwd_kwargs: dict) -> torch.Tensor:
    """
    error computation (discard 5 columns)
    """
    w0, w1, w2, w3, w4 = get_ith_bit(weight_int, i=0), get_ith_bit(weight_int, i=1), get_ith_bit(weight_int, i=2), get_ith_bit(weight_int, i=3), get_ith_bit(weight_int, i=4)
    x4_0, x3_0, x2_0, x1_0, x0 = get_last_n_bits(input_int, n=5), get_last_n_bits(input_int, n=4), get_last_n_bits(input_int, n=3), get_last_n_bits(input_int, n=2), get_last_n_bits(input_int, n=1)
    _as, _ws = act_scale, weight_scale
    t0 = fwd_func(_as * x4_0, _ws * w0, bias=None, **fwd_kwargs)
    t1 = fwd_func(_as * x3_0, _ws * w1, bias=None, **fwd_kwargs)
    t2 = fwd_func(_as * x2_0, _ws * w2, bias=None, **fwd_kwargs)
    t3 = fwd_func(_as * x1_0, _ws * w3, bias=None, **fwd_kwargs)
    t4 = fwd_func(_as * x0,   _ws * w4, bias=None, **fwd_kwargs)
    error = t0 + t1 + t2 + t3 + t4
    return error


def error_of_discard_6_columns(act_scale: torch.Tensor, weight_scale: torch.Tensor, input_int: torch.Tensor, weight_int: torch.Tensor, fwd_func: torch.nn.functional, fwd_kwargs: dict) -> torch.Tensor:
    """
    error computation (discard 6 columns)
    """
    w0, w1, w2, w3, w4, w5 = get_ith_bit(weight_int, i=0), get_ith_bit(weight_int, i=1), get_ith_bit(weight_int, i=2), get_ith_bit(weight_int, i=3), get_ith_bit(weight_int, i=4), get_ith_bit(weight_int, i=5)
    x5_0, x4_0, x3_0, x2_0, x1_0, x0 = get_last_n_bits(input_int, n=6), get_last_n_bits(input_int, n=5), get_last_n_bits(input_int, n=4), get_last_n_bits(input_int, n=3), get_last_n_bits(input_int, n=2), get_last_n_bits(input_int, n=1)
    _as, _ws = act_scale, weight_scale
    t0 = fwd_func(_as * x5_0, _ws * w0, bias=None, **fwd_kwargs)
    t1 = fwd_func(_as * x4_0, _ws * w1, bias=None, **fwd_kwargs)
    t2 = fwd_func(_as * x3_0, _ws * w2, bias=None, **fwd_kwargs)
    t3 = fwd_func(_as * x2_0, _ws * w3, bias=None, **fwd_kwargs)
    t4 = fwd_func(_as * x1_0, _ws * w4, bias=None, **fwd_kwargs)
    t5 = fwd_func(_as * x0  , _ws * w5 , bias=None , **fwd_kwargs)
    error = t0 + t1 + t2 + t3 + t4 + t5
    return error


def int_error_of_discard_6_columns(input_int: torch.Tensor, weight_int: torch.Tensor, fwd_func: torch.nn.functional, fwd_kwargs: dict) -> torch.Tensor:
    """
    error computation (discard 6 columns)
    """
    w0, w1, w2, w3, w4, w5 = get_ith_bit(weight_int, i=0), get_ith_bit(weight_int, i=1), get_ith_bit(weight_int, i=2), get_ith_bit(weight_int, i=3), get_ith_bit(weight_int, i=4), get_ith_bit(weight_int, i=5)
    x5_0, x4_0, x3_0, x2_0, x1_0, x0 = get_last_n_bits(input_int, n=6), get_last_n_bits(input_int, n=5), get_last_n_bits(input_int, n=4), get_last_n_bits(input_int, n=3), get_last_n_bits(input_int, n=2), get_last_n_bits(input_int, n=1)
    t0 = fwd_func(x5_0, w0 , bias=None , **fwd_kwargs)
    t1 = fwd_func(x4_0, w1 , bias=None , **fwd_kwargs)
    t2 = fwd_func(x3_0, w2 , bias=None , **fwd_kwargs)
    t3 = fwd_func(x2_0, w3 , bias=None , **fwd_kwargs)
    t4 = fwd_func(x1_0, w4 , bias=None , **fwd_kwargs)
    t5 = fwd_func(x0  , w5 , bias=None , **fwd_kwargs)
    error = t0 + t1 + t2 + t3 + t4 + t5
    return error


def error_of_discard_7_columns(act_scale: torch.Tensor, weight_scale: torch.Tensor, input_int: torch.Tensor, weight_int: torch.Tensor, fwd_func: torch.nn.functional, fwd_kwargs: dict) -> torch.Tensor:
    """
    error computation (discard 7 columns)
    """
    w0, w1, w2, w3, w4, w5, w6 = get_ith_bit(weight_int, i=0), get_ith_bit(weight_int, i=1), get_ith_bit(weight_int, i=2), get_ith_bit(weight_int, i=3), get_ith_bit(weight_int, i=4), get_ith_bit(weight_int, i=5), get_ith_bit(weight_int, i=6)
    x6_0, x5_0, x4_0, x3_0, x2_0, x1_0, x0 = get_last_n_bits(input_int, n=7), get_last_n_bits(input_int, n=6), get_last_n_bits(input_int, n=5), get_last_n_bits(input_int, n=4), get_last_n_bits(input_int, n=3), get_last_n_bits(input_int, n=2), get_last_n_bits(input_int, n=1)
    _as, _ws = act_scale, weight_scale
    t0 = fwd_func(_as * x6_0, _ws * w0 , bias=None , **fwd_kwargs)
    t1 = fwd_func(_as * x5_0, _ws * w1 , bias=None , **fwd_kwargs)
    t2 = fwd_func(_as * x4_0, _ws * w2 , bias=None , **fwd_kwargs)
    t3 = fwd_func(_as * x3_0, _ws * w3 , bias=None , **fwd_kwargs)
    t4 = fwd_func(_as * x2_0, _ws * w4 , bias=None , **fwd_kwargs)
    t5 = fwd_func(_as * x1_0, _ws * w5 , bias=None , **fwd_kwargs)
    t6 = fwd_func(_as * x0  , _ws * w6 , bias=None , **fwd_kwargs)
    error = t0 + t1 + t2 + t3 + t4 + t5 + t6
    return error


def error_of_discard_8_columns(act_scale: torch.Tensor, weight_scale: torch.Tensor, input_int: torch.Tensor, weight_int: torch.Tensor, fwd_func: torch.nn.functional, fwd_kwargs: dict) -> torch.Tensor:
    """
    error computation (discard 8 columns)
    """
    w0, w1, w2, w3, w4, w5, w6, w7 = get_ith_bit(weight_int, i=0), get_ith_bit(weight_int, i=1), get_ith_bit(weight_int, i=2), get_ith_bit(weight_int, i=3), get_ith_bit(weight_int, i=4), get_ith_bit(weight_int, i=5), get_ith_bit(weight_int, i=6), get_ith_bit(weight_int, i=7)
    x7_0, x6_0, x5_0, x4_0, x3_0, x2_0, x1_0, x0 = get_last_n_bits(input_int, n=8), get_last_n_bits(input_int, n=7), get_last_n_bits(input_int, n=6), get_last_n_bits(input_int, n=5), get_last_n_bits(input_int, n=4), get_last_n_bits(input_int, n=3), get_last_n_bits(input_int, n=2), get_last_n_bits(input_int, n=1)
    _as, _ws = act_scale, weight_scale
    t0 = fwd_func(_as * x7_0 , _ws * w0 , bias=None , **fwd_kwargs)
    t1 = fwd_func(_as * x6_0 , _ws * w1 , bias=None , **fwd_kwargs)
    t2 = fwd_func(_as * x5_0 , _ws * w2 , bias=None , **fwd_kwargs)
    t3 = fwd_func(_as * x4_0 , _ws * w3 , bias=None , **fwd_kwargs)
    t4 = fwd_func(_as * x3_0 , _ws * w4 , bias=None , **fwd_kwargs)
    t5 = fwd_func(_as * x2_0 , _ws * w5 , bias=None , **fwd_kwargs)
    t6 = fwd_func(_as * x1_0 , _ws * w6 , bias=None , **fwd_kwargs)
    t7 = fwd_func(_as * x0   , _ws * w7 , bias=None , **fwd_kwargs)
    error = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7
    return error


def int_error_of_discard_8_columns(input_int: torch.Tensor, weight_int: torch.Tensor, fwd_func: torch.nn.functional, fwd_kwargs: dict) -> torch.Tensor:
    """
    error computation (discard 8 columns)
    """
    w0, w1, w2, w3, w4, w5, w6, w7 = get_ith_bit(weight_int, i=0), get_ith_bit(weight_int, i=1), get_ith_bit(weight_int, i=2), get_ith_bit(weight_int, i=3), get_ith_bit(weight_int, i=4), get_ith_bit(weight_int, i=5), get_ith_bit(weight_int, i=6), get_ith_bit(weight_int, i=7)
    x7_0, x6_0, x5_0, x4_0, x3_0, x2_0, x1_0, x0 = get_last_n_bits(input_int, n=8), get_last_n_bits(input_int, n=7), get_last_n_bits(input_int, n=6), get_last_n_bits(input_int, n=5), get_last_n_bits(input_int, n=4), get_last_n_bits(input_int, n=3), get_last_n_bits(input_int, n=2), get_last_n_bits(input_int, n=1)
    t0 = fwd_func(x7_0 , w0 , bias=None , **fwd_kwargs)
    t1 = fwd_func(x6_0 , w1 , bias=None , **fwd_kwargs)
    t2 = fwd_func(x5_0 , w2 , bias=None , **fwd_kwargs)
    t3 = fwd_func(x4_0 , w3 , bias=None , **fwd_kwargs)
    t4 = fwd_func(x3_0 , w4 , bias=None , **fwd_kwargs)
    t5 = fwd_func(x2_0 , w5 , bias=None , **fwd_kwargs)
    t6 = fwd_func(x1_0 , w6 , bias=None , **fwd_kwargs)
    t7 = fwd_func(x0   , w7 , bias=None , **fwd_kwargs)
    error = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7
    return error