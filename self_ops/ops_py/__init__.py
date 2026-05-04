from .approx_ops import \
    init_lookup_tables, \
    init_lookup_tables_for_layer, \
    approx_linear_op, \
    approx_conv2d_baseline_op, \
    approx_conv2d_baseline_op_for_layer
__all__ = ['init_lookup_tables', 
           'init_lookup_tables_for_layer',
           'approx_linear_op',
           'approx_conv2d_baseline_op',
           'approx_conv2d_baseline_op_for_layer'
        ]