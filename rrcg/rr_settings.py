from gpytorch.settings import _value_context, _feature_flag


class max_rr_cg_iter_list(_value_context):
    """
    J in rr-cg estimation.
    If it is used, then we are not sampling J from rr-distribution
    """
    _global_value = [1000]


class rr_cg_nsamples(_value_context):
    _global_value = 2 # for the purpose of backprop


class use_rr_cg(_feature_flag):
    _default = False


class use_prespecified_rr_iter(_feature_flag):
    _default = False


class use_rr_lanczos(_feature_flag):
    _default = False
