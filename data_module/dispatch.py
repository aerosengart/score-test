"""
Dispatch dictionaries for Data classes.

DATA REGIME: {
    'fx_int':        true/false  --- shared fixed intercept?
    'fx_targ_int':   true/false  --- target-specific fixed intercept?
    'rd_int':        true/false  --- random intercept?
    'fx_slope':      non-neg int --- number of shared fixed slopes
    'fx_targ_slope': non-neg int --- number of target-specific fixed slopes
    'rd_slope':      non-neg int --- number of random slopes
}
"""

lin_pred_types = {
    'fx_int_rd_int': {
        'fx_int':      True,
        'fx_targ_int': False,
        'rd_int':      True,
        'fx_slope':      0,
        'fx_targ_slope': 0,
        'rd_slope':      0,
    },
    'fx_targ_int_rd_slope': {
        'fx_int':      False,
        'fx_targ_int': True,
        'rd_int':      False,
        'fx_slope':      0,
        'fx_targ_slope': 0,
        'rd_slope':      1,
    },
    'fx_targ_int_fx_slope_rd_slope': {
        'fx_int':      False,
        'fx_targ_int': True,
        'rd_int':      False,
        'fx_slope':      1,
        'fx_targ_slope': 0,
        'rd_slope':      1,
    },
    'fx_targ_int_targ_slope_rd_slope': {
        'fx_int':      False,
        'fx_targ_int': True,
        'rd_int':      False,
        'fx_slope':      0,
        'fx_targ_slope': 1,
        'rd_slope':      1,
    }
}