'''
General grab bag of helper functions
'''

import numpy as np
import os

from .settings import DEFAULT_AWRAL_MASK
from .precision import iround, quantize, aquantize, sanitize_cell, sanitize_geo_array

def load_mask(fn_mask=DEFAULT_AWRAL_MASK):
    '''
    Identify the AWAP cells required for the continental AWRA-L run.
    Returns a list of 2 element tuples, with each element the indices of a AWAP cell.
    '''
    return list(zip(*np.where(np.logical_not(load_mask_grid(fn_mask)))))

def load_mask_grid(fn_mask=DEFAULT_AWRAL_MASK):
    if os.path.splitext(fn_mask)[1] == '.flt':
        return _load_mask_flt(fn_mask)
    elif os.path.splitext(fn_mask)[1] == '.h5':
        return _load_mask_h5(fn_mask)
    else:
        raise Exception("unknown mask grid format: %s" % fn_mask)

def _load_mask_h5(fn_mask=DEFAULT_AWRAL_MASK):
    import h5py
    h = h5py.File(fn_mask,'r')
    return h['parameters']['mask'][:] <= 0

def _load_mask_flt(fn_mask=DEFAULT_AWRAL_MASK):
    import osgeo.gdal as gd
    gd_mask = gd.Open(fn_mask)
    bd_mask = gd_mask.GetRasterBand(1)
    return bd_mask.ReadAsArray() <= 0

def load_meta():
    import pandas as _pd
    import os as _os
    # from settings import AWRAPATH as _AWRAPATH

    #TODO - metadata csv should it be a module

    p = _os.path.join(_os.path.dirname(__file__),'data','awral_outputs.csv')
    output_meta = _pd.DataFrame.from_csv(p)

    # Read input metadata into dataframe as well and concat it with output metadata
    # input_meta = _pd.DataFrame.from_csv(_os.path.join(_AWRAPATH,"Landscape/Metadata/awraL_inputs.csv"))

    # return _pd.concat([output_meta, input_meta])
    return output_meta