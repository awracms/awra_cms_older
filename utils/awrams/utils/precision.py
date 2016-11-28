'''
Functionality to manipulate the precision of numeric values
'''
import numpy as np

def iround(value):
    '''
    Return a rounded integer from a fp value
    '''
    return int(round(value))

def quantize(value,units,op=round,precision=3):
    '''
    Quantize to the nearest units
    '''
    return round(op(value / units) * units,precision)

def aquantize(value,units,op=np.around,precision=3):
    '''
    Quantize an array the nearest units
    '''
    return np.around(op(value / units) * units,precision)

def sanitize_cell(cell):
    '''
    Round a lat/lon pair to correct units
    '''
    return quantize(cell[0],0.05),quantize(cell[1],0.05)

def sanitize_geo_array(data):
    '''
    Round a lat/lon pair to correct units
    '''
    return aquantize(data,0.05)
