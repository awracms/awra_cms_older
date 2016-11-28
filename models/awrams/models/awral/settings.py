'''
Settings for AWRA-L model

To override, create and edit ~/.awrams/awral.py
'''

from os.path import dirname as _dirname
from os.path import join as _join

DEFAULT_PARAMETER_FILE=_join(_dirname(__file__),'data/DefaultParameters.json')

SPATIAL_FILE = None
CLIMATE_DATA = None

#+++ TODO: Remove these Bureau hard codes
SPATIAL_FILE = _join(_dirname(__file__),'data/spatial_parameters.h5')
CLIMATE_DATA = './'


from awrams.utils.settings_manager import get_settings as _get_settings
exec(_get_settings('awral'))
