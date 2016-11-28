import os
from logging import FATAL,CRITICAL,ERROR,WARNING,INFO,DEBUG
from getpass import getuser
from socket import gethostname

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

# DEFAULT_AWRAL_MASK = os.path.join(dir_path,'data','mask.flt')
DEFAULT_AWRAL_MASK = os.path.join(dir_path,'data','mask.h5')
CATCHMENT_SHAPEFILE = os.path.join(dir_path,'data','Final_list_all_attributes.shp')

#### LOGGING OPTIONS
APPNAME='awrams'

# NOTE: can use environment variables (eg $HOME) and home directory shortcuts (eg ~ or ~username)
LOGFORMAT='%(asctime)s %(levelname)s %(message)s'

LOG_TO_STDOUT=False
LOG_TO_STDERR=True
LOG_TO_FILE=True

LOG_SETTINGS_LOAD=True

#FILE_LOGGING_MODE OPTIONS
APPENDFILE='append'
TIMESTAMPEDFILE='timestamp'
ROTATEDSIZEDFILE='rotatedsized'
DAILYROTATEDFILE='dailyrotated'

#Set default file logging mode to the standard logfile to be appended to
FILE_LOGGING_MODE=APPENDFILE
#OTHER OPTIONS ARE BELOW AND SHOULD BE SET IN USER SETTINGS FILES:
#FILE_LOGGING_MODE=TIMESTAMPEDFILE
#FILE_LOGGING_MODE=DAILYROTATEDFILE
#FILE_LOGGING_MODE=ROTATEDSIZEDFILE

#### SETUP DEFAULT LOGFILE DESTINATION PATH
USER=getuser()
HOST=gethostname().split('.')[0]
HOME=os.path.expanduser('~')

#Standard logfile name in default location of the mounted drive
#Both the standard and timestamped log file names are constructed from this basename
#The basename can be overridden in user settings files

LOGFILEBASE=os.path.join(HOME,'%s'%APPNAME)

# the following are the default values which affect DAILYROTATEDFILE and ROTATEDSIZEDFILE modes only
#If you select one of these FILE_LOGGING_MODEs you can then customise how many or what size the files are

#ROTATEDSIZEDFILE mode is affected by these params:
#How many files to rotate:
ROTATEDSIZEDFILES=10
#Sze of the file before it rotates:
ROTATEDSIZEDBYTES=20000

#DAILYROTATEDFILE mode is affected by:
# How many files to rotate(on a daily basis) so 7 is a week's worth of daily files
DAILYROTATEDFILES=7

LOG_LEVEL=INFO
DEBUG_MODULES=[]

LOG_LEVEL_ON_CELL_RUN=DEBUG
########################## LOGGING ENDED ###################

DEFAULT_CHUNKSIZE = (75,1,50)
CHUNKSIZE=(75, 1, 50)
VAR_CHUNK_CACHE_SIZE = 2**20 # =1048576 ie 1Mb
VAR_CHUNK_CACHE_NELEMS = 1009 # prime number
VAR_CHUNK_CACHE_PREEMPTION = 0.75 # 1 for read or write only
DEFAULT_PRECISION='float32' # WARNING - NOT USED -- +++Seems to get used in netcdf_wrapper.py
VARIABLE_PRECISION = {}#{v:'float64' for v in 'mleaf_dr mleaf_sr s0_dr s0_sr sd_dr sd_sr ss_sr ss_dr sg_bal sr_bal'.split()} # WARNING - NOT USED

DB_OPEN_WITH = "_h5py" # "netCDF4" # OR "_h5py" OR "_nc"

from .settings_manager import get_settings as _get_settings
exec(_get_settings('utils'))

