import os
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
BENCHMARK_SITES = os.path.join(dir_path,'data','SiteLocationsWithUniqueID.csv')
SOILMOISTURE_OBS_PATH = os.path.join(dir_path,'data')

FIG_SIZE = (14,6)
MONTHLY_REJECTION_THRESHOLD=15
ANNUAL_REJECTION_THRESHOLD=6

SM_MODEL_VARNAMES = ['s0_avg', 'ss_avg', 'sd_avg']
SM_MODEL_LAYERS = {'s0_avg': 100., 'ss_avg': 900., 'sd_avg': 5000.}
SM_OBSERVED_LAYERS = ('profile','top','shallow','middle','deep')
