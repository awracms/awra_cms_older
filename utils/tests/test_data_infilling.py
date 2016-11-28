
import numpy as np
import datetime as dt
from awrams.utils.ts.time_series_infilling import FillWithZeros,FillWithClimatology
from .test_gridded_time_series import create_mock_data, create_dummy_dataset
from nose.tools import with_setup,raises,assert_list_equal,assert_equal
from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('test_data_infilling')

def setup_gappy():
    global dataset
    dataset = create_dummy_dataset(opener=mock_open_gappy_data)

def mock_open_gappy_data(self,fn):
    return create_gappy_data(fn)

def create_gappy_data(fn):
    # insert_gaps method used because broadcasting doesn't seem to work
    # with diskless netCDF files with python-netCDF4
    def insert_gaps(array):
        array[5,:,:] = np.nan
        array[10,20,30] = np.nan
        array[15:30,20,30] = np.nan
        array[50,10:18,30] = np.nan

    data = create_mock_data(fn,data_modifier=insert_gaps)
#    data.variables['temp_min'][5,:,:] = np.nan
#    data.variables['temp_min'][10,20,30] = np.nan
#    data.variables['temp_min'][15:30,20,30] = np.nan
#    data.variables['temp_min'][50,10:18,30] = np.nan

    return data

def setup_with_zero_fill():
    setup_gappy()
    dataset.gap_filler = FillWithZeros()

def setup_with_climatology():
    setup_gappy()
    dataset.gap_filler = FillWithClimatology(FakeClimatology())

def teardown_gappy():
    global dataset
    dataset.close_all()
    dataset = None

class FakeClimatology(object):
    def __init__(self):
        self.freq = 'monthly'
        
    def get(self,month,location):
        return -month

    def get_for_location(self,location):
        return location[0]*np.arange(1,13)

@with_setup(setup_gappy,teardown_gappy)
def test_detect_nan_gaps():
    filler = FillWithZeros()
    time_series = dataset.open_files[0].variables['temp_min'][:,20,30]
    print("Time Series looks like:" + str(time_series))
    result = filler.has_gaps(time_series,(20,30))
    assert result


@with_setup(setup_gappy,teardown_gappy)
@raises(BaseException)
def test_gap_triggers_exception():
    time_series = dataset.retrieve_time_series((10,30))

@with_setup(setup_with_zero_fill,teardown_gappy)
def test_gap_filled_with_zero():
    time_series = dataset.retrieve_time_series((10,30))
    assert not np.any(time_series==np.nan)
    assert (not np.ma.isMA(time_series)) or (not np.any(time_series.mask))
    assert_equal(time_series[5],0)
    assert_equal(time_series[4],4)

    time_series = dataset.retrieve_time_series((20,30))
    assert not np.any(time_series==np.nan)
    assert (not np.ma.isMA(time_series)) or (not np.any(time_series.mask))
    assert_equal(time_series[5],0)
    assert_equal(time_series[4],0)

@with_setup(setup_with_climatology,teardown_gappy)
def test_gap_filled_with_climatology():
    time_series = dataset.retrieve_time_series((10,30))
    assert not np.any(time_series==np.nan)
    assert (not np.ma.isMA(time_series)) or (not np.any(time_series.mask))
    assert_equal(time_series[5],10) # January
    assert_equal(time_series[4],4) # Not a gap

    time_series = dataset.retrieve_time_series((10,30))
    assert not np.any(time_series==np.nan)
    assert (not np.ma.isMA(time_series)) or (not np.any(time_series.mask))
    assert_equal(time_series[5],10) # January
    assert_equal(time_series[50],20) # February

@with_setup(setup_with_climatology,teardown_gappy)
def test_extend_prior_to_start_of_record():
    start=dt.datetime(1999,1,1)
    end=dt.datetime(2000,12,31)
    time_series = dataset.retrieve_time_series((10,30),start=start,end=end)
    assert_equal(365+366,len(time_series))
    np.testing.assert_array_equal(time_series[0:31],[10]*31)
    np.testing.assert_array_equal(time_series[334:365],[120]*31)

@with_setup(setup_gappy,teardown_gappy)
@raises(BaseException)
def test_request_to_extent_triggers_exception():
    start=dt.datetime(1999,1,1)
    end=dt.datetime(1999,12,31)
    time_series = dataset.retrieve_time_series((10,30),start=start,end=end)
