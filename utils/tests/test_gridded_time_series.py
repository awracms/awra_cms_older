"""
Tests for the ClimateDataSet class
"""
import sys

#sys.path.append("LocalPackages")
import netCDF4 as nc
import numpy as np
import datetime as dt
import tempfile
from awrams.utils.ts import gridded_time_series
import re
import pandas as pd
from calendar import isleap
from awrams.utils.ts.gridded_time_series import ClimateDataSet,FileMatcher,NoMatchingFilesException
from nose.tools import nottest,with_setup,assert_almost_equal,assert_equal,assert_true,assert_tuple_equal,raises
from numpy.testing import assert_array_equal
import os
from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('test_gridded_time_series')
TEST_DATA_NUM_ROWS=50
TEST_DATA_NUM_COLS=100

def days_in_year(year):
    if isleap(year):
        return 366
    return 365

def days_upto(year):
    """
    Return the number of days from the beginning of the test period to the
    beginning of the year specified
    """
    return sum([days_in_year(y) for y in range(2000,year)]) # + EPOCH

def create_mock_data(fn,data_modifier=None):
    from netCDF4 import Dataset
    pattern = re.compile(r"(?P<year>[1-2][0-9][0-9][0-9])")
    dataset = nc.Dataset(fn,'w',diskless=True)
    year = int(pattern.findall(fn)[0])
    days = days_in_year(year)
    days_prior = days_upto(year)
    val_range = list(range(days_prior,days_prior+days))
    vals = np.array(val_range).reshape(days,1,1) * np.ones((days,TEST_DATA_NUM_ROWS,TEST_DATA_NUM_COLS))
    vals[:,20,30] = 0
    if not data_modifier is None:
        data_modifier(vals)

    ds = Dataset(tempfile.mktemp()+'_'+fn,'w',diskless=True)
    ds.createDimension('latitude',TEST_DATA_NUM_ROWS)
    latitude = ds.createVariable('latitude','f8',dimensions=('latitude',))
    latitude[...] = np.arange(-19,-21.5,-0.05)

    ds.createDimension('longitude',TEST_DATA_NUM_COLS)
    longitude = ds.createVariable('longitude','f8',dimensions=('longitude',))
    longitude[...] = np.arange(135,140,0.05)

    ds.createDimension('time')
    time = ds.createVariable('time','i',dimensions=('time',))
    time.setncattr('units','days since 2000-01-01')
    time[:] = val_range

    provenance = ds.createGroup('provenance')
    exist = provenance.createVariable('exist','i',dimensions=('time',))
    exist[...] = list(range(days))

    temp_min = ds.createVariable('temp_min','f',dimensions=('time','latitude','longitude'))
    temp_min.setncattr('units',"degrees C")
    temp_min[...] = vals

    return ds
#    return Sample.build(tempfile.gettempdir()+os.path.sep+fn,diskless=True)

def mock_open_data(self,fn):
    return create_mock_data(fn)

def mock_locate_files(self):
    return ["minT_%d.nc" % y for y in range(2000,2010)]

def fake_glob(self):
    return ["rechunked_minT_%d.nc"%y for y in range(1990,2000)]

def create_dummy_dataset(locator=mock_locate_files,opener=mock_open_data):
    """
    Create dummy minT (minimum Temperature) data, daily, from 1/1/2000 to 31/12/2010.
    Grid from -19,135 to -21.5,140 on a 0.05 grid (50 x 100)
    """
    locate_files = FileMatcher.locate
    FileMatcher.locate = locator

    open_data = ClimateDataSet._open_data
    ClimateDataSet._open_data = opener

    dataset = ClimateDataSet('temp_min')
    FileMatcher.locate = locate_files
    ClimateDataSet._open_data = open_data
    return dataset

def setup_dummy_data():
    logger.debug('setup_dummy_data')
    global dataset
    dataset = create_dummy_dataset()
    from awrams.utils.test_support import MockLoggingHandler
    global handler
    handler = MockLoggingHandler()
    gridded_time_series.logger.addHandler(handler)

def setup_dummy_search():
    logger.debug('setup_dummy_search')
    global dataset, saved_glob

    saved_glob = FileMatcher.locate
    FileMatcher.locate = fake_glob
    open_data = ClimateDataSet._open_data
    ClimateDataSet._open_data = mock_open_data

    dataset = ClimateDataSet('temp_min',search_pattern="rechunked_minT_*.nc")

    ClimateDataSet._open_data = open_data
    FileMatcher.locate = saved_glob

def teardown_fn():
    logger.debug('teardown_fn')
    global dataset
    dataset.close_all()
    dataset = None
    global handler
    gridded_time_series.logger.removeHandler(handler)

DAYS_IN_DECADE = 3653
EPOCH=0


@with_setup(setup_dummy_data, teardown_fn)
def test_locate_all():
    assert dataset.files == mock_locate_files(None)
    print(dataset.start_date,dataset.end_date)
    assert dataset.start_date == dt.datetime(2000,1,1)
    assert dataset.end_date == dt.datetime(2009,12,31)
    assert np.all(dataset.shape == (DAYS_IN_DECADE,50,100))

@with_setup(setup_dummy_data, teardown_fn)
def test_access_time_series():
    time_series = dataset.retrieve_time_series((25,35))
    assert len(time_series) == DAYS_IN_DECADE
    assert np.all(time_series == list(range(EPOCH,EPOCH+DAYS_IN_DECADE)))

    time_series2 = dataset.retrieve_time_series((20,30))
    assert len(time_series2) == DAYS_IN_DECADE
    assert np.all(time_series2 == 0)

#@with_setup(setup_dummy_data, teardown_fn)
#def test_access_short_time_series():
#    time_series = dataset.retrieve_time_series((25,35),start=dt.datetime(2000,2,1),end=dt.datetime(2001,1,31))
#    assert len(time_series) == 366
#    assert np.all(time_series == range(31,31+366))

@with_setup(setup_dummy_data, teardown_fn)
def test_access_grid():
    grid = dataset.retrieve_grid(dt.datetime(2001,1,1))
    assert grid.shape == (50,100)
    assert grid[25,20] == 366

    grid2 = dataset.retrieve_grid(dt.datetime(2001,1,31))
    assert grid2.shape == (50,100)
    assert grid2[25,20] ==396

@with_setup(setup_dummy_data,teardown_fn)
def test_latitude_for_cell():
    assert_almost_equal(dataset.latitude_for_cell([0,55]),-19)
    assert_almost_equal(dataset.latitude_for_cell([0,40]),-19)
    assert_almost_equal(dataset.latitude_for_cell([1,40]),-19.05)
    assert_almost_equal(dataset.latitude_for_cell([49,40]),-21.45)

def check_locate_date(date,file_start_timestep,file_offset):
    file,offset = dataset.locate_day(date)
    assert file.variables['time'][0] == (dataset.epoch_offset + file_start_timestep)
    assert offset == file_offset

@with_setup(setup_dummy_data, teardown_fn)
def test_locate_day():
    check_locate_date(dt.datetime(2000,1,1),0,0) # Start of record
    check_locate_date(dt.datetime(2009,12,31),3288,364) # Start of record
    check_locate_date(dt.datetime(2002,2,2),731,32)

@with_setup(setup_dummy_data, teardown_fn)
@raises(BaseException)
def test_locate_out_of_range_day():
    dataset.locate_day(dt.datetime(2012,1,1))

@nottest
@with_setup(setup_dummy_data, teardown_fn)
def test_locate_time_period():
    start = dt.datetime(2005,2,1) # Day number 1858 (0 based), day 31 in file
    end = dt.datetime(2008,6,29) # Day number 3102, day 180 in file
    time_period = pd.date_range(start,end)

    time_slices = dataset._locate_period(time_period)
    assert len(time_slices) == 4
    expected = [("2005",slice(31,None)),
                ("2006",slice(None,None)),
                ("2007",slice(None,None)),
                ("2008",slice(None,181))] # To include day #180
    for comparison in zip(expected,time_slices):
        assert_true(comparison[0][0] in comparison[1][0].filepath())
        assert_equal(comparison[0][1],comparison[1][1])

@nottest
@with_setup(setup_dummy_data, teardown_fn)
def test_locate_short_time_period():
    start = dt.datetime(2005,2,1) # Day number 1858 (0 based), day 31 in file
    end   = dt.datetime(2005,3,30) # Day number 1916 (0 based), day 89 in file
    time_period = pd.date_range(start,end)
    time_slices = dataset._locate_period(time_period)
    assert len(time_slices) == 1
    assert_true("2005" in time_slices[0][0].filepath())
    assert_equal(time_slices[0][1],slice(31,89))

@nottest
@with_setup(setup_dummy_data, teardown_fn)
def test_locate_time_period_at_start():
    start = dt.datetime(2000,1,1) # Day number 0 (0 based), day 0 in file
    end   = dt.datetime(2000,1,31) # Day number 30 (0 based), day 30 in file
    time_period = pd.date_range(start,end)
    time_slices = dataset._locate_period(time_period)
    assert len(time_slices) == 1
    assert_true("2000" in time_slices[0][0].filepath())
    assert_equal(time_slices[0][1],slice(0,31))

@with_setup(setup_dummy_data, teardown_fn)
def test_retrieve_3d_short():
    start = dt.datetime(2005,2,1) # Day number 1858 (0 based), day 31 in file
    end   = dt.datetime(2005,3,30) # Day number 1916 (0 based), day 89 in file
    time_period = pd.date_range(start,end)

    extent = [slice(15,25),slice(25,45)]
    data = dataset.retrieve(time_period,extent)
    assert_tuple_equal((58,10,20),data.shape)
    assert_array_equal(data[:,0,0],np.arange(1858,1916))
    assert_array_equal(data[:,5,5],np.zeros(58))

@with_setup(setup_dummy_data, teardown_fn)
def test_retrieve_3d():
    start = dt.datetime(2005,2,1) # Day number 1858 (0 based), day 31 in file
    end = dt.datetime(2008,6,29) # Day number 3103, day 180 in file
    time_period = pd.date_range(start,end)

    extent = [slice(15,25),slice(25,45)]
    data = dataset.retrieve(time_period,extent)
    assert_tuple_equal((1245,10,20),data.shape)
    assert_array_equal(data[:,0,0],np.arange(1858,3103))
    assert_array_equal(data[:,5,5],np.zeros(1245))

@with_setup(setup_dummy_data, teardown_fn)
def test_retrieve_scalar():
    check_retrieve_scalar(dt.datetime(2005,2,1),[15,25],1858)
    # Day number 1858 (0 based), day 31 in file

    check_retrieve_scalar(dt.datetime(2000,1,1),[15,25],0)

    check_retrieve_scalar(dt.datetime(1990,1,1),[15,25],np.nan)

def check_retrieve_scalar(when,where,what):
    start = when
    end = when
    time_period = [start,end]

    extent = where
    data = dataset.retrieve(time_period,extent)

    assert np.isscalar(data)
    if np.isnan(what):
        assert np.isnan(data)
    else:
        assert_equal(data,what)

@with_setup(setup_dummy_data, teardown_fn)
def test_log_out_of_range():
    criticals_before = len(handler.messages['critical'])
    raised = False
    try:
        ts = dataset.retrieve([dataset.start_date,dataset.end_date],[slice(500,1000),slice(250,750)])
    except IndexError:
        raised = True

    assert raised
    criticals_after = len(handler.messages['critical'])
    assert criticals_after == (criticals_before+1)
    assert handler.messages['critical'][-1].startswith('Attempt to retrieve data beyond extent')

@with_setup(setup_dummy_search,teardown_fn)
def test_locate_files():
    assert dataset.files[0] == 'rechunked_minT_1990.nc'
    assert dataset.files[-1] == 'rechunked_minT_1999.nc'
    assert len(dataset.files) == 10

def test_should_fail_on_no_files():
    def fail_glob(pattern):
        return []

    saved_glob = FileMatcher.locate
    FileMatcher.locate = fail_glob
    exception_raised = False
    exception_message_good = False
    message_received = ""
    try:
        dataset = ClimateDataSet('temp_min',search_pattern="rechunked_minT_*.nc")
    except NoMatchingFilesException as e:
        exception_raised = True
        exception_message_good = e.message.startswith("No matching files for variable temp_min")
        message_received = e.message

    FileMatcher.locate = saved_glob

    assert exception_raised
    if not exception_message_good:
        raise Exception(message_received)

    assert exception_message_good
