from awrams.utils.catchments import *
from awrams.utils.extents import default

from nose.tools import nottest,assert_almost_equal

def test_catch():
    try:
        from osgeo import ogr
    except ImportError:
        return

    catchments = CatchmentDB()
    c = catchments.by_name.Lachlan_Gunning()
    print(type(c))
    print(c.cell_count)
    print(c.cell_list())
    print(c.area)
    print(c.lat_origin,c.lon_origin)

    assert c.cell_count == len(c.cell_list()) == 47
    try:
        import osr
        assert_almost_equal(c.area,574230028.333,places=3)
    except ImportError:
        pass

    c = catchments.get_by_id(410033)

    assert c.id == 410033

    FAIL_BAD_ID = False

    try:
        c = catchments.get_by_id(2597129839)
    except CatchmentNotFoundError:
        FAIL_BAD_ID = True

    assert(FAIL_BAD_ID)

    # assert False

def test_catchment_contains():
    try:
        from osgeo import ogr
    except ImportError:
        return

    cdb = CatchmentDB()

    c = cdb.get_by_id(410033)

    assert(c.contains(c))

    assert default().contains(c)

def test_shape():
    try:
        from osgeo import ogr
    except ImportError:
        return

    shp_db = ShapefileDB()

    # df = shp_db._get_records_df()
    # print(type(df[df['StationID'] == '421103']))

    e = shp_db._get_by_field('StationID','421103')

    assert e.cell_count == 9
    assert e.cell_list()[0] == (465,744)

    try:
        import osr
        assert_almost_equal(e.area, 70621298.7211, 4)
    except ImportError:
        pass

    assert e.lat_origin == -33.25
    assert e.lon_origin == 149.15
    assert e.y_min == 465
    assert e.x_min == 743

    # assert False
