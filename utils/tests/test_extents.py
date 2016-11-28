import numpy as np

import awrams.utils.extents as extents

from nose.tools import nottest,assert_almost_equal

@nottest
def build_mock_array():
    GLOBAL_GEOREF = extents.global_georef()
    print(GLOBAL_GEOREF.nlats,GLOBAL_GEOREF.nlons)
    data = np.zeros(shape=(GLOBAL_GEOREF.nlats,GLOBAL_GEOREF.nlons))
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):
            data[i,j] = i * data.shape[0] + j
    return data

# @nottest
def test_extent_all():
    e = extents.default()

    print(e.shape)
    assert(e.shape == extents.load_mask_grid().shape)

    print(len(e.cell_list()))
    assert(len(e.cell_list()) == 281655)

# @nottest
def test_cell():
    GLOBAL_GEOREF = extents.global_georef()

    c = extents.from_cell_offset(100,200)
    assert(c.shape == (1,1))
    assert(c.x_index == slice(200, 201, None) and c.y_index == slice(100, 101, None))
    assert c.parent_ref.lat_origin == GLOBAL_GEOREF.lat_origin and c.parent_ref.lon_origin == GLOBAL_GEOREF.lon_origin
    assert not c.parent_ref.lat_origin == c.lat_origin and not c.parent_ref.lon_origin == c.lon_origin

    ### test itercells, which generates cells from local mask, honours parent_ref cell indices
    cells = [cell for cell in c.itercells()]
    assert len(cells) == 1
    assert cells[0].y_min == 100 and cells[0].x_min == 200
    assert cells[0].lat_origin == c.lat_origin and cells[0].lon_origin == c.lon_origin

    ### translate
    c_t = c.translate_localise_origin()
    assert(c_t.x_index == slice(0, 1, None) and c_t.y_index == slice(0, 1, None))
    assert c_t.parent_ref.lat_origin == c_t.lat_origin and c_t.parent_ref.lon_origin == c_t.lon_origin

    ### test itercells, which generates cells from local mask, honours parent_ref cell indices
    cells = [cell for cell in c_t.itercells()]
    assert len(cells) == 1
    assert cells[0].y_min == 0 and cells[0].x_min == 0
    assert cells[0].lat_origin == c_t.lat_origin and cells[0].lon_origin == c_t.lon_origin

    # assert False

# @nottest
def test_contains():
    bb0 = extents.from_boundary_offset(0,0,399,399)
    bb1 = extents.from_boundary_offset(50,50,100,100)

    assert bb0.contains(bb1)
    assert not bb1.contains(bb0)

    assert bb0.contains(extents.from_cell_offset(250,250))

    assert extents.default().contains(bb0)

# @nottest
def test_bb_translate_mask():

    bb = extents.from_boundary_offset(100,100,399,399)

    bb_t = bb.translate_localise_origin()

    assert (bb.mask == bb_t.mask).all()

# @nottest
def test_translate_subdata():
    mock_array = build_mock_array()

    bb = extents.from_boundary_offset(100,100,399,399)

    sub_data = mock_array[bb.x_index,bb.y_index]

    assert (sub_data == mock_array[100:400,100:400]).all()

    bb1 = extents.from_boundary_offset(20,20,29,29,bb.parent_ref)

    print(bb1)

    assert(sub_data[bb1.x_index,bb1.y_index] == mock_array[120:130,120:130]).all()

# @nottest
def test_multiextent():
    m = dict(a=extents.from_cell_offset(250,250),
             b=extents.from_cell_offset(275,275),
             c=extents.from_cell_offset(300,300),
             d=extents.from_boundary_offset(200,290,202,292))
    e = extents.from_multiple(m)
    # print(e.mask.shape)
    # print(e.mask[49:52,:2])
    # print(e.mask[:4,40:44])

    for k,cell in m.items():
        # print(k,cell)
        assert(e.contains(cell))

    # for cell in e:
    #     print("LLL",cell)
    #
    assert e.x_min == 250
    assert e.x_max == 300
    assert e.y_min == 200
    assert e.y_max == 300

    assert e.cell_count == 12

    # assert False

# @nottest
def test_contains():
    e = extents.default()

    b1 = extents.from_boundary_offset(200,200,210,210)
    assert e.contains(b1)

    b2 = extents.from_boundary_offset(675,835,685,845)
    assert not e.contains(b2)

    b3 = extents.from_boundary_offset(700,900,710,910)
    assert not e.contains(b3)

    # translate parent_ref to local (from global)
    b1t = b1.translate_localise_origin()
    assert e.contains(b1t)

    # assert False

# @nottest
def test_area_with_osr():
    try:
        import osr

        c = extents.from_cell_offset(300,200)
        c.compute_areas()
        assert_almost_equal(c.area,27956330.541281,places=6)
        print("300,200",c.area)

        c = extents.from_cell_offset(300,400)
        c.compute_areas()
        assert_almost_equal(c.area,27956330.541269,places=6)
        print("300,400",c.area)

    except ImportError:
        pass

    try:
        import osr

        c = extents.from_boundary_offset(300,200,302,201)
        c.compute_areas()
        assert_almost_equal(c.area,167671123.116,places=3)
        print(c.areas.shape,c.areas)

        for cell in c:
            print(cell)
            cc = extents.from_cell_offset(*cell)
            cc.compute_areas()
            lcell = c.localise_cell(cell)
            print(cc.area, c.areas[lcell[0],lcell[1]])
            assert cc.area == c.areas[lcell[0],lcell[1]]

    except ImportError:
        pass
    # assert False

def test_area_without_osr():
    import awrams.utils.extents
    awrams.utils.extents._LONGLAT_TO_AEA = None

    c = extents.from_cell_offset(300,200)
    c.compute_areas()
    # assert_almost_equal(c.area,27956330.541281,places=6) # with gdal
    assert_almost_equal(c.area,28044093.890163,places=6)

    c = extents.from_boundary_offset(300,200,301,201)
    c.compute_areas()
    # assert_almost_equal(c.area,111803049.532,places=3) # with gdal
    assert_almost_equal(c.area,112153279.927,places=3)

    # assert False

# @nottest
def test_extent_all():
    e = extents.default()

    print(e.shape)
    assert(e.shape == extents.load_mask_grid().shape)

    print(len(e.cell_list()))
    assert(len(e.cell_list()) == 281655)
    # assert False

@nottest
def test_mask():
    from awrams.utils.helpers import load_mask_grid,load_mask
    import os
    print("GRID")
    m1 = load_mask(os.path.join(os.path.dirname(__file__),'../awrams/utils/data/mask.flt'))
    print(len(m1))

    print("H5")
    m2 = load_mask(os.path.join(os.path.dirname(__file__),'../awrams/utils/data/mask.h5'))
    print(len(m2))

    assert len(m1) == len(m2) == 281655


if __name__ == '__main__':
    # build_mock_array()
    test_extent_all()
    # test_translate_subdata()
    test_mask()
