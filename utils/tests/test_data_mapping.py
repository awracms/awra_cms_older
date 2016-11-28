import os

def test_get_padded_by_coords():
    from awrams.utils.io.data_mapping import SplitFileManager
    from awrams.utils.extents import from_boundary_offset
    from awrams.utils.mapping_types import gen_coordset
    import awrams.utils.datetools as dt

    path = os.path.join(os.path.dirname(__file__),'..','..','test_data','simulation')

    sfm = SplitFileManager.open_existing(path,'temp_min_day_*.nc','temp_min_day')
    # return sfm
    extent = from_boundary_offset(200,200,231,231)
    period = dt.dates('2011')
    coords = gen_coordset(period,extent)

    data = sfm.get_padded_by_coords(coords)
    print(data.shape)
    print(coords.shape)
    assert data.shape == coords.shape

if __name__ == '__main__':
    test_get_padded_by_coords()