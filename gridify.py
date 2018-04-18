import numpy as np
import geopandas as gp
import pandas    as pd

def to_gdf(df):
    from geopandas import GeoDataFrame
    from pyproj import Proj, transform
    from shapely.geometry import Point

    inProj  = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:2263', preserve_units = True) # https://github.com/jswhit/pyproj/issues/67

    df['geometry'] = df.apply(lambda x: Point(transform(inProj, outProj, x.LONGITUDE , x.LATITUDE)), axis =1 )
    gdf = GeoDataFrame(df, geometry = 'geometry')
    return gdf

def gridify(gdf, edge_size = 1000, cut = False, NYS_plane=True):
    import gpd_lite_toolbox as glt


    grid = glt.gridify_data(gdf, edge_size, 'COUNT', method = np.sum, cut = cut)

    grid['COUNT'] = grid.COUNT.replace(-1,0) # seems to be a bug in gridify function that defaults to -1

    grid['centroid'] = grid.geometry.centroid
    grid['x_point'] = grid.centroid.apply(lambda x: x.coords.xy[0][0])
    grid['y_point'] = grid.centroid.apply(lambda x: x.coords.xy[1][0])

    if NYS_plane:
        grid['x_point'] = (grid['x_point'] - 900000) / 10000  #+ np.random.normal(0,.001, size= n)
        grid['y_point'] = (grid['y_point'] - 100000) / 10000  #+ np.random.normal(0,.001, size = n)


    return grid


def get_counts_byfreq(gdf, grid, freq = 'M'):

    out_grid = gp.sjoin(gdf[['geometry','DATE','COUNT']], grid[['geometry']] , op = 'within').copy()
    out_grid.rename(columns= {'index_right':'GRID_SQUARE'}, inplace = True)

    out_grid['DATETIME'] = pd.to_datetime(out_grid['DATE'])

    out_grid.set_index('DATETIME', inplace = True)

    counts =  out_grid.groupby([pd.Grouper(freq = freq), 'GRID_SQUARE'])[['COUNT']].sum()
    dt_idx = pd.date_range(counts.index.min()[0] +  pd.offsets.MonthEnd(0) , counts.index.max()[0] +  pd.offsets.MonthEnd(0), freq = freq)
    out = pd.DataFrame( index = pd.MultiIndex.from_product([dt_idx, grid.index])).reset_index()
    out.columns = ['DATETIME', 'GRID_SQUARE']
    counts.reset_index(inplace = True)
    out = out.merge(counts, how = 'left', on = ['DATETIME', 'GRID_SQUARE']).fillna(0)
    out = out.merge(grid[['x_point', 'y_point']], left_on = 'GRID_SQUARE', right_index = True, how = 'left')
    out.sort_values(['GRID_SQUARE','DATETIME'], inplace = True)

    date_ind_mapper = pd.Series(out.DATETIME.unique()).reset_index().set_index(0)
    date_ind_mapper['index'] += 1

    out['DATE_IND'] = out.DATETIME.map(date_ind_mapper['index'])
    out['GRID_SQUARE'] += 1

    out['GRID_SQUARE'] = out['GRID_SQUARE'].astype(np.float32)
    out['DATE_IND']    = out['DATE_IND'].astype(np.float32)


    return out
