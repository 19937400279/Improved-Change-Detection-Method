import pandas as pd
import os

def sort_excel(dir_path, sm_excel_2018_2020_all_path, ts_excel_2018_2020_all_path, angle_excel_2017_2021_all_path, vv_excel_2017_2021_all_path, vh_excel_2017_2021_all_path, ndvi_excel_SG_Interpolate_2017_2021_all_path,
               sm_2019_2020_sort_path, ts_2019_2020_sort_path, angle_2017_2021_sort_path, vv_2017_2021_sort_path, vh_2017_2021_sort_path, ndvi_2017_2021_sort_path):
    # 保存文件前，先清空旧文件
    del_files(dir_path)
    print(dir_path + '  文件夹已清空')

    # 读取表格 header=0设置第一行为列索引 index_col=0设置第一列为行索引
    sm_18_20_df = pd.read_excel(sm_excel_2018_2020_all_path, header=0, index_col=0)
    ts_18_20_df = pd.read_excel(ts_excel_2018_2020_all_path, header=0, index_col=0)
    angle_2017_2021_df = pd.read_excel(angle_excel_2017_2021_all_path, header=0, index_col=0)
    vv_2017_2021_df = pd.read_excel(vv_excel_2017_2021_all_path, header=0, index_col=0)
    vh_2017_2021_df = pd.read_excel(vh_excel_2017_2021_all_path, header=0, index_col=0)
    ndvi_2017_2021_df = pd.read_excel(ndvi_excel_SG_Interpolate_2017_2021_all_path, header=0, index_col=0)

    # 筛选日期
    site_name = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M9', 'M10', 'M11', 'M12', 'L5', 'L6', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14']
    data_time_2019_2020 = [column_name for column_name in sm_18_20_df.columns if ('2019' in column_name or '2020' in column_name)]

    # 筛选数据 sm和ts筛选日期和站点  ang,vv,vh,ndvi只需要筛选站点
    sm_2019_2020_sort_df = sm_18_20_df.loc[site_name, data_time_2019_2020]
    ts_2019_2020_sort_df = ts_18_20_df.loc[site_name, data_time_2019_2020]
    angle_2017_2021_sort_df = angle_2017_2021_df.loc[site_name, :]
    vv_2017_2021_sort_df = vv_2017_2021_df.loc[site_name, :]
    vh_2017_2021_sort_df = vh_2017_2021_df.loc[site_name, :]
    ndvi_2017_2021_sort_df = ndvi_2017_2021_df.loc[site_name, :]

    # 保存excel文件
    sm_2019_2020_sort_df.to_excel(sm_2019_2020_sort_path)
    print(sm_2019_2020_sort_path + '  已保存')
    ts_2019_2020_sort_df.to_excel(ts_2019_2020_sort_path)
    print(ts_2019_2020_sort_path + '  已保存')
    angle_2017_2021_sort_df.to_excel(angle_2017_2021_sort_path)
    print(angle_2017_2021_sort_path + '  已保存')
    vv_2017_2021_sort_df.to_excel(vv_2017_2021_sort_path)
    print(vv_2017_2021_sort_path + '  已保存')
    vh_2017_2021_sort_df.to_excel(vh_2017_2021_sort_path)
    print(vh_2017_2021_sort_path + '  已保存')
    ndvi_2017_2021_sort_df.to_excel(ndvi_2017_2021_sort_path)
    print(ndvi_2017_2021_sort_path + '  已保存')

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)

if __name__ == '__main__':
    dir_path = r'E:\04 Method\01 Table/'  # 文件夹目录
    sm_excel_2018_2020_all_path = r'E:\01 SMN\04 Table\SM 2018-2020 All.xlsx'
    ts_excel_2018_2020_all_path = r'E:\01 SMN\04 Table\TS 2018-2020 All.xlsx'
    angle_excel_2017_2021_all_path = r'E:\02 Sentinel\04 Table\1x1 All\Sentinel Angle 2017-2021 All.xlsx'
    vv_excel_2017_2021_all_path = r'E:\02 Sentinel\04 Table\1x1 All\Sentinel VV 2017-2021 All.xlsx'
    vh_excel_2017_2021_all_path = r'E:\02 Sentinel\04 Table\1x1 All\Sentinel VH 2017-2021 All.xlsx'
    ndvi_excel_SG_Interpolate_2017_2021_all_path = r'E:\03 Modis\03 Table\NDVI SG Interpolate SortDate 2017-2021 All.xlsx'
    sm_2019_2020_sort_path = r'E:\04 Method\01 Table\sm_2019_2020_sort.xlsx'  # 筛选后的数据保存路径
    ts_2019_2020_sort_path = r'E:\04 Method\01 Table\ts_2019_2020_sort.xlsx'
    angle_2017_2021_sort_path = r'E:\04 Method\01 Table\angle_2017_2021_sort.xlsx'
    vv_2017_2021_sort_path = r'E:\04 Method\01 Table\vv_2017_2021_sort.xlsx'
    vh_2017_2021_sort_path = r'E:\04 Method\01 Table\vh_2017_2021_sort.xlsx'
    ndvi_2017_2021_sort_path = r'E:\04 Method\01 Table\ndvi_2017_2021_sort.xlsx'
    sort_excel(dir_path, sm_excel_2018_2020_all_path, ts_excel_2018_2020_all_path, angle_excel_2017_2021_all_path, vv_excel_2017_2021_all_path, vh_excel_2017_2021_all_path, ndvi_excel_SG_Interpolate_2017_2021_all_path,
               sm_2019_2020_sort_path, ts_2019_2020_sort_path, angle_2017_2021_sort_path, vv_2017_2021_sort_path, vh_2017_2021_sort_path, ndvi_2017_2021_sort_path)




# todo 数据分析