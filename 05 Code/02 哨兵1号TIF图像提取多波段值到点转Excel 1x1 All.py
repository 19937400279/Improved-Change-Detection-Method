import arcpy
import pandas as pd
import os

def extract_multivalue_to_point(image_path, point_feature):
    # 提取哨兵1号TIF图像多波段数据到点
    arcpy.env.workspace = image_path  # 设置ArcPy工作空间
    raster_all = arcpy.ListRasters('*', 'tif')  # 获取所有的tif文件
    for raster in raster_all:
        print([raster, raster.split('_')[4][2:8]])  # 打印文件名和日期
        arcpy.sa.ExtractMultiValuesToPoints(point_feature, [[raster, raster.split('_')[4][2:8]]])  # 多值提取至点并指定列名
    print('提取哨兵1号TIF图像多波段数据到点运行完成')


def feature_table_to_excel(feature_table, export_csv, export_excel):
    # 导出属性表
    if os.path.isfile(export_csv):  # 如果当前目录存在这些文件，就先删除
        os.remove(export_csv)
    if os.path.isfile(export_excel):
        os.remove(export_excel)
    arcpy.conversion.ExportTable(feature_table, export_csv)
    print("属性表转csv成功")
    csv_file = pd.read_csv(export_csv, low_memory=False, encoding='utf-8')
    csv_file = csv_file.iloc[:, 1:]
    csv_file.to_excel(export_excel, index=False, encoding='gbk')
    # 把excel读进来，然后修改列名，重新写入
    inDataFrame = pd.read_excel(export_excel)  # 读取Excel文件
    del_names = ['OID_', 'id', 'Longitude', 'Latitude']
    for del_name in del_names:
        if del_name in inDataFrame.columns:
            inDataFrame.drop(columns=del_name, inplace=True)  # 删除无关的列
    column_name = inDataFrame.columns
    column_name = ['20' + name for name in column_name if name != 'Sites']
    column_name.insert(0, 'Sites')
    inDataFrame.columns = column_name
    if os.path.isfile(export_excel):  # 删除旧的Excel文件
        os.remove(export_excel)
    inDataFrame.to_excel(export_excel, index=False)
    print("csv转Excel成功")
    if os.path.isfile(r'E:\02 Sentinel\04 Table\1x1 All\schema.ini'):
        os.remove(r'E:\02 Sentinel\04 Table\1x1 All\schema.ini')
    if os.path.isfile(r'E:\02 Sentinel\04 Table\1x1 All\Sentinel 2017-2021 All.csv.xml'):
        os.remove(r'E:\02 Sentinel\04 Table\1x1 All\Sentinel 2017-2021 All.csv.xml')


def organize_excel(excel_path, out_vh_excel_path, out_vv_excel_path, out_angle_excel_path):
    # 整理Excel表格中的多波段数据
    for i in range(6):
        if os.path.isfile(out_vh_excel_path[i]):  # 如果当前目录存在这些文件，就先删除
            os.remove(out_vh_excel_path[i])
        if os.path.isfile(out_vv_excel_path[i]):
            os.remove(out_vv_excel_path[i])
        if os.path.isfile(out_angle_excel_path[i]):
            os.remove(out_angle_excel_path[i])
    inDataFrame = pd.read_excel(excel_path)  # 读取Excel文件
    vhDataFrame = pd.DataFrame()
    vvDataFrame = pd.DataFrame()
    angleDataFrame = pd.DataFrame()
    vhDataFrame = pd.concat([vhDataFrame, inDataFrame.iloc[:, 0:1]], axis=1)  # 复制站点名
    vvDataFrame = pd.concat([vvDataFrame, inDataFrame.iloc[:, 0:1]], axis=1)
    angleDataFrame = pd.concat([angleDataFrame, inDataFrame.iloc[:, 0:1]], axis=1)
    dataNum = inDataFrame.shape[1]
    for i in range(1, dataNum):  # 依次获取b1,b2,b3波段
        if i % 3 == 1:
            vhDataFrame = pd.concat([vhDataFrame, inDataFrame.iloc[:, i]], axis=1)
        if i % 3 == 2:
            vvDataFrame = pd.concat([vvDataFrame, inDataFrame.iloc[:, i]], axis=1)
        if i % 3 == 0:
            angleDataFrame = pd.concat([angleDataFrame, inDataFrame.iloc[:, i]], axis=1)
    # 获取每一年的索引范围
    column_name = list(vhDataFrame.columns)
    b1_17_num = 0
    b1_18_num = 0
    b1_19_num = 0
    b1_20_num = 0
    b1_21_num = 0
    for item in column_name:
        if 'b1_17' in item:
            b1_17_num = b1_17_num + 1
        if 'b1_18' in item:
            b1_18_num = b1_18_num + 1
        if 'b1_19' in item:
            b1_19_num = b1_19_num + 1
        if 'b1_20' in item:
            b1_20_num = b1_20_num + 1
        if 'b1_21' in item:
            b1_21_num = b1_21_num + 1
    year2017_index = b1_17_num + 1
    year2018_index = year2017_index + b1_18_num
    year2019_index = year2018_index + b1_19_num
    year2020_index = year2019_index + b1_20_num
    year2021_index = year2020_index + b1_21_num
    year_index = [1, year2017_index, year2018_index, year2019_index, year2020_index, year2021_index]
    # 按年份整理数据
    for year in range(5):  # 依次获取2017-2021的数据并输出保存到excel
        vh = pd.DataFrame()
        vv = pd.DataFrame()
        angle = pd.DataFrame()
        vh = pd.concat([vh, vhDataFrame.iloc[:, 0:1]], axis=1)
        vv = pd.concat([vv, vvDataFrame.iloc[:, 0:1]], axis=1)
        angle = pd.concat([angle, angleDataFrame.iloc[:, 0:1]], axis=1)
        vh = pd.concat([vh, vhDataFrame.iloc[:, year_index[year]:year_index[year+1]]], axis=1)
        vv = pd.concat([vv, vvDataFrame.iloc[:, year_index[year]:year_index[year+1]]], axis=1)
        angle = pd.concat([angle, angleDataFrame.iloc[:, year_index[year]:year_index[year+1]]], axis=1)
        out_vh_name = out_vh_excel_path[year]
        out_vv_name = out_vv_excel_path[year]
        out_angle_name = out_angle_excel_path[year]
        # 删除列名称中的b1_ b2_ b3_ s1 = re.sub('[\t\r]', '', s)
        vh.columns = [name.replace('b1_', '') for name in list(vh.columns)]
        vv.columns = [name.replace('b2_', '') for name in list(vv.columns)]
        angle.columns = [name.replace('b3_', '') for name in list(angle.columns)]
        vh.to_excel(out_vh_name, index=False)
        vv.to_excel(out_vv_name, index=False)
        angle.to_excel(out_angle_name, index=False)
    # 所有年份汇总 2017-2021
    out_vh_name = out_vh_excel_path[5]
    out_vv_name = out_vv_excel_path[5]
    out_angle_name = out_angle_excel_path[5]
    # 删除列名称中的b1_ b2_ b3_
    vhDataFrame.columns = [name.replace('b1_', '') for name in list(vhDataFrame.columns)]
    vvDataFrame.columns = [name.replace('b2_', '') for name in list(vvDataFrame.columns)]
    angleDataFrame.columns = [name.replace('b3_', '') for name in list(angleDataFrame.columns)]
    vhDataFrame.to_excel(out_vh_name, index=False)
    vvDataFrame.to_excel(out_vv_name, index=False)
    angleDataFrame.to_excel(out_angle_name, index=False)
    print('表格数据按年份整理完成')

if __name__ == '__main__':
    # 提取哨兵1号TIF图像多波段数据到点
    image_path = r'E:\02 Sentinel\05 Data\Sentinel 2017-2021'  # 哨兵TIF图像输入路径
    point_feature = r'E:\01 SMN\05 Shapefile\01 Sentinel Point All\sentinel_point_all.shp'  # 全部站点
    # extract_multivalue_to_point(image_path, point_feature)

    # 导出属性表
    feature_table = r'E:\01 SMN\05 Shapefile\01 Sentinel Point All\sentinel_point_all.shp'  # 全部站点
    export_csv = r'E:\02 Sentinel\04 Table\1x1 All\Sentinel 2017-2021 All.csv'
    export_excel = r'E:\02 Sentinel\04 Table\1x1 All\Sentinel 2017-2021 All.xlsx'
    feature_table_to_excel(feature_table, export_csv, export_excel)

    # 整理Excel表格中的多波段数据
    excel_path = r'E:\02 Sentinel\04 Table\1x1 All\Sentinel 2017-2021 All.xlsx'
    out_vh_excel_path = [r'E:\02 Sentinel\04 Table\1x1 All\Sentinel VH 2017 All.xlsx', r'E:\02 Sentinel\04 Table\1x1 All\Sentinel VH 2018 All.xlsx', r'E:\02 Sentinel\04 Table\1x1 All\Sentinel VH 2019 All.xlsx', r'E:\02 Sentinel\04 Table\1x1 All\Sentinel VH 2020 All.xlsx', r'E:\02 Sentinel\04 Table\1x1 All\Sentinel VH 2021 All.xlsx', r'E:\02 Sentinel\04 Table\1x1 All\Sentinel VH 2017-2021 All.xlsx']
    out_vv_excel_path = [r'E:\02 Sentinel\04 Table\1x1 All\Sentinel VV 2017 All.xlsx', r'E:\02 Sentinel\04 Table\1x1 All\Sentinel VV 2018 All.xlsx', r'E:\02 Sentinel\04 Table\1x1 All\Sentinel VV 2019 All.xlsx', r'E:\02 Sentinel\04 Table\1x1 All\Sentinel VV 2020 All.xlsx', r'E:\02 Sentinel\04 Table\1x1 All\Sentinel VV 2021 All.xlsx', r'E:\02 Sentinel\04 Table\1x1 All\Sentinel VV 2017-2021 All.xlsx']
    out_angle_excel_path = [r'E:\02 Sentinel\04 Table\1x1 All\Sentinel Angle 2017 All.xlsx', r'E:\02 Sentinel\04 Table\1x1 All\Sentinel Angle 2018 All.xlsx', r'E:\02 Sentinel\04 Table\1x1 All\Sentinel Angle 2019 All.xlsx', r'E:\02 Sentinel\04 Table\1x1 All\Sentinel Angle 2020 All.xlsx', r'E:\02 Sentinel\04 Table\1x1 All\Sentinel Angle 2021 All.xlsx', r'E:\02 Sentinel\04 Table\1x1 All\Sentinel Angle 2017-2021 All.xlsx']
    organize_excel(excel_path, out_vh_excel_path, out_vv_excel_path, out_angle_excel_path)


