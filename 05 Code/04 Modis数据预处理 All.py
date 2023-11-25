# -*- coding: UTF-8 -*-
import arcpy
import glob
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import datetime
import scipy.interpolate as spi
from matplotlib.font_manager import FontProperties


def ToDate(a):
    return datetime.datetime.strptime(a, '%Y%m%d').date()

def out_day_by_date(date):
    # 根据输入的日期计算该日期是在当年的第几天
    year=date.year
    month=date.month
    day=date.day
    months=[0,31,59,90,120,151,181,212,243,273,304,334]
    if 0<month<=12:
        sum=months[month-1]
    else:
        print("month error")
    sum+=day
    leap=0
    #接下来判断平年闰年
    if(year%400==0) or ((year%4)==0) and (year%100!=0):#and的优先级大于or    1、世纪闰年:能被400整除的为世纪闰年 2、普通闰年:能被4整除但不能被100整除的年份为普通闰年
        leap=1
    if(leap==1) and (month>2):
        sum+=1   # 判断输入年的如果是闰年,且输入的月大于2月,则该年总天数加1
    return sum


def out_date_by_day(year, day):
    # 根据输入的年份和天数计算对应的日期
    first_day = datetime.datetime(year, 1, 1)
    add_day = datetime.timedelta(days=day - 1)
    return datetime.datetime.strftime(first_day + add_day, "%Y%m%d")

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)

def data_preprocessing(inPath, outPath):
    # MODIS NDVI 数据预处理  不进行裁剪
    file_list = glob.glob(r'E:\03 Modis\04 Data\01 HEGOUT\Extract_13Q11*')  # 清除缓存文件
    for file in file_list:
        os.remove(file)
    del_files(outPath)
    print(outPath + '  旧tif文件已删除')
    arcpy.env.workspace = inPath  # 设置ArcPy工作空间
    raster_all = arcpy.ListRasters('*', 'tif')  # 获取所有的tif文件
    for raster in raster_all:
        filename = os.path.basename(raster)
        outname = os.path.join(outPath, filename)  # 输出文件名
        # raster = arcpy.sa.ExtractByMask(raster, mask)  # 按掩模裁剪
        raster = arcpy.sa.Con(raster, 0, raster, "VALUE < 0")  # 小于0的值全部赋值为0
        raster = arcpy.sa.Con(raster, 10000, raster, "VALUE > 10000")  # 大于10000的值全部赋值为10000
        raster = arcpy.sa.Times(raster, 0.0001)  # 比例缩放除以10000
        raster.save(outname)  # 保存文件
        print(os.path.basename(outname) + '  正在处理...')
    file_list = glob.glob(r'E:\03 Modis\04 Data\02 Preprocess\*.xml')  # 清除缓存文件
    for file in file_list:
        os.remove(file)
    file_list = glob.glob(r'E:\03 Modis\04 Data\02 Preprocess\*.tfw')  # 清除缓存文件
    for file in file_list:
        os.remove(file)
    print("MODIS NDVI 数据预处理完成")


def extract_multivalue_to_point(image_path, point_feature):
    # 提取MODIS NDVI TIF图像数据到点
    arcpy.env.workspace = image_path  # 设置ArcPy工作空间
    raster_all = arcpy.ListRasters('*', 'tif')  # 获取所有的tif文件
    for raster in raster_all:
        print([raster, str(out_date_by_day(int(raster.split('.')[1][1:5]), int(raster.split('.')[1][5:])))[2:]], '正在执行多值提取到点...')
        arcpy.sa.ExtractMultiValuesToPoints(point_feature, [[raster, str(out_date_by_day(int(raster.split('.')[1][1:5]), int(raster.split('.')[1][5:])))[2:]]], 'NONE')
    print('提取MODIS NDVI TIF图像数据到点运行完成')


def feature_table_to_excel(feature_table, export_csv, export_excel):
    # 导出属性表
    if os.path.isfile(export_csv):  # 如果当前目录存在这些文件，就先删除
        os.remove(export_csv)
        print(export_csv + '  旧文件已删除')
    if os.path.isfile(export_excel):
        os.remove(export_excel)
        print(export_excel + '  旧文件已删除')
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
    if os.path.isfile(r'E:\03 Modis\03 Table\schema.ini'):
        os.remove(r'E:\03 Modis\03 Table\schema.ini')
    if os.path.isfile(r'E:\03 Modis\03 Table\NDVI 2017-2021 All.csv.xml'):
        os.remove(r'E:\03 Modis\03 Table\NDVI 2017-2021 All.csv.xml')


def SG_Filter(data_path, export_path, picture_path):
    inDataFrame = pd.read_excel(data_path)  # 读取Excel文件
    column_name = inDataFrame.columns
    outDataFrame = pd.DataFrame(columns=column_name)
    outDataFrame.iloc[:, 0] = inDataFrame.iloc[:, 0]  # 复制第一列
    x = list(range(0, inDataFrame.shape[1]-1))
    # 删除之前的NDVI SG 2017-2021.xlsx
    if os.path.isfile(export_path):  # 如果当前目录存在这些文件，就先删除
        os.remove(export_path)
        print('NDVI SG 2017-2021 All.xlsx  旧文件已删除')
    # 绘图前删除文件夹下图片
    del_files(picture_path)
    print('SG Filter NDVI 2017-2021 All  旧图片已删除')
    for row in range(inDataFrame.shape[0]):  # 遍历行
        data = list(inDataFrame.iloc[row, 1:])
        y = savgol_filter(data, 13, 3, mode="nearest")
        y[y < 0] = 0  # 将SG滤波后，小于0的数，赋值为0
        outDataFrame.iloc[row, 1:] = list(y)  # 保存SG滤波后的NDVI数据
        # 绘图
        plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率 1920x1080
        fig, ax = plt.subplots()  # 创建画布fig 创建绘图区域ax1 sdc
        ax.plot(x, data, color='tab:blue', linestyle='-', label='pre_filter NDVI')
        ax.plot(x, y, color='tab:red', linestyle='-', label="sg_filter NDVI")
        ax.set_xlabel('Date', fontsize=12)  # 设置x轴标签与字体大小
        ax.set_ylabel('NDVI', fontsize=12)
        ax.grid(alpha=0.5, linestyle=':')  # 开启虚线网格 透明度0.5
        plt.title("MODIS NDVI Pre-Filter and SG-Filter at " + str(outDataFrame.iloc[row, 0]) + ' 2017-2021' + '.png', fontsize=16)  # 设置图名
        plt.legend()
        fig.tight_layout()  # 自动调整布局 避免重叠
        # plt.show()
        plt.savefig(picture_path + ' ' + str(outDataFrame.iloc[row, 0]) + ' 2017-2021' + '.png')
        plt.close()
        print("MODIS NDVI Pre-Filter and SG-Filter at " + str(outDataFrame.iloc[row, 0]) + ' 2017-2021' + '.png' + '  正在保存SG滤波图片...')
    outDataFrame.to_excel(export_path, index=False)
    print('2017-2021  All SG滤波已完成，结果保存在 ' + export_path)


def ndvi_interpolate(data_path, export_path, picture_path):
    inDataFrame = pd.read_excel(data_path)  # 读取Excel文件
    site_name = inDataFrame.iloc[:, 0]
    column_name = inDataFrame.columns[1:]  # 获取列名
    # column_name = ['20'+name for name in column_name] # 列名转日期
    column_day = []  # 保存天数
    for date in column_name:
        column_day.append((ToDate(date)-ToDate('20170101')).days+1)  # ToDate将日期转为datetime格式的日期，计算出天数
    interpolate_x = range(1, column_day[-1]+1)
    column_date = [out_date_by_day(2017, date) for date in interpolate_x]  #以2017年1月1日为第一天，计算出日期
    column_date = list(column_date)
    column_date.insert(0, 'site')
    outDataFrame = pd.DataFrame(columns=column_date)  # 创建新的df，设置列名为插值后的天数（）
    outDataFrame.iloc[:, 0] = inDataFrame.iloc[:, 0]  # 复制第一列
    # 删除之前的NDVI SG 2017-2021.xlsx
    if os.path.isfile(export_path):  # 如果当前目录存在这些文件，就先删除
        os.remove(export_path)
        print('已删除旧的NDVI SG Interpolate 2017-2021 All.xlsx文件')
    # 绘图前删除文件夹下图片
    del_files(picture_path)
    print('Interpolation SG NDVI 2017-2021 All  旧图片已删除')
    for row in range(inDataFrame.shape[0]):  #遍历每一行进行插值
        x = column_day
        y = inDataFrame.iloc[row, 1:]
        # 进行三次样条拟合
        ipo3 = spi.splrep(x, y, k=3)  # 源数据点导入，生成参数
        interpolate_y = spi.splev(interpolate_x, ipo3)  # 根据观测点和样条参数，生成插值
        interpolate_y[interpolate_y < 0] = 0  # 插值后小于0的数赋值为0
        # 绘图
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        ax.plot(x, y, color='tab:blue', label='Before interpolation NDVI', )
        ax.plot(interpolate_x, interpolate_y, '.', color='tab:red', label='After interpolation')
        ax.set_ylabel('NDVI')
        ax.set_ylabel('Day')
        ax.set_title('NDVI and Interpolation NDVI at Site ' + site_name[row] + ' 2017-2021')
        ax.legend()
        fig.tight_layout()
        ax.grid(alpha=0.5, linestyle=':')
        # plt.show()
        plt.savefig(picture_path + ' ' + str(site_name[row]) + ' 2017-2021' + '.png')
        print(str(site_name[row]) + ' 2017-2021' + '.png' + ' 正在保存三次样条插值图片')
        plt.close()
        # 将插值结果保存到excel
        outDataFrame.iloc[row, 1:] = interpolate_y
    outDataFrame.to_excel(export_path, index=False)
    print('2017-2021 All NDVI 三次样条插值已完成，结果保存在' + export_path)


def sort_date(data_path, sentinel_path, out_excel_path):
    for i in range(6):   # 如果当前目录存在这些文件，就先删除
        if os.path.isfile(out_excel_path[i]):
            os.remove(out_excel_path[i])
    sentinel_df = pd.read_excel(sentinel_path)
    sentinel_time = sentinel_df.columns[1:]  # 哨兵过境时间
    dataFrame = pd.read_excel(data_path, header=0, index_col=0)  # 返回一个包含CSV文件数据的DataFrame  header=0设置第一行为列索引  index_col=0设置第一列为行索引
    dataFrameOutput = dataFrame.loc[:, sentinel_time]  # 筛选出所有的数据
    dataFrameOutput.to_excel(out_excel_path[5])
    # 需要获取哨兵每一年的过境时间
    year = []
    year.append([date for date in list(sentinel_time) if '2017' in date])
    year.append([date for date in list(sentinel_time) if '2018' in date])
    year.append([date for date in list(sentinel_time) if '2019' in date])
    year.append([date for date in list(sentinel_time) if '2020' in date])
    year.append([date for date in list(sentinel_time) if '2021' in date])
    dataFrame = dataFrameOutput  # 获取筛选后的数据
    for num in range(5):
        dataFrameOutput = dataFrame.loc[:, year[num]]  # 按列名进行查找
        dataFrameOutput.to_excel(out_excel_path[num])
    print('插值后的NDVI All数据按哨兵1号A星过境日期筛选完成，数据保存在'+out_excel_path[5].replace(' 2017-2021', ''))

if __name__ == '__main__':
    # 数据预处理
    # mask = r"E:\03 Modis\01 Shapefile\01 Area\MODIS_AREA.shp"
    inPath = r'E:\03 Modis\04 Data\01 HEGOUT'
    outPath = r"E:\03 Modis\04 Data\02 Preprocess"
    data_preprocessing(inPath, outPath)

    # 提取NDVI到点
    image_path = r"E:\03 Modis\04 Data\02 Preprocess"  # MODIS NDVI 预处理后TIF图像路径
    point_feature = r'E:\01 SMN\05 Shapefile\02 Modis Point All\modis_point_all.shp'
    # extract_multivalue_to_point(image_path, point_feature)

    # 属性表转Excel
    feature_table = r'E:\01 SMN\05 Shapefile\02 Modis Point All\modis_point_all.shp'
    export_csv = r'E:\03 Modis\03 Table\NDVI 2017-2021 All.csv'
    export_excel = r'E:\03 Modis\03 Table\NDVI 2017-2021 All.xlsx'
    feature_table_to_excel(feature_table, export_csv, export_excel)

    # SG滤波
    data_path = r'E:\03 Modis\03 Table\NDVI 2017-2021 All.xlsx'
    export_path = r'E:\03 Modis\03 Table\NDVI SG 2017-2021 All.xlsx'
    picture_path = r'E:\03 Modis\06 Picture\SG Filter NDVI 2017-2021 All/'
    SG_Filter(data_path, export_path, picture_path)

    # 三次样条插值
    data_path = r'E:\03 Modis\03 Table\NDVI SG 2017-2021 All.xlsx'  # SG滤波后的NDVI数据
    export_path = r'E:\03 Modis\03 Table\NDVI SG Interpolate 2017-2021 All.xlsx'
    picture_path = r'E:\03 Modis\06 Picture\Interpolation SG NDVI 2017-2021 All/'
    ndvi_interpolate(data_path, export_path, picture_path)

    # 按日期筛选
    data_path = r'E:\03 Modis\03 Table\NDVI SG Interpolate 2017-2021 All.xlsx'
    out_excel_path = [r'E:\03 Modis\03 Table\NDVI SG Interpolate SortDate 2017 All.xlsx', r'E:\03 Modis\03 Table\NDVI SG Interpolate SortDate 2018 All.xlsx', r'E:\03 Modis\03 Table\NDVI SG Interpolate SortDate 2019 All.xlsx', r'E:\03 Modis\03 Table\NDVI SG Interpolate SortDate 2020 All.xlsx', r'E:\03 Modis\03 Table\NDVI SG Interpolate SortDate 2021 All.xlsx', r'E:\03 Modis\03 Table\NDVI SG Interpolate SortDate 2017-2021 All.xlsx']
    sentinel_path = r'E:\02 Sentinel\04 Table\1x1 All\Sentinel VV 2017-2021 All.xlsx'  # 用于获取哨兵1号过境时间
    sort_date(data_path, sentinel_path, out_excel_path)

