import arcpy
import os
import pandas as pd

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)

def xlsx_to_csv(original_excel, sentinel_point_out_path, modis_point_out_path):
    # 根据SMN的站点经纬度生成csv文件
    del_files(os.path.dirname(sentinel_point_out_path))
    print(os.path.dirname(sentinel_point_out_path) + '  旧文件清除完成')
    del_files(os.path.dirname(modis_point_out_path))
    print(os.path.dirname(modis_point_out_path) + '  旧文件清除完成')
    df = pd.read_excel(original_excel)
    df.to_csv(sentinel_point_out_path, sep=',', index=False)
    print(sentinel_point_out_path + '  csv文件保存成功')
    df.to_csv(modis_point_out_path, sep=',', index=False)
    print(modis_point_out_path + '.  csv文件保存成功')

def sentinel_point(dir_path, csv_path, table_name, shapefile_name):
    # 根据csv文件生成哨兵点要素
    arcpy.env.overwriteOutput = True
    arcpy.env.workspace = dir_path
    arcpy.TableToTable_conversion(csv_path, dir_path, table_name)  # 将csv文件转换成table并保存到dir_path
    table_path = dir_path + '/' + table_name + '.dbf'
    arcpy.management.XYTableToPoint(table_path, shapefile_name, "Longitude", "Latitude")  # 将table转换成point feature
    print(dir_path + shapefile_name + '.shp  shp文件保存成功')

def modis_point(dir_path, csv_path, table_name, shapefile_name):
    # 根据csv文件生成Modis点要素
    arcpy.env.overwriteOutput = True
    arcpy.env.workspace = dir_path
    arcpy.TableToTable_conversion(csv_path, dir_path, table_name)  # 将csv文件转换成table并保存到dir_path
    table_path = dir_path + '/' + table_name + '.dbf'
    arcpy.management.XYTableToPoint(table_path, shapefile_name, "Longitude", "Latitude")  # 将table转换成point feature
    print(dir_path + shapefile_name + '.shp  shp文件保存成功')


if __name__ == '__main__':
    # 将MSN的站点经纬度excel文件转为csv文件
    original_excel = r'E:\01 SMN\01 Data\lat-lon coordinates of stations in SMN-SDR_All.xlsx'
    sentinel_point_out_path = r'E:\01 SMN\05 Shapefile\01 Sentinel Point All\sentinel_point_all.csv'
    modis_point_out_path = r'E:\01 SMN\05 Shapefile\02 Modis Point All\modis_point_all.csv'
    # xlsx_to_csv(original_excel, sentinel_point_out_path, modis_point_out_path)

    # 生成哨兵point feature
    dir_path = r'E:\01 SMN\05 Shapefile\01 Sentinel Point All'
    csv_path = r'E:\01 SMN\05 Shapefile\01 Sentinel Point All\sentinel_point_all.csv'
    table_name = r'sentinel_point_table_all'
    shapefile_name = 'sentinel_point_all'
    # sentinel_point(dir_path, csv_path, table_name, shapefile_name)

    # 生成Modis point feature
    dir_path = r'E:\01 SMN\05 Shapefile\02 Modis Point All'
    csv_path = r'E:\01 SMN\05 Shapefile\02 Modis Point All\modis_point_all.csv'
    table_name = r'modis_point_table_all'
    shapefile_name = 'modis_point_all'
    # modis_point(dir_path, csv_path, table_name, shapefile_name)
