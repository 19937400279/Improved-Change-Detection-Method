import pandas as pd
from pandas.api.types import CategoricalDtype
import os
from numpy import size

def arrange_smn_data(dir_path, sentinel_path, output_file_name, out_dir, output_file_name_all):
    # 删除旧数据
    del_files(out_dir)
    print(out_dir+'  旧文件已删除')

    # 获取5cm的土壤温度和湿度excel文件名
    file_SM_05cm = []
    file_TS_05cm = []
    for dir_path, dir_names, file_names in os.walk(dir_path):
        file_SM_05cm.extend([dir_path + '/' + file_name for file_name in file_names if 'SM_05cm' in file_name])  # 获取5cm土壤湿度文件名
        file_TS_05cm.extend([dir_path + '/' + file_name for file_name in file_names if 'TS_05cm' in file_name])  # 获取5cm土壤温度文件名

    # 获取哨兵过境时间
    pass_time_df = pd.read_excel(sentinel_path)
    pass_times = pass_time_df.columns[1:]
    pass_times = [time[:4] + '-' + time[4:6] + '-' + time[6:8] + ' 18:00:00' for time in pass_times]

    # 保存2018-2020土壤温度和湿度
    sm_all_dataFrameOutput = pd.DataFrame()  # 创建一个空的dataFrame
    ts_all_dataFrameOutput = pd.DataFrame()  # 创建一个空的dataFrame

    sort = ['SM_05cm_2018', 'SM_05cm_2019', 'SM_05cm_2020', 'TS_05cm_2018', 'TS_05cm_2019', 'TS_05cm_2020']
    year = ['2018-', '2019-', '2020-', '2018-', '2019-', '2020-']
    for num in range(3):  # 依次遍历2018-2020土壤湿度和土壤温度excel文件，筛选数据并保存为6个excel
        # 用于汇总筛选出的土壤湿度数据，行索引设置为哨兵过境时间，列索引设置为站点名
        row_name = []  # 行名
        column_name = []  # 列名
        for sm_file in file_SM_05cm:
            if sort[num] in sm_file:
                column_name.append((os.path.basename(sm_file)).split('_')[1])
        for pass_time in pass_times:
            if year[num] in pass_time:
                row_name.append(pass_time)
        sm_dataFrameOutput = pd.DataFrame(index=row_name, columns=column_name)  # 创建一个空的dataFrame

        for sm_file in file_SM_05cm:  # 遍历这一年所有土壤湿度文件
            if sort[num] in sm_file:  # 筛选土壤湿度文件，筛选出某一年的，然后依次打开文件，对日期进行筛选，然后将筛选的数据进行合并
                dataFrame = pd.read_csv(sm_file, header=0, index_col=0)  # 打开文件  header=0设置第一行为列索引  index_col=0设置第一列为行索引
                dataFrame = dataFrame[~dataFrame.index.duplicated(keep='first')]    # 删除行索引重复的行
                row_names = dataFrame.index  #获取行名称（对应采样日期）
                # 筛选数据，然后将数据汇总到总表
                for pass_time in pass_times:  # 遍历过境日期
                    if pass_time in row_names:  # 如果过境时间与表格行名称相等，则保存相应的数据
                        # 保存相应的数据，汇总到新表，并添加站点名和日期名
                        row_index = pass_time
                        column_index = (os.path.basename(sm_file)).split('_')[1]
                        sm_dataFrameOutput.loc[row_index, column_index] = float(dataFrame.loc[row_index, 'Soil_VWC_05'])
        sm_dataFrameOutput = pd.DataFrame(sm_dataFrameOutput.values.T, index=sm_dataFrameOutput.columns, columns=sm_dataFrameOutput.index)  # 表格转置 行列名互换
        cat_size_order = CategoricalDtype(["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12", "L13", "L14", "L15"], ordered=True)
        sm_dataFrameOutput.index = sm_dataFrameOutput.index.astype(cat_size_order)
        sm_dataFrameOutput = sm_dataFrameOutput.sort_index()  # 按指定的行名顺序进行排序
        column_name = list(sm_dataFrameOutput.columns)
        column_name = [name[:4]+name[5:7]+name[8:10] for name in column_name]
        sm_dataFrameOutput.columns = column_name  # 格式化列名
        sm_all_dataFrameOutput = pd.concat([sm_all_dataFrameOutput, sm_dataFrameOutput], axis=1)
        # 保存湿度表格
        sm_dataFrameOutput.to_excel(output_file_name[num])
        print(output_file_name[num] + '  文件写入成功')


        # 用于汇总筛选出的土壤温度数据，行索引设置为哨兵过境时间，列索引设置为站点名
        row_name = []  # 行名
        column_name = []  # 列名
        for ts_file in file_TS_05cm:
            if sort[num+3] in ts_file:
                column_name.append((os.path.basename(ts_file)).split('_')[1])
        for pass_time in pass_times:
            if year[num+3] in pass_time:
                row_name.append(pass_time)
        ts_dataFrameOutput = pd.DataFrame(index=row_name, columns=column_name)  # 创建一个空的dataFrame

        for ts_file in file_TS_05cm:  # 遍历这一年所有土壤湿度文件
            if sort[num+3] in ts_file:  # 筛选土壤湿度文件，筛选出某一年的，然后依次打开文件，对日期进行筛选，然后将筛选的数据进行合并
                dataFrame = pd.read_csv(ts_file, header=0, index_col=0)  # 打开文件  header=0设置第一行为列索引  index_col=0设置第一列为行索引
                dataFrame = dataFrame[~dataFrame.index.duplicated(keep='first')]    # 删除行索引重复的行
                row_names = dataFrame.index  #获取行名称（对应采样日期）
                # 筛选数据，然后将数据汇总到总表
                for pass_time in pass_times:  # 遍历过境日期
                    if pass_time in row_names:  # 如果过境时间与表格行名称相等，则保存相应的数据
                        # 保存相应的数据，汇总到新表，并添加站点名和日期名
                        row_index = pass_time
                        column_index = (os.path.basename(ts_file)).split('_')[1]
                        ts_dataFrameOutput.loc[row_index, column_index] = float(dataFrame.loc[row_index, 'Soil_TEM_05'])
        ts_dataFrameOutput = pd.DataFrame(ts_dataFrameOutput.values.T, index=ts_dataFrameOutput.columns, columns=ts_dataFrameOutput.index)  # 表格转置 行列名互换
        cat_size_order = CategoricalDtype(["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12", "M13", "M14", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12", "L13", "L14", "L15"], ordered=True)
        ts_dataFrameOutput.index = ts_dataFrameOutput.index.astype(cat_size_order)
        ts_dataFrameOutput = ts_dataFrameOutput.sort_index()  # 按指定的行名顺序进行排序
        column_name = list(ts_dataFrameOutput.columns)
        column_name = [name[:4]+name[5:7]+name[8:10] for name in column_name]
        ts_dataFrameOutput.columns = column_name  # 格式化列名
        ts_all_dataFrameOutput = pd.concat([ts_all_dataFrameOutput, ts_dataFrameOutput], axis=1)
        # 保存表格
        ts_dataFrameOutput.to_excel(output_file_name[num+3])
        print(output_file_name[num+3] + '  文件写入成功')

    # 保存2018-2020年的数据
    sm_all_dataFrameOutput.to_excel(output_file_name_all[0])
    print(output_file_name_all[0] + '  文件写入成功')
    ts_all_dataFrameOutput.to_excel(output_file_name_all[1])
    print(output_file_name_all[1] + '  文件写入成功')

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
    # 根据哨兵1号的过境时间，获取相应的实测土壤湿度和温度数据
    dir_path = r'E:\01 SMN\01 Data/'  # 实测数据所在文件夹
    sentinel_path = r'E:\02 Sentinel\04 Table\1x1 All\Sentinel VV 2017-2021 All.xlsx'
    output_file_name = [r'E:\01 SMN\04 Table\SM 2018 All.xlsx', r'E:\01 SMN\04 Table\SM 2019 All.xlsx', r'E:\01 SMN\04 Table\SM 2020 All.xlsx', r'E:\01 SMN\04 Table\TS 2018 All.xlsx', r'E:\01 SMN\04 Table\TS 2019 All.xlsx', r'E:\01 SMN\04 Table\TS 2020 All.xlsx']
    output_file_name_all = [r'E:\01 SMN\04 Table\SM 2018-2020 All.xlsx', r'E:\01 SMN\04 Table\TS 2018-2020 All.xlsx']
    out_dir = r'E:\01 SMN\04 Table/'
    arrange_smn_data(dir_path, sentinel_path, output_file_name, out_dir, output_file_name_all)  # 实测数据整理

