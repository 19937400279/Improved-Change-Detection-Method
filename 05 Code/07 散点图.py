# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)

def sm_vv_scatter(sm_path, vv_path, picture_dir):
    del_files(picture_dir)
    print(picture_dir + '  旧图片已删除')
    sm_df = pd.read_excel(sm_path, index_col=0, header=0)  # sm2019和2020年的数据
    date_time = sm_df.columns
    site_name = sm_df.index
    vv_2017_2021_df = pd.read_excel(vv_path, index_col=0, header=0)  # vv2017-2021年的数据
    vv_df = vv_2017_2021_df.loc[:, date_time]  # 获取2019和2020年的vv数据
    sm_list = sm_df.values  # 获取sm数值
    vv_list = vv_df.values
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    fig, ax = plt.subplots()
    ax.scatter(sm_list, vv_list)
    plt.title('2019 and 2020 year soil moisture and vv polarization backscattering coefficient at all sites', fontsize=16)
    ax.set_xlabel(r"Soil Moisture ($cm^3/cm^3$)", fontsize=14)
    ax.set_ylabel(r"VV Backscatter($linear$)", fontsize=14)
    fig.tight_layout()
    # plt.show()
    plt.savefig(picture_dir + 'sm vv all' + '.png')
    plt.close()
    print(picture_dir + 'sm vv all' + '.png' + '  图片保存成功')
    for row in range(sm_df.shape[0]):
        fig, ax = plt.subplots()
        ax.scatter(sm_list[row, :], vv_list[row, :])
        plt.title('2019 and 2020 year soil moisture and vv polarization backscattering coefficient at ' + site_name[row], fontsize=16)
        ax.set_xlabel(r"Soil Moisture ($cm^3/cm^3$)", fontsize=14)
        ax.set_ylabel(r"vv polarization backscattering coefficient ($linear$)", fontsize=14)
        fig.tight_layout()
        # plt.show()
        plt.savefig(picture_dir + 'sm vv ' + site_name[row] + '.png')
        plt.close()
        print(picture_dir + 'sm vv ' + site_name[row] + '.png' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹

def ndvi_vv_scatter_2017_2021(ndvi_path, vv_path, picture_dir):
    del_files(picture_dir)
    print(picture_dir + '  旧图片已删除')
    ndvi_2017_2021_df = pd.read_excel(ndvi_path, index_col=0, header=0)  # ndvi 2017和2021年的数据
    # date_time = ndvi_df.columns
    site_name = ndvi_2017_2021_df.index
    vv_2017_2021_df = pd.read_excel(vv_path, index_col=0, header=0)  # vv 2017-2021年的数据
    ndvi_list = ndvi_2017_2021_df.values
    vv_list = vv_2017_2021_df.values
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    fig, ax = plt.subplots()
    ax.scatter(ndvi_list, vv_list)
    plt.title('2017 and 2021 year NDVI and vv polarization backscattering coefficient at all sites', fontsize=16)
    ax.set_xlabel(r"NDVI", fontsize=14)
    ax.set_ylabel(r"vv polarization backscattering coefficient ($linear$)", fontsize=14)
    fig.tight_layout()
    # plt.show()
    plt.savefig(picture_dir + 'ndvi vv all' + '.png')
    plt.close()
    print(picture_dir + 'ndvi vv all' + '.png' + '  图片保存成功')
    for row in range(ndvi_2017_2021_df.shape[0]):
        fig, ax = plt.subplots()
        ax.scatter(ndvi_list[row, :], vv_list[row, :])
        plt.title('2017 and 2021 year NDVI and vv polarization backscattering coefficient at ' + site_name[row], fontsize=16)
        ax.set_xlabel(r"NDVI", fontsize=14)
        ax.set_ylabel(r"vv polarization backscattering coefficient ($linear$)", fontsize=14)
        fig.tight_layout()
        # plt.show()
        plt.savefig(picture_dir + 'ndvi vv ' + site_name[row] + '.png')
        plt.close()
        print(picture_dir + 'ndvi vv ' + site_name[row] + '.png' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹

def ndvi_vv_scatter(ndvi_path, vv_path, picture_dir):
    del_files(picture_dir)
    print(picture_dir + '  旧图片已删除')
    ndvi_2017_2021_df = pd.read_excel(ndvi_path, index_col=0, header=0)  # ndvi 2017和2021年的数据
    column_name = ndvi_2017_2021_df.columns
    date_time = [row_name for row_name in column_name if ('2019' in row_name or '2020' in row_name)]
    site_name = ndvi_2017_2021_df.index
    vv_2017_2021_df = pd.read_excel(vv_path, index_col=0, header=0)  # vv 2017-2021年的数据
    ndvi_list = ndvi_2017_2021_df[date_time].values  # 获取2019-2020年ndvi数值
    vv_list = vv_2017_2021_df[date_time].values  # 获取2019-2020年vv数值
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    # fig, ax = plt.subplots()
    # ax.scatter(ndvi_list, vv_list)
    # plt.title('2019 and 2020 year NDVI and vv polarization backscattering coefficient at all sites', fontsize=16)
    # ax.set_xlabel(r"NDVI", fontsize=14)
    # ax.set_ylabel(r"vv polarization backscattering coefficient ($linear$)", fontsize=14)
    # fig.tight_layout()
    # # plt.show()
    # plt.savefig(picture_dir + 'ndvi vv all' + '.png')
    # plt.close()
    print(picture_dir + 'ndvi vv all' + '.png' + '  图片保存成功')
    for row in range(ndvi_2017_2021_df.shape[0]):
        fig, ax = plt.subplots()
        ax.scatter(ndvi_list[row, :], vv_list[row, :], color='tab:orange')
        # plt.title('2019 and 2020 year NDVI and vv polarization backscattering coefficient at ' + site_name[row], fontsize=16)
        ax.set_xlabel(r"NDVI", fontsize=20)
        ax.set_ylabel(r"VV Backscatter ($linear$)", fontsize=20)
        xticks = np.linspace(0, 1.0, 21)
        plt.xticks(xticks)
        plt.yticks([-0.005, 0, 0.025, 0.05, 0.075,  0.1, 0.105])
        plt.xlim(-0.0, 1.0)
        plt.ylim(-0.005, 0.105)
        plt.tick_params(labelsize=15)
        fig.tight_layout()
        plt.grid(alpha=0.15, ls='--')
        fig.tight_layout()
        # plt.show()
        plt.savefig(picture_dir + 'ndvi vv ' + site_name[row] + '.png')
        plt.close()
        print(picture_dir + 'ndvi vv ' + site_name[row] + '.png' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹


def ndvi_sm_scatter(ndvi_path, sm_path, picture_dir):
    del_files(picture_dir)
    print(picture_dir + '  旧图片已删除')
    ndvi_2017_2021_df = pd.read_excel(ndvi_path, index_col=0, header=0)  # sm2019和2020年的数据
    site_name = ndvi_2017_2021_df.index
    sm_df = pd.read_excel(sm_path, index_col=0, header=0)  # vv2017-2021年的数据
    date_time = sm_df.columns
    ndvi_df = ndvi_2017_2021_df.loc[:, date_time]  # 获取2019和2020年的vv数据
    ndvi_list = ndvi_df.values
    sm_list = sm_df.values  # 获取sm数值
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    # fig, ax = plt.subplots()
    # ax.scatter(ndvi_list, sm_list, color='tab:green')
    # # plt.title('2019 and 2020 year NDVI and soil moisture at all sites', fontsize=16)
    # ax.set_xlabel(r"NDVI", fontsize=20)
    # ax.set_ylabel(r"Soil Moisture ($cm^3/cm^3$)", fontsize=20)
    # plt.xticks([-0.05, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.05])
    # plt.yticks([-0.02, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.52])
    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-0.02, 0.52)
    # plt.tick_params(labelsize=15)
    # plt.grid(alpha=0.25, ls='--')
    # fig.tight_layout()
    # plt.show()
    # # plt.savefig(picture_dir + 'ndvi sm all' + '.png')
    # plt.close()
    # print(picture_dir + 'ndvi sm all' + '.png' + '  图片保存成功')
    for row in range(ndvi_df.shape[0]):
        fig, ax = plt.subplots()
        ax.scatter(ndvi_list[row, :], sm_list[row, :], color='tab:green')
        # plt.title('2019 and 2020 year NDVI and soil moisture at ' + site_name[row], fontsize=16)
        ax.set_xlabel(r"NDVI", fontsize=20)
        ax.set_ylabel(r"Soil Moisture ($cm^3/cm^3$)", fontsize=20)
        xticks = np.linspace(0, 1.0, 21)
        plt.xticks(xticks)
        # plt.xticks([-0.05, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.05])
        plt.yticks([-0.02, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.32, 0.35, 0.4, 0.45, 0.5, 0.52])
        plt.xlim(-0.0, 1.0)
        plt.ylim(-0.02, 0.52)
        plt.tick_params(labelsize=15)
        fig.tight_layout()
        plt.grid(alpha=0.15, ls='--')
        # plt.show()
        plt.savefig(picture_dir + 'ndvi sm ' + site_name[row] + '.png')
        plt.close()
        print(picture_dir + 'ndvi sm ' + site_name[row] + '.png' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹

def ts_sm_scatter(ts_path, sm_path, picture_dir):
    del_files(picture_dir)
    print(picture_dir + '  旧图片已删除')
    ts_df = pd.read_excel(ts_path, index_col=0, header=0)  # sm2019和2020年的数据
    site_name = ts_df.index
    sm_df = pd.read_excel(sm_path, index_col=0, header=0)  # vv2017-2021年的数据
    ts_list = ts_df.values
    sm_list = sm_df.values  # 获取sm数值
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    fig, ax = plt.subplots()
    ax.scatter(ts_list, sm_list)
    plt.title('2019 and 2020 year soil temperature and soil moisture at all sites', fontsize=16)
    ax.set_xlabel(r"Soil Temperature ($℃$)", fontsize=14)
    ax.set_ylabel(r"Soil Moisture ($cm^3/cm^3$)", fontsize=14)
    fig.tight_layout()
    # plt.show()
    plt.savefig(picture_dir + 'ts sm all' + '.png')
    plt.close()
    print(picture_dir + 'ts sm all' + '.png' + '  图片保存成功')
    for row in range(ts_df.shape[0]):
        fig, ax = plt.subplots()
        ax.scatter(ts_list[row, :], sm_list[row, :])
        plt.title('2019 and 2020 year soil temperature and soil moisture at ' + site_name[row], fontsize=16)
        ax.set_xlabel(r"Soil Temperature ($℃$)", fontsize=14)
        ax.set_ylabel(r"Soil Moisture ($cm^3/cm^3$)", fontsize=14)
        fig.tight_layout()
        # plt.show()
        plt.savefig(picture_dir + 'ts sm ' + site_name[row] + '.png')
        plt.close()
        print(picture_dir + 'ts sm ' + site_name[row] + '.png' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹

def sm_dvv_scatter(sm_path, vv_path, picture_dir):
    del_files(picture_dir)
    print(picture_dir + '  旧图片已删除')
    sm_df = pd.read_excel(sm_path, index_col=0, header=0)  # sm2019和2020年的数据
    vv_2017_2021_df = pd.read_excel(vv_path, index_col=0, header=0)  # vv2017-2021年的数据
    site_name = sm_df.index
    date_time_2019 = [name for name in sm_df.columns if '2019' in name]
    date_time_2020 = [name for name in sm_df.columns if '2020' in name]
    sm = sm_df.values  # 获取2019-2020土壤湿度
    vv_2019 = np.array(vv_2017_2021_df.loc[:, date_time_2019].values)  # 获取2019年vv极化数据
    vv_2020 = np.array(vv_2017_2021_df.loc[:, date_time_2020].values)  # 获取2020年vv极化数据
    # 2019年和2020年各站点最小vv极化后向散射系数
    min_vv_2019 = np.min(vv_2019, 1)  # 2019年vv极化数据各站点最小值
    min_vv_2020 = np.min(vv_2020, 1)
    # 计算各站点的dvv
    dvv_2019 = vv_2019 - min_vv_2019[:, None]
    dvv_2020 = vv_2020 - min_vv_2020[:, None]
    dvv = np.hstack((dvv_2019, dvv_2020))
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    fig, ax = plt.subplots()
    ax.scatter(sm, dvv)
    plt.title('2019 and 2020 year soil moisture and difference vv polarization backscattering coefficient at all sites', fontsize=16)
    ax.set_xlabel(r"Soil Moisture ($cm^3/cm^3$)", fontsize=14)
    ax.set_ylabel(r"difference vv polarization backscattering coefficient ($linear$)", fontsize=14)
    fig.tight_layout()
    # plt.show()
    plt.savefig(picture_dir + 'sm dvv all' + '.png')
    plt.close()
    print(picture_dir + 'sm dvv all' + '.png' + '  图片保存成功')
    for row in range(sm_df.shape[0]):
        fig, ax = plt.subplots()
        ax.scatter(sm[row, :], dvv[row, :])
        plt.title('2019 and 2020 year soil moisture and difference vv polarization backscattering coefficient at ' + site_name[row], fontsize=16)
        ax.set_xlabel(r"Soil Moisture ($cm^3/cm^3$)", fontsize=14)
        ax.set_ylabel(r"difference vv polarization backscattering coefficient ($linear$)", fontsize=14)
        fig.tight_layout()
        # plt.show()
        plt.savefig(picture_dir + 'sm dvv ' + site_name[row] + '.png')
        plt.close()
        print(picture_dir + 'sm dvv ' + site_name[row] + '.png' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹

def ndvi_dvv_scatter(ndvi_path, vv_path, picture_dir):
    del_files(picture_dir)
    print(picture_dir + '  旧图片已删除')
    ndvi_2017_2021_df = pd.read_excel(ndvi_path, index_col=0, header=0)  # ndvi 2017和2021年的数据
    vv_2017_2021_df = pd.read_excel(vv_path, index_col=0, header=0)  # vv2017-2021年的数据
    site_name = ndvi_2017_2021_df.index
    date_time_2019 = [name for name in ndvi_2017_2021_df.columns if '2019' in name]
    date_time_2020 = [name for name in ndvi_2017_2021_df.columns if '2020' in name]
    date_time_2019_2020 = date_time_2019[:]
    date_time_2019_2020.extend(date_time_2020)
    ndvi = ndvi_2017_2021_df.loc[:, date_time_2019_2020].values  # 获取2019-2020 NDVI数据
    vv_2019 = np.array(vv_2017_2021_df.loc[:, date_time_2019].values)  # 获取2019年vv极化数据
    vv_2020 = np.array(vv_2017_2021_df.loc[:, date_time_2020].values)  # 获取2020年vv极化数据
    # 2019年和2020年各站点最小vv极化后向散射系数
    min_vv_2019 = np.min(vv_2019, 1)  # 2019年vv极化数据各站点最小值
    min_vv_2020 = np.min(vv_2020, 1)
    # 计算各站点的dvv
    dvv_2019 = vv_2019 - min_vv_2019[:, None]
    dvv_2020 = vv_2020 - min_vv_2020[:, None]
    dvv = np.hstack((dvv_2019, dvv_2020))
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    fig, ax = plt.subplots()
    ax.scatter(ndvi, dvv)
    plt.title('2019 and 2020 year NDVI and difference vv polarization backscattering coefficient at all sites', fontsize=16)
    ax.set_xlabel(r"NDVI", fontsize=14)
    ax.set_ylabel(r"difference vv polarization backscattering coefficient ($linear$)", fontsize=14)
    fig.tight_layout()
    # plt.show()
    plt.savefig(picture_dir + 'ndvi dvv all' + '.png')
    plt.close()
    print(picture_dir + 'ndvi dvv all' + '.png' + '  图片保存成功')
    for row in range(ndvi_2017_2021_df.shape[0]):
        fig, ax = plt.subplots()
        ax.scatter(ndvi[row, :], dvv[row, :])
        plt.title('2019 and 2020 year NDVI and difference vv polarization backscattering coefficient at ' + site_name[row], fontsize=16)
        ax.set_xlabel(r"NDVI", fontsize=14)
        ax.set_ylabel(r"difference vv polarization backscattering coefficient ($linear$)", fontsize=14)
        fig.tight_layout()
        # plt.show()
        plt.savefig(picture_dir + 'ndvi dvv ' + site_name[row] + '.png')
        plt.close()
        print(picture_dir + 'ndvi dvv ' + site_name[row] + '.png' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹

def ndvi_dvv_scatter_2017_2021(ndvi_path, vv_path, picture_dir):
    del_files(picture_dir)
    print(picture_dir + '  旧图片已删除')
    ndvi_2017_2021_df = pd.read_excel(ndvi_path, index_col=0, header=0)  # ndvi 2017和2021年的数据
    vv_2017_2021_df = pd.read_excel(vv_path, index_col=0, header=0)  # vv2017-2021年的数据
    site_name = ndvi_2017_2021_df.index
    ndvi = ndvi_2017_2021_df.values  # 获取2019-2020 NDVI数据
    date_time_2017 = [name for name in ndvi_2017_2021_df.columns if '2017' in name]
    date_time_2018 = [name for name in ndvi_2017_2021_df.columns if '2018' in name]
    date_time_2019 = [name for name in ndvi_2017_2021_df.columns if '2019' in name]
    date_time_2020 = [name for name in ndvi_2017_2021_df.columns if '2020' in name]
    date_time_2021 = [name for name in ndvi_2017_2021_df.columns if '2021' in name]
    vv_2017 = np.array(vv_2017_2021_df.loc[:, date_time_2017].values)  # 获取2019年vv极化数据
    vv_2018 = np.array(vv_2017_2021_df.loc[:, date_time_2018].values)  # 获取2020年vv极化数据
    vv_2019 = np.array(vv_2017_2021_df.loc[:, date_time_2019].values)  # 获取2019年vv极化数据
    vv_2020 = np.array(vv_2017_2021_df.loc[:, date_time_2020].values)  # 获取2020年vv极化数据
    vv_2021 = np.array(vv_2017_2021_df.loc[:, date_time_2021].values)  # 获取2020年vv极化数据
    # 2019年和2020年各站点最小vv极化后向散射系数
    min_vv_2017 = np.min(vv_2017, 1)  # 2017年vv极化数据各站点最小值
    min_vv_2018 = np.min(vv_2018, 1)
    min_vv_2019 = np.min(vv_2019, 1)  # 2019年vv极化数据各站点最小值
    min_vv_2020 = np.min(vv_2020, 1)
    min_vv_2021 = np.min(vv_2021, 1)
    # 计算各站点的dvv
    dvv_2017 = vv_2017 - min_vv_2017[:, None]
    dvv_2018 = vv_2018 - min_vv_2018[:, None]
    dvv_2019 = vv_2019 - min_vv_2019[:, None]
    dvv_2020 = vv_2020 - min_vv_2020[:, None]
    dvv_2021 = vv_2021 - min_vv_2021[:, None]
    dvv = np.hstack((dvv_2017, dvv_2018, dvv_2019, dvv_2020, dvv_2021))  # 拼接数据
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    fig, ax = plt.subplots()
    ax.scatter(ndvi, dvv)
    plt.title('2017 and 2021 year NDVI and difference vv polarization backscattering coefficient at all sites', fontsize=16)
    ax.set_xlabel(r"NDVI", fontsize=14)
    ax.set_ylabel(r"difference vv polarization backscattering coefficient ($linear$)", fontsize=14)
    fig.tight_layout()
    # plt.show()
    plt.savefig(picture_dir + 'ndvi dvv all' + '.png')
    plt.close()
    print(picture_dir + 'ndvi dvv all' + '.png' + '  图片保存成功')
    for row in range(ndvi_2017_2021_df.shape[0]):
        fig, ax = plt.subplots()
        ax.scatter(ndvi[row, :], dvv[row, :])
        plt.title('2017 and 2021 year NDVI and difference vv polarization backscattering coefficient at ' + site_name[row], fontsize=16)
        ax.set_xlabel(r"NDVI", fontsize=14)
        ax.set_ylabel(r"difference vv polarization backscattering coefficient ($linear$)", fontsize=14)
        fig.tight_layout()
        # plt.show()
        plt.savefig(picture_dir + 'ndvi dvv ' + site_name[row] + '.png')
        plt.close()
        print(picture_dir + 'ndvi dvv ' + site_name[row] + '.png' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹

def cdsm_cdvv_scatter(sm_path, vv_path, picture_dir):
    sm_df = pd.read_excel(sm_path, index_col=0, header=0)
    vv_2017_2021_df = pd.read_excel(vv_path, index_col=0, header=0)  # vv2017-2021年的数据
    site_name = sm_df.index
    date_time_2019 = [name for name in vv_2017_2021_df.columns if '2019' in name]
    date_time_2020 = [name for name in vv_2017_2021_df.columns if '2020' in name]
    date_time_2019_2020 = date_time_2019[:]
    date_time_2019_2020.extend(date_time_2020)
    vv_df = vv_2017_2021_df.loc[:, date_time_2019_2020]
    cdvv = np.zeros([vv_df.shape[0], vv_df.shape[1]-1])
    cdsm = np.zeros([vv_df.shape[0], vv_df.shape[1]-1])
    for row_num in range(vv_df.shape[0]):
        for column_num in range(vv_df.shape[1]-1):
            now_vv = vv_df.iloc[row_num, column_num]
            next_vv = vv_df.iloc[row_num, column_num+1]
            cdvv[row_num, column_num] = next_vv - now_vv
            now_sm = sm_df.iloc[row_num, column_num]
            next_sm = sm_df.iloc[row_num, column_num+1]
            cdsm[row_num, column_num] = next_sm - now_sm
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    fig, ax = plt.subplots()
    ax.scatter(cdvv, cdsm)
    plt.title('2019 and 2020 year continuous difference soil moisture and continuous difference vv polarization backscattering coefficient at all sites', fontsize=14)
    ax.set_xlabel(r"continuous difference soil moisture ($cm^3/cm^3$)", fontsize=12)
    ax.set_ylabel(r"continuous difference vv polarization backscattering coefficient ($linear$)", fontsize=12)
    fig.tight_layout()
    # plt.show()
    plt.savefig(picture_dir + 'cdsm csvv all' + '.png')
    plt.close()
    print(picture_dir + 'ndvi dvv all' + '.png' + '  图片保存成功')
    for row in range(vv_2017_2021_df.shape[0]):
        fig, ax = plt.subplots()
        ax.scatter(cdvv[row, :], cdsm[row, :])
        plt.title('2019 and 2020 year continuous difference soil moisture and continuous difference vv polarization backscattering coefficient at ' + site_name[row], fontsize=16)
        ax.set_xlabel(r"continuous difference soil moisture ($cm^3/cm^3$)", fontsize=12)
        ax.set_ylabel(r"continuous difference vv polarization backscattering coefficient ($linear$)", fontsize=12)
        fig.tight_layout()
        # plt.show()
        plt.savefig(picture_dir + 'cdsm csvv ' + site_name[row] + '.png')
        plt.close()
        print(picture_dir + 'cdsm csvv ' + site_name[row] + '.png' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹

def dsm_dvv_scatter(sm_path, vv_path, picture_dir):
    del_files(picture_dir)
    print(picture_dir + '  旧图片已删除')
    sm_df = pd.read_excel(sm_path, index_col=0, header=0)  # ndvi 2017和2021年的数据
    vv_2017_2021_df = pd.read_excel(vv_path, index_col=0, header=0)  # vv2017-2021年的数据
    site_name = sm_df.index
    date_time_2019 = [name for name in sm_df.columns if '2019' in name]
    date_time_2020 = [name for name in sm_df.columns if '2020' in name]
    sm_2019 = np.array(sm_df.loc[:, date_time_2019].values)  # 获取2019年sm数据
    sm_2020 = np.array(sm_df.loc[:, date_time_2020].values)  # 获取2020年sm数据
    vv_2019 = np.array(vv_2017_2021_df.loc[:, date_time_2019].values)  # 获取2019年vv极化数据
    vv_2020 = np.array(vv_2017_2021_df.loc[:, date_time_2020].values)  # 获取2020年vv极化数据
    # 2019年和2020年各站点最小vv极化后向散射系数
    min_sm_2019 = np.min(sm_2019, 1)  # 2019年vv极化数据各站点最小值
    min_sm_2020 = np.min(sm_2020, 1)
    min_vv_2019 = np.min(vv_2019, 1)  # 2019年vv极化数据各站点最小值
    min_vv_2020 = np.min(vv_2020, 1)
    # 计算各站点的dsm和dvv
    dsm_2019 = sm_2019 - min_sm_2019[:, None]
    dsm_2020 = sm_2020 - min_sm_2020[:, None]
    dvv_2019 = vv_2019 - min_vv_2019[:, None]
    dvv_2020 = vv_2020 - min_vv_2020[:, None]
    dsm = np.hstack((dsm_2019, dsm_2020))
    dvv = np.hstack((dvv_2019, dvv_2020))  # 拼接dvv数据

    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    fig, ax = plt.subplots()
    ax.scatter(dsm, dvv)
    plt.title('2019 and 2020 year difference soil moisture and difference vv polarization backscattering coefficient at all sites', fontsize=16)
    ax.set_xlabel(r"difference soil moisture ($cm^3/cm^3$)", fontsize=14)
    ax.set_ylabel(r"difference vv polarization backscattering coefficient ($linear$)", fontsize=14)
    fig.tight_layout()
    # plt.show()
    plt.savefig(picture_dir + 'dsm dvv all' + '.png')
    plt.close()
    print(picture_dir + 'dsm dvv all' + '.png' + '  图片保存成功')
    for row in range(sm_df.shape[0]):
        fig, ax = plt.subplots()
        ax.scatter(dsm[row, :], dvv[row, :])
        plt.title('2017 and 2021 year difference soil moisture and difference vv polarization backscattering coefficient at ' + site_name[row], fontsize=16)
        ax.set_xlabel(r"difference soil moisture ($cm^3/cm^3$)", fontsize=14)
        ax.set_ylabel(r"difference vv polarization backscattering coefficient ($linear$)", fontsize=14)
        fig.tight_layout()
        # plt.show()
        plt.savefig(picture_dir + 'dsm dvv ' + site_name[row] + '.png')
        plt.close()
        print(picture_dir + 'dsm dvv ' + site_name[row] + '.png' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹

def sdsm_sdvv_scatter(sm_path, vv_path, picture_dir):
    del_files(picture_dir)
    print(picture_dir + '  旧图片已删除')
    sm_df = pd.read_excel(sm_path, index_col=0, header=0)  # ndvi 2017和2021年的数据
    vv_2017_2021_df = pd.read_excel(vv_path, index_col=0, header=0)  # vv2017-2021年的数据
    site_name = sm_df.index
    date_time_2019 = [name for name in sm_df.columns if '2019' in name]
    date_time_2020 = [name for name in sm_df.columns if '2020' in name]
    sm_2019 = np.array(sm_df.loc[:, date_time_2019].values)  # 获取2019年sm数据
    sm_2020 = np.array(sm_df.loc[:, date_time_2020].values)  # 获取2020年sm数据
    vv_2019 = np.array(vv_2017_2021_df.loc[:, date_time_2019].values)  # 获取2019年vv极化数据
    vv_2020 = np.array(vv_2017_2021_df.loc[:, date_time_2020].values)  # 获取2020年vv极化数据
    # 2019年和2020年各站点最小vv极化后向散射系数
    min_sm_2019 = np.min(sm_2019, 1)  # 2019年vv极化数据各站点最小值
    max_sm_2019 = np.max(sm_2019, 1)  # 2019年vv极化数据各站点最大值
    min_sm_2020 = np.min(sm_2020, 1)
    max_sm_2020 = np.max(sm_2020, 1)
    min_vv_2019 = np.min(vv_2019, 1)  # 2019年vv极化数据各站点最小值
    max_vv_2019 = np.max(vv_2019, 1)
    min_vv_2020 = np.min(vv_2020, 1)
    max_vv_2020 = np.max(vv_2020, 1)
    # 计算各站点的dsm和dvv
    dsm_2019 = sm_2019 - min_sm_2019[:, None]
    sdsm_2019 = dsm_2019 / (max_sm_2019 - min_sm_2019)[:, None]  # 计算2019年土壤湿度变化斜率
    dsm_2020 = sm_2020 - min_sm_2020[:, None]
    sdsm_2020 = dsm_2020 / (max_sm_2020 - min_sm_2020)[:, None]
    dvv_2019 = vv_2019 - min_vv_2019[:, None]
    sdvv_2019 = dvv_2019 / (max_vv_2019 - min_vv_2019)[:, None]  # 计算2019年vv后向散射系数变化斜率
    dvv_2020 = vv_2020 - min_vv_2020[:, None]
    sdvv_2020 = dvv_2020 / (max_vv_2020 - min_vv_2020)[:, None]
    sdsm = np.hstack((sdsm_2019, sdsm_2020))
    sdvv = np.hstack((sdvv_2019, sdvv_2020))  # 拼接dvv数据
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    fig, ax = plt.subplots()
    ax.scatter(sdsm, sdvv)
    plt.title('2019 and 2020 year slope difference soil moisture and slope difference vv polarization backscattering coefficient at all sites', fontsize=16)
    ax.set_xlabel(r"slope difference soil moisture", fontsize=14)
    ax.set_ylabel(r"slope difference vv polarization backscattering coefficient", fontsize=14)
    fig.tight_layout()
    # plt.show()
    plt.savefig(picture_dir + 'sdsm sdvv all' + '.png')
    plt.close()
    print(picture_dir + 'sdsm sdvv all' + '.png' + '  图片保存成功')
    for row in range(sm_df.shape[0]):
        fig, ax = plt.subplots()
        ax.scatter(sdsm[row, :], sdvv[row, :])
        plt.title('2019 and 2020 year slope difference soil moisture and slope difference vv polarization backscattering coefficient at ' + site_name[row], fontsize=16)
        ax.set_xlabel(r"slope difference soil moisture ($cm^3/cm^3$)", fontsize=14)
        ax.set_ylabel(r"slope difference vv polarization backscattering coefficient ($linear$)", fontsize=14)
        fig.tight_layout()
        # plt.show()
        plt.savefig(picture_dir + 'sdsm sdvv ' + site_name[row] + '.png')
        plt.close()
        print(picture_dir + 'sdsm sdvv ' + site_name[row] + '.png' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹


def ndvi_vv_sm_double_yaxis(ndvi_path, vv_path, sm_path, picture_dir):
    del_files(picture_dir)
    print(picture_dir + '  旧图片已删除')
    ndvi_2017_2021_df = pd.read_excel(ndvi_path, index_col=0, header=0)  # ndvi 2017和2021年的数据
    column_name = ndvi_2017_2021_df.columns
    date_time = [row_name for row_name in column_name if ('2019' in row_name or '2020' in row_name)]
    site_name = ndvi_2017_2021_df.index
    vv_2017_2021_df = pd.read_excel(vv_path, index_col=0, header=0)  # vv 2017-2021年的数据
    sm_df = pd.read_excel(sm_path, index_col=0, header=0)
    ndvi_list = ndvi_2017_2021_df[date_time].values  # 获取2019-2020年ndvi数值
    vv_list = vv_2017_2021_df[date_time].values  # 获取2019-2020年vv数值
    sm_list = sm_df.values
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    fig, ax1 = plt.subplots()
    fig_1 = ax1.scatter(ndvi_list, vv_list, color='tab:blue', label='NDVI and VV')
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 0.52)
    plt.title('2019 and 2020 year NDVI and vv polarization backscattering coefficient and soilmoisture at all sites', fontsize=16)
    ax1.set_xlabel(r"NDVI", fontsize=14)
    ax1.set_ylabel(r"vv polarization backscattering coefficient ($linear$)", fontsize=14, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')  # 设置ax1的y轴刻度颜色
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"soil moisture ($cm^3/cm^3$)", fontsize=14, color='tab:red')
    fig_2 = ax2.scatter(ndvi_list, sm_list, color='tab:red', label='NDVI and SM')
    ax2.tick_params(axis='y', labelcolor='tab:red')  # 设置ax2的y轴刻度颜色
    plt.legend(handles=[fig_1, fig_2], fontsize=12)
    plt.ylim(-0.02, 0.52)
    fig.tight_layout()
    ax1.grid(alpha=0.3, linestyle='--')
    ax2.grid(alpha=0.3, linestyle='--')
    # plt.show()
    plt.savefig(picture_dir + 'ndvi vv sm all' + '.png')
    plt.close()
    print(picture_dir + 'ndvi vv sm all' + '.png' + '  图片保存成功')
    for row in range(ndvi_2017_2021_df.shape[0]):
        fig, ax1 = plt.subplots()
        fig_1 = ax1.scatter(ndvi_list[row, :], vv_list[row, :], color='tab:blue', label='NDVI and VV')
        plt.title('2019 and 2020 year NDVI and vv polarization backscattering coefficient and soilmoisture at ' + site_name[row], fontsize=16)
        plt.xticks(np.arange(0, 1.1, 0.2))
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 0.52)
        ax1.set_xlabel(r"NDVI", fontsize=14)
        ax1.set_ylabel(r"vv polarization backscattering coefficient ($linear$)", fontsize=14, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')  # 设置ax1的y轴刻度颜色
        ax2 = ax1.twinx()
        ax2.set_ylabel(r"soil moisture ($cm^3/cm^3$)", fontsize=14, color='tab:red')
        fig_2 = ax2.scatter(ndvi_list[row, :], sm_list[row, :], color='tab:red', label='NDVI and SM')
        ax2.tick_params(axis='y', labelcolor='tab:red')  # 设置ax2的y轴刻度颜色
        plt.legend(handles=[fig_1, fig_2], fontsize=12)
        plt.ylim(-0.02, 0.52)
        fig.tight_layout()
        ax1.grid(alpha=0.3, linestyle='--')
        ax2.grid(alpha=0.3, linestyle='--')
        # plt.show()
        plt.savefig(picture_dir + 'ndvi vv sm ' + site_name[row] + '.png')
        plt.close()
        print(picture_dir + 'ndvi vv sm ' + site_name[row] + '.png' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹

def ndvi_vv_sm_2019_double_yaxis(ndvi_path, vv_path, sm_path, picture_dir):
    del_files(picture_dir)
    print(picture_dir + '  旧图片已删除')
    ndvi_2017_2021_df = pd.read_excel(ndvi_path, index_col=0, header=0)  # ndvi 2017和2021年的数据
    column_name = ndvi_2017_2021_df.columns
    date_time = [row_name for row_name in column_name if ('2019' in row_name in row_name)]
    site_name = ndvi_2017_2021_df.index
    vv_2017_2021_df = pd.read_excel(vv_path, index_col=0, header=0)  # vv 2017-2021年的数据
    sm_df = pd.read_excel(sm_path, index_col=0, header=0)
    ndvi_list = ndvi_2017_2021_df[date_time].values  # 获取2019年ndvi数值
    vv_list = vv_2017_2021_df[date_time].values  # 获取2019年vv数值
    sm_list = sm_df[date_time].values
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    fig, ax1 = plt.subplots()
    fig_1 = ax1.scatter(ndvi_list, vv_list, color='tab:blue', label='NDVI and VV')
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 0.52)
    plt.title('2019 year NDVI and vv polarization backscattering coefficient and soilmoisture at all sites', fontsize=16)
    ax1.set_xlabel(r"NDVI", fontsize=14)
    ax1.set_ylabel(r"vv polarization backscattering coefficient ($linear$)", fontsize=14, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')  # 设置ax1的y轴刻度颜色
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"soil moisture ($cm^3/cm^3$)", fontsize=14, color='tab:red')
    fig_2 = ax2.scatter(ndvi_list, sm_list, color='tab:red', label='NDVI and SM')
    ax2.tick_params(axis='y', labelcolor='tab:red')  # 设置ax2的y轴刻度颜色
    plt.legend(handles=[fig_1, fig_2], fontsize=12)
    plt.ylim(-0.02, 0.52)
    fig.tight_layout()
    ax1.grid(alpha=0.3, linestyle='--')
    ax2.grid(alpha=0.3, linestyle='--')
    # plt.show()
    plt.savefig(picture_dir + 'ndvi vv sm all' + '.png')
    plt.close()
    print(picture_dir + 'ndvi vv sm all' + '.png' + '  图片保存成功')
    for row in range(ndvi_2017_2021_df.shape[0]):
        fig, ax1 = plt.subplots()
        fig_1 = ax1.scatter(ndvi_list[row, :], vv_list[row, :], color='tab:blue', label='NDVI and VV')
        plt.title('2019 year NDVI and vv polarization backscattering coefficient and soilmoisture at ' + site_name[row], fontsize=16)
        plt.xticks(np.arange(0, 1.1, 0.2))
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 0.52)
        ax1.set_xlabel(r"NDVI", fontsize=14)
        ax1.set_ylabel(r"vv polarization backscattering coefficient ($linear$)", fontsize=14, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')  # 设置ax1的y轴刻度颜色
        ax2 = ax1.twinx()
        ax2.set_ylabel(r"soil moisture ($cm^3/cm^3$)", fontsize=14, color='tab:red')
        fig_2 = ax2.scatter(ndvi_list[row, :], sm_list[row, :], color='tab:red', label='NDVI and SM')
        ax2.tick_params(axis='y', labelcolor='tab:red')  # 设置ax2的y轴刻度颜色
        plt.legend(handles=[fig_1, fig_2], fontsize=12)
        plt.ylim(-0.02, 0.52)
        fig.tight_layout()
        ax1.grid(alpha=0.3, linestyle='--')
        ax2.grid(alpha=0.3, linestyle='--')
        # plt.show()
        plt.savefig(picture_dir + 'ndvi vv sm ' + site_name[row] + '.png')
        plt.close()
        print(picture_dir + 'ndvi vv sm ' + site_name[row] + '.png' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹

def ndvi_vv_sm_2020_double_yaxis(ndvi_path, vv_path, sm_path, picture_dir):
    del_files(picture_dir)
    print(picture_dir + '  旧图片已删除')
    ndvi_2017_2021_df = pd.read_excel(ndvi_path, index_col=0, header=0)  # ndvi 2017和2021年的数据
    column_name = ndvi_2017_2021_df.columns
    date_time = [name for name in column_name if ('2020' in row_name in row_name)]
    site_name = ndvi_2017_2021_df.index
    vv_2017_2021_df = pd.read_excel(vv_path, index_col=0, header=0)  # vv 2017-2021年的数据
    sm_df = pd.read_excel(sm_path, index_col=0, header=0)
    ndvi_list = ndvi_2017_2021_df[date_time].values  # 获取2019年ndvi数值
    vv_list = vv_2017_2021_df[date_time].values  # 获取2019年vv数值
    sm_list = sm_df[date_time].values
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    fig, ax1 = plt.subplots()
    fig_1 = ax1.scatter(ndvi_list, vv_list, color='tab:blue', label='NDVI and VV')
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 0.52)
    plt.title('2020 year NDVI and vv polarization backscattering coefficient and soilmoisture at all sites', fontsize=16)
    ax1.set_xlabel(r"NDVI", fontsize=14)
    ax1.set_ylabel(r"vv polarization backscattering coefficient ($linear$)", fontsize=14, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')  # 设置ax1的y轴刻度颜色
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"soil moisture ($cm^3/cm^3$)", fontsize=14, color='tab:red')
    fig_2 = ax2.scatter(ndvi_list, sm_list, color='tab:red', label='NDVI and SM')
    ax2.tick_params(axis='y', labelcolor='tab:red')  # 设置ax2的y轴刻度颜色
    plt.legend(handles=[fig_1, fig_2], fontsize=12)
    plt.ylim(-0.02, 0.52)
    fig.tight_layout()
    ax1.grid(alpha=0.3, linestyle='--')
    ax2.grid(alpha=0.3, linestyle='--')
    # plt.show()
    plt.savefig(picture_dir + 'ndvi vv sm all' + '.png')
    plt.close()
    print(picture_dir + 'ndvi vv sm all' + '.png' + '  图片保存成功')
    for row in range(ndvi_2017_2021_df.shape[0]):
        fig, ax1 = plt.subplots()
        fig_1 = ax1.scatter(ndvi_list[row, :], vv_list[row, :], color='tab:blue', label='NDVI and VV')
        plt.title('2020 year NDVI and vv polarization backscattering coefficient and soilmoisture at ' + site_name[row], fontsize=16)
        plt.xticks(np.arange(0, 1.1, 0.2))
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 0.52)
        ax1.set_xlabel(r"NDVI", fontsize=14)
        ax1.set_ylabel(r"vv polarization backscattering coefficient ($linear$)", fontsize=14, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')  # 设置ax1的y轴刻度颜色
        ax2 = ax1.twinx()
        ax2.set_ylabel(r"soil moisture ($cm^3/cm^3$)", fontsize=14, color='tab:red')
        fig_2 = ax2.scatter(ndvi_list[row, :], sm_list[row, :], color='tab:red', label='NDVI and SM')
        ax2.tick_params(axis='y', labelcolor='tab:red')  # 设置ax2的y轴刻度颜色
        plt.legend(handles=[fig_1, fig_2], fontsize=12)
        plt.ylim(-0.02, 0.52)
        fig.tight_layout()
        ax1.grid(alpha=0.3, linestyle='--')
        ax2.grid(alpha=0.3, linestyle='--')
        # plt.show()
        plt.savefig(picture_dir + 'ndvi vv sm ' + site_name[row] + '.png')
        plt.close()
        print(picture_dir + 'ndvi vv sm ' + site_name[row] + '.png' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹

if __name__ == '__main__':
    sm_path = r'E:\04 Method\01 Table\sm_2019_2020_sort.xlsx'
    ts_path = r'E:\04 Method\01 Table\ts_2019_2020_sort.xlsx'
    vv_path = r'E:\04 Method\01 Table\vv_2017_2021_sort.xlsx'
    ndvi_path = r'E:\04 Method\01 Table\ndvi_2017_2021_sort.xlsx'

    picture_dir = r'E:\04 Method\02 Picture\02 散点图\单Y轴散点图\sm_vv/'  # 土壤湿度和vv极化散点图 2019-2020年
    # sm_vv_scatter(sm_path, vv_path, picture_dir)

    picture_dir = r'E:\04 Method\02 Picture\02 散点图\单Y轴散点图\sm_dvv/'
    # sm_dvv_scatter(sm_path, vv_path, picture_dir)

    picture_dir = r'E:\04 Method\02 Picture\02 散点图\单Y轴散点图\ndvi_vv/'  # NDVI和vv极化散点图 2019-2020年
    ndvi_vv_scatter(ndvi_path, vv_path, picture_dir)

    picture_dir = r'E:\04 Method\02 Picture\02 散点图\单Y轴散点图\ndvi_vv_2017_2021/'  # NDVI和vv极化散点图 2017-2021年
    # ndvi_vv_scatter_2017_2021(ndvi_path, vv_path, picture_dir)

    picture_dir = r'E:\04 Method\02 Picture\02 散点图\单Y轴散点图\ndvi_dvv/'
    # ndvi_dvv_scatter(ndvi_path, vv_path, picture_dir)

    picture_dir = r'E:\04 Method\02 Picture\02 散点图\单Y轴散点图\ndvi_dvv_2017_2021/'
    # ndvi_dvv_scatter_2017_2021(ndvi_path, vv_path, picture_dir)

    picture_dir = r'E:\04 Method\02 Picture\02 散点图\单Y轴散点图\ndvi_sm/'  # NDVI和SM
    # ndvi_sm_scatter(ndvi_path, sm_path, picture_dir)

    picture_dir = r'E:\04 Method\02 Picture\02 散点图\单Y轴散点图\ts_sm/'
    # ts_sm_scatter(ts_path, sm_path, picture_dir)

    picture_dir = r'E:\04 Method\02 Picture\02 散点图\单Y轴散点图\cdsm_cdvv/'  # 连续两个日期的后向散射系数差值与连续两个日期的土壤湿度差值
    # cdsm_cdvv_scatter(sm_path, vv_path, picture_dir)

    picture_dir = r'E:\04 Method\02 Picture\02 散点图\单Y轴散点图\dsm_dvv/'
    # dsm_dvv_scatter(sm_path, vv_path, picture_dir)

    picture_dir = r'E:\04 Method\02 Picture\02 散点图\单Y轴散点图\sdsm_sdvv/'  # 土壤湿度变化的斜率与vv后向散射系数变化的斜率 slope
    # sdsm_sdvv_scatter(sm_path, vv_path, picture_dir)

    picture_dir = r'E:\04 Method\02 Picture\02 散点图\双Y轴散点图\ndvi_sm_vv/'
    # ndvi_vv_sm_double_yaxis(ndvi_path, vv_path, sm_path, picture_dir)  # 双Y轴散点图

    picture_dir = r'E:\04 Method\02 Picture\02 散点图\双Y轴散点图\ndvi_sm_vv_2019/'
    # ndvi_vv_sm_2019_double_yaxis(ndvi_path, vv_path, sm_path, picture_dir)  # 双Y轴散点图 2019年

    picture_dir = r'E:\04 Method\02 Picture\02 散点图\双Y轴散点图\ndvi_sm_vv_2020/'
    # ndvi_vv_sm_2020_double_yaxis(ndvi_path, vv_path, sm_path, picture_dir)  # 双Y轴散点图 2019年



