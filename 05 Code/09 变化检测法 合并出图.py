# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde
import scipy.stats as ss
import math



def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


def wagner_cd_model(sm_path, vv_path, vh_path, picture_dir):
    sm_df = pd.read_excel(sm_path, index_col=0, header=0)  # ndvi 2017和2021年的数据
    vv_2017_2021_df = pd.read_excel(vv_path, index_col=0, header=0)  # vv2017-2021年的数据
    vh_2017_2021_df = pd.read_excel(vh_path, index_col=0, header=0)  # vv2017-2021年的数据
    date_time_2019 = [name for name in sm_df.columns if '2019' in name]
    date_time_2020 = [name for name in sm_df.columns if '2020' in name]
    a = 1.0458
    b = 0.0022
    sm_2019 = np.array(sm_df.loc[:, date_time_2019].values * a - b)  # 获取2019年sm数据，并进行校正
    sm_2020 = np.array(sm_df.loc[:, date_time_2020].values * a - b)  # 获取2020年sm数据，并进行校正

    vv_2019 = np.array(vv_2017_2021_df.loc[:, date_time_2019].values)  # 获取2019年vv极化数据
    vv_2020 = np.array(vv_2017_2021_df.loc[:, date_time_2020].values)  # 获取2020年vv极化数据
    vh_2019 = np.array(vh_2017_2021_df.loc[:, date_time_2019].values)  # 获取2019年vv极化数据
    vh_2020 = np.array(vh_2017_2021_df.loc[:, date_time_2020].values)  # 获取2020年vv极化数据
    # 2019年和2020年各站点最小vv极化后向散射系数
    min_sm_2019 = np.min(sm_2019, 1)  # 2019年vv极化数据各站点最小值
    max_sm_2019 = np.max(sm_2019, 1)  # 2019年vv极化数据各站点最大值
    min_sm_2020 = np.min(sm_2020, 1)
    max_sm_2020 = np.max(sm_2020, 1)
    min_vv_2019 = np.min(vv_2019, 1)  # 2019年vv极化数据各站点最小值
    max_vv_2019 = np.max(vv_2019, 1)
    min_vv_2020 = np.min(vv_2020, 1)
    max_vv_2020 = np.max(vv_2020, 1)
    min_vh_2019 = np.min(vh_2019, 1)  # 2019年vv极化数据各站点最小值
    max_vh_2019 = np.max(vh_2019, 1)
    min_vh_2020 = np.min(vh_2020, 1)
    max_vh_2020 = np.max(vh_2020, 1)
    # 基于变化检测法反演土壤湿度
    min_sm = min_sm_2019[:, None]
    max_sm = max_sm_2019[:, None]
    min_vv = min_vv_2019[:, None]
    max_vv = max_vv_2019[:, None]
    now_vv = vv_2019
    min_vh = min_vh_2019[:, None]
    max_vh = max_vh_2019[:, None]
    now_vh = vh_2019
    inversion_sm_2019_vv = (now_vv - min_vv) / (max_vv - min_vv) * (max_sm - min_sm) + min_sm
    inversion_sm_2019_vh = (now_vh - min_vh) / (max_vh - min_vh) * (max_sm - min_sm) + min_sm
    min_sm = min_sm_2020[:, None]
    max_sm = max_sm_2020[:, None]
    min_vv = min_vv_2020[:, None]
    max_vv = max_vv_2020[:, None]
    min_vh = min_vh_2020[:, None]
    max_vh = max_vh_2020[:, None]
    now_vv = vv_2020
    now_vh = vh_2020
    inversion_sm_2020_vv = (now_vv - min_vv) / (max_vv - min_vv) * (max_sm - min_sm) + min_sm
    inversion_sm_2020_vh = (now_vh - min_vh) / (max_vh - min_vh) * (max_sm - min_sm) + min_sm
    inversion_sm_vv = np.hstack((inversion_sm_2019_vv, inversion_sm_2020_vv))  # 拼接数据
    inversion_sm_vh = np.hstack((inversion_sm_2019_vh, inversion_sm_2020_vh))  # 拼接数据
    measured_sm = np.hstack((sm_2019, sm_2020))

    return inversion_sm_vv, inversion_sm_vh, measured_sm

    plt.rcParams['figure.figsize'] = (19.2, 9.2)  # 绘图分辨率
    fig, axs = plt.subplots(1, 2)
    for ax in axs.ravel():
        ax.set_xticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
        ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
        ax.set_xlim(0.0, 0.52)
        ax.set_ylim(0.0, 0.52)
        ax.plot([0.0, 0.52], [0.0, 0.52], ls="--", c=".3", linewidth=1.0, alpha=0.8)
        ax.tick_params(labelsize=13)
    # 绘制子图1
    x = measured_sm.flatten()
    y = inversion_sm_vh.flatten()
    xy = np.vstack([x, y])  # 将两个维度的数据叠加
    z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[0].scatter(x, y, c=z, s=30, cmap='Spectral_r')
    axs[0].set_xlabel(r"Measured soil moisture ($cm^3/cm^3$)", fontsize=18)
    axs[0].set_ylabel(r"Soil moisture retrieval based on VH backscatter ($cm^3/cm^3$)", fontsize=18)
    r = ss.pearsonr(measured_sm.flatten(), inversion_sm_vh.flatten())[0]
    R2 = r2_score(measured_sm.flatten(), inversion_sm_vh.flatten())
    RMSE = sqrt(mean_squared_error(measured_sm.flatten(), inversion_sm_vh.flatten()))
    MAE = mean_absolute_error(measured_sm.flatten(), inversion_sm_vh.flatten())
    MSE = mean_squared_error(measured_sm.flatten(), inversion_sm_vh.flatten())
    evaluation_indicator = ' (a)  R:{:.3f}  R2:{:.3f}  RMSE:{:.3f}  MAE:{:.3f}  MSE:{:.3f}'.format(r, R2, RMSE, MAE, MSE)
    axs[0].text(0.005, 0.975, evaluation_indicator, transform=axs[0].transAxes, fontdict={'size': '15'})
    # 绘制子图2
    x = measured_sm.flatten()
    y = inversion_sm_vv.flatten()
    xy = np.vstack([x, y])  # 将两个维度的数据叠加
    z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[1].scatter(x, y, c=z, s=30, cmap='Spectral_r')
    axs[1].set_xlabel(r"Measured soil moisture ($cm^3/cm^3$)", fontsize=18)
    axs[1].set_ylabel(r"Soil moisture retrieval based on VV backscatter ($cm^3/cm^3$)", fontsize=18)
    r = ss.pearsonr(measured_sm.flatten(), inversion_sm_vv.flatten())[0]
    R2 = r2_score(measured_sm.flatten(), inversion_sm_vv.flatten())
    RMSE = sqrt(mean_squared_error(measured_sm.flatten(), inversion_sm_vv.flatten()))
    MAE = mean_absolute_error(measured_sm.flatten(), inversion_sm_vv.flatten())
    MSE = mean_squared_error(measured_sm.flatten(), inversion_sm_vv.flatten())
    evaluation_indicator = ' (b)  R:{:.3f}  R2:{:.3f}  RMSE:{:.3f}  MAE:{:.3f}  MSE:{:.3f}'.format(r, R2, RMSE, MAE, MSE)
    axs[1].text(0.005, 0.975, evaluation_indicator, transform=axs[1].transAxes, fontdict={'size': '15'})
    fig.tight_layout()
    # plt.show()
    plt.savefig(picture_dir + '01 Wagner提出的CD反演法' + '.png')


def zribi_cd_model(ndvi_path, sm_path, vv_path, vh_path, picture_dir):
    sm_df = pd.read_excel(sm_path, index_col=0, header=0)
    vv_2017_2021_df = pd.read_excel(vv_path, index_col=0, header=0)  # vv 2017-2021年的数据
    vh_2017_2021_df = pd.read_excel(vh_path, index_col=0, header=0)  # vh 2017-2021年的数据
    ndvi_2017_2021_df = pd.read_excel(ndvi_path, index_col=0, header=0)  # vv 2017-2021年的数据
    site_name = sm_df.index
    date_time_2019 = [name for name in sm_df.columns if '2019' in name]
    date_time_2020 = [name for name in sm_df.columns if '2020' in name]
    sm_2019 = np.array(sm_df.loc[:, date_time_2019].values)  # 获取2019年sm数据
    sm_2020 = np.array(sm_df.loc[:, date_time_2020].values)  # 获取2020年sm数据
    vv_2019 = np.array(vv_2017_2021_df.loc[:, date_time_2019].values)  # 获取2019年vv极化数据
    vv_2020 = np.array(vv_2017_2021_df.loc[:, date_time_2020].values)  # 获取2020年vv极化数据
    vh_2019 = np.array(vh_2017_2021_df.loc[:, date_time_2019].values)  # 获取2019年vh极化数据
    vh_2020 = np.array(vh_2017_2021_df.loc[:, date_time_2020].values)  # 获取2020年vh极化数据
    ndvi_2019 = np.array(ndvi_2017_2021_df.loc[:, date_time_2019].values)  # 获取2019年ndvi极化数据
    ndvi_2020 = np.array(ndvi_2017_2021_df.loc[:, date_time_2019].values)  # 获取2020年ndvi极化数据

    # 基于NDVI分段，0.25, 0.5, 1.0，获取各区间的最大和最小土壤湿度
    intervals = 3
    min_vv_2019 = np.zeros([sm_df.shape[0], intervals])
    max_vv_2019 = np.zeros([sm_df.shape[0], intervals])
    min_vv_2020 = np.zeros([sm_df.shape[0], intervals])
    max_vv_2020 = np.zeros([sm_df.shape[0], intervals])
    min_vh_2019 = np.zeros([sm_df.shape[0], intervals])
    max_vh_2019 = np.zeros([sm_df.shape[0], intervals])
    min_vh_2020 = np.zeros([sm_df.shape[0], intervals])
    max_vh_2020 = np.zeros([sm_df.shape[0], intervals])
    min_sm_2019 = np.zeros([sm_df.shape[0], intervals])
    max_sm_2019 = np.zeros([sm_df.shape[0], intervals])
    min_sm_2020 = np.zeros([sm_df.shape[0], intervals])
    max_sm_2020 = np.zeros([sm_df.shape[0], intervals])

    # 2019年 各站点 各NDVI分段区间 VV和VH极化最大值最小值
    for site in range(sm_df.shape[0]):
        vv_temp_01 = np.zeros((0,))
        vv_temp_02 = np.zeros((0,))
        vv_temp_03 = np.zeros((0,))
        vh_temp_01 = np.zeros((0,))
        vh_temp_02 = np.zeros((0,))
        vh_temp_03 = np.zeros((0,))
        for date in range(len(date_time_2019)):
            if ndvi_2019[site, date] <= 0.25:
                vv_temp_01 = np.append(vv_temp_01, vv_2019[site, date])  # 追加值
                vh_temp_01 = np.append(vh_temp_01, vh_2019[site, date])
            elif ndvi_2019[site, date] <= 0.5:
                vv_temp_02 = np.append(vv_temp_02, vv_2019[site, date])  # 追加值
                vh_temp_02 = np.append(vh_temp_02, vh_2019[site, date])
            elif ndvi_2019[site, date] <= 1.0:
                vv_temp_03 = np.append(vv_temp_03, vv_2019[site, date])
                vh_temp_03 = np.append(vh_temp_03, vh_2019[site, date])
            else:
                print('Error')
        if vv_temp_01.any():
            min_vv_2019[site, 0] = np.min(vv_temp_01)
            max_vv_2019[site, 0] = np.max(vv_temp_01)
        if vh_temp_01.any():
            min_vh_2019[site, 0] = np.min(vh_temp_01)
            max_vh_2019[site, 0] = np.max(vh_temp_01)
        if vv_temp_02.any():
            min_vv_2019[site, 1] = np.min(vv_temp_02)
            max_vv_2019[site, 1] = np.max(vv_temp_02)
        if vh_temp_02.any():
            min_vh_2019[site, 1] = np.min(vh_temp_02)
            max_vh_2019[site, 1] = np.max(vh_temp_02)
        if vv_temp_03.any():
            min_vv_2019[site, 2] = np.min(vv_temp_03)
            max_vv_2019[site, 2] = np.max(vv_temp_03)
        if vh_temp_03.any():
            min_vh_2019[site, 2] = np.min(vh_temp_03)
            max_vh_2019[site, 2] = np.max(vh_temp_03)

        min_sm_2019[site, 0] = np.min(sm_2019[site, :])
        max_sm_2019[site, 0] = np.max(sm_2019[site, :])


    # 2020年 各站点 各NDVI分段区间 VV和VH极化最大值最小值
    for site in range(sm_df.shape[0]):
        vv_temp_01 = np.zeros((0,))
        vv_temp_02 = np.zeros((0,))
        vv_temp_03 = np.zeros((0,))
        vh_temp_01 = np.zeros((0,))
        vh_temp_02 = np.zeros((0,))
        vh_temp_03 = np.zeros((0,))
        for date in range(len(date_time_2020)):
            if ndvi_2020[site, date] <= 0.25:
                vv_temp_01 = np.append(vv_temp_01, vv_2020[site, date])  # 追加值
                vh_temp_01 = np.append(vh_temp_01, vh_2020[site, date])
            elif ndvi_2020[site, date] <= 0.5:
                vv_temp_02 = np.append(vv_temp_02, vv_2020[site, date])  # 追加值
                vh_temp_02 = np.append(vh_temp_02, vh_2020[site, date])
            elif ndvi_2020[site, date] <= 1.0:
                vv_temp_03 = np.append(vv_temp_03, vv_2020[site, date])
                vh_temp_03 = np.append(vh_temp_03, vh_2020[site, date])
            else:
                print('Error')
        if vv_temp_01.any():
            min_vv_2020[site, 0] = np.min(vv_temp_01)
            max_vv_2020[site, 0] = np.max(vv_temp_01)
        if vh_temp_01.any():
            min_vh_2020[site, 0] = np.min(vh_temp_01)
            max_vh_2020[site, 0] = np.max(vh_temp_01)
        if vv_temp_02.any():
            min_vv_2020[site, 1] = np.min(vv_temp_02)
            max_vv_2020[site, 1] = np.max(vv_temp_02)
        if vh_temp_02.any():
            min_vh_2020[site, 1] = np.min(vh_temp_02)
            max_vh_2020[site, 1] = np.max(vh_temp_02)
        if vv_temp_03.any():
            min_vv_2020[site, 2] = np.min(vv_temp_03)
            max_vv_2020[site, 2] = np.max(vv_temp_03)
        if vh_temp_03.any():
            min_vh_2020[site, 2] = np.min(vh_temp_03)
            max_vh_2020[site, 2] = np.max(vh_temp_03)

        min_sm_2020[site, 0] = np.min(sm_2020[site, :])
        max_sm_2020[site, 0] = np.max(sm_2020[site, :])

    # 反演土壤湿度
    inversion_sm_2019_vv = np.zeros([sm_df.shape[0], len(date_time_2019)])
    inversion_sm_2019_vh = np.zeros([sm_df.shape[0], len(date_time_2019)])
    inversion_sm_2020_vv = np.zeros([sm_df.shape[0], len(date_time_2020)])
    inversion_sm_2020_vh = np.zeros([sm_df.shape[0], len(date_time_2020)])

    # NDVI分段反演2019年土壤湿度
    for site in range(sm_df.shape[0]):
        for date in range(len(date_time_2019)):
            if ndvi_2019[site, date] <= 0.25:
                ndvi_index = 0
                now_vv = vv_2019[site, date]
                min_vv = min_vv_2019[site, ndvi_index]
                max_vv = max_vv_2019[site, ndvi_index]
                now_vh = vh_2019[site, date]
                min_vh = min_vh_2019[site, ndvi_index]
                max_vh = max_vh_2019[site, ndvi_index]
                min_sm = min_sm_2019[site, 0]
                max_sm = max_sm_2019[site, 0]
                inversion_sm_2019_vv[site, date] = (now_vv - min_vv) / (max_vv - min_vv) * (max_sm - min_sm) + min_sm
                inversion_sm_2019_vh[site, date] = (now_vh - min_vh) / (max_vh - min_vh) * (max_sm - min_sm) + min_sm
            elif ndvi_2019[site, date] <= 0.5:
                ndvi_index = 1
                now_vv = vv_2019[site, date]
                min_vv = min_vv_2019[site, ndvi_index]
                max_vv = max_vv_2019[site, ndvi_index]
                now_vh = vh_2019[site, date]
                min_vh = min_vh_2019[site, ndvi_index]
                max_vh = max_vh_2019[site, ndvi_index]
                min_sm = min_sm_2019[site, 0]
                max_sm = max_sm_2019[site, 0]
                inversion_sm_2019_vv[site, date] = (now_vv - min_vv) / (max_vv - min_vv) * (max_sm - min_sm) + min_sm
                inversion_sm_2019_vh[site, date] = (now_vh - min_vh) / (max_vh - min_vh) * (max_sm - min_sm) + min_sm
            elif ndvi_2019[site, date] <= 1.0:
                ndvi_index = 2
                now_vv = vv_2019[site, date]
                min_vv = min_vv_2019[site, ndvi_index]
                max_vv = max_vv_2019[site, ndvi_index]
                now_vh = vh_2019[site, date]
                min_vh = min_vh_2019[site, ndvi_index]
                max_vh = max_vh_2019[site, ndvi_index]
                min_sm = min_sm_2019[site, 0]
                max_sm = max_sm_2019[site, 0]
                inversion_sm_2019_vv[site, date] = (now_vv - min_vv) / (max_vv - min_vv) * (max_sm - min_sm) + min_sm
                inversion_sm_2019_vh[site, date] = (now_vh - min_vh) / (max_vh - min_vh) * (max_sm - min_sm) + min_sm

        # NDVI分段反演2020年土壤湿度
        for site in range(sm_df.shape[0]):
            for date in range(len(date_time_2020)):
                if ndvi_2020[site, date] <= 0.25:
                    ndvi_index = 0
                    now_vv = vv_2020[site, date]
                    min_vv = min_vv_2020[site, ndvi_index]
                    max_vv = max_vv_2020[site, ndvi_index]
                    now_vh = vh_2020[site, date]
                    min_vh = min_vh_2020[site, ndvi_index]
                    max_vh = max_vh_2020[site, ndvi_index]
                    min_sm = min_sm_2020[site, 0]
                    max_sm = max_sm_2020[site, 0]
                    inversion_sm_2020_vv[site, date] = (now_vv - min_vv) / (max_vv - min_vv) * (
                            max_sm - min_sm) + min_sm
                    inversion_sm_2020_vh[site, date] = (now_vh - min_vh) / (max_vh - min_vh) * (
                            max_sm - min_sm) + min_sm
                elif ndvi_2020[site, date] <= 0.5:
                    ndvi_index = 1
                    now_vv = vv_2020[site, date]
                    min_vv = min_vv_2020[site, ndvi_index]
                    max_vv = max_vv_2020[site, ndvi_index]
                    now_vh = vh_2020[site, date]
                    min_vh = min_vh_2020[site, ndvi_index]
                    max_vh = max_vh_2020[site, ndvi_index]
                    min_sm = min_sm_2020[site, 0]
                    max_sm = max_sm_2020[site, 0]
                    inversion_sm_2020_vv[site, date] = (now_vv - min_vv) / (max_vv - min_vv) * (
                            max_sm - min_sm) + min_sm
                    inversion_sm_2020_vh[site, date] = (now_vh - min_vh) / (max_vh - min_vh) * (
                            max_sm - min_sm) + min_sm
                elif ndvi_2020[site, date] <= 1.0:
                    ndvi_index = 2
                    now_vv = vv_2020[site, date]
                    min_vv = min_vv_2020[site, ndvi_index]
                    max_vv = max_vv_2020[site, ndvi_index]
                    now_vh = vh_2020[site, date]
                    min_vh = min_vh_2020[site, ndvi_index]
                    max_vh = max_vh_2020[site, ndvi_index]
                    min_sm = min_sm_2020[site, 0]
                    max_sm = max_sm_2020[site, 0]
                    inversion_sm_2020_vv[site, date] = (now_vv - min_vv) / (max_vv - min_vv) * (
                            max_sm - min_sm) + min_sm
                    inversion_sm_2020_vh[site, date] = (now_vh - min_vh) / (max_vh - min_vh) * (
                            max_sm - min_sm) + min_sm

    inversion_sm_vv = np.hstack((inversion_sm_2019_vv, inversion_sm_2020_vv))  # 拼接数据
    inversion_sm_vh = np.hstack((inversion_sm_2019_vh, inversion_sm_2020_vh))  # 拼接数据
    measured_sm = np.hstack((sm_2019, sm_2020))

    return inversion_sm_vv, inversion_sm_vh, measured_sm

    plt.rcParams['figure.figsize'] = (19.2, 9.2)  # 绘图分辨率
    fig, axs = plt.subplots(1, 2)
    for ax in axs.ravel():
        ax.set_xticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
        ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
        ax.set_xlim(0.0, 0.52)
        ax.set_ylim(0.0, 0.52)
        ax.plot([0.0, 0.52], [0.0, 0.52], ls="--", c=".3", linewidth=1.0, alpha=0.8)
        ax.tick_params(labelsize=13)
    # 绘制子图1
    x = measured_sm.flatten()
    y = inversion_sm_vh.flatten()
    xy = np.vstack([x, y])  # 将两个维度的数据叠加
    z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[0].scatter(x, y, c=z, s=30, cmap='Spectral_r')
    axs[0].set_xlabel(r"Measured soil moisture ($cm^3/cm^3$)", fontsize=18)
    axs[0].set_ylabel(r"Soil moisture retrieval based on VH backscatter ($cm^3/cm^3$)", fontsize=18)
    r = ss.pearsonr(measured_sm.flatten(), inversion_sm_vh.flatten())[0]
    R2 = r2_score(measured_sm.flatten(), inversion_sm_vh.flatten())
    RMSE = sqrt(mean_squared_error(measured_sm.flatten(), inversion_sm_vh.flatten()))
    MAE = mean_absolute_error(measured_sm.flatten(), inversion_sm_vh.flatten())
    MSE = mean_squared_error(measured_sm.flatten(), inversion_sm_vh.flatten())
    evaluation_indicator = ' (c)  R:{:.3f}  R2:{:.3f}  RMSE:{:.3f}  MAE:{:.3f}  MSE:{:.3f}'.format(r, R2, RMSE, MAE, MSE)
    axs[0].text(0.005, 0.975, evaluation_indicator, transform=axs[0].transAxes, fontdict={'size': '15'})
    # 绘制子图2
    x = measured_sm.flatten()
    y = inversion_sm_vv.flatten()
    xy = np.vstack([x, y])  # 将两个维度的数据叠加
    z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[1].scatter(x, y, c=z, s=30, cmap='Spectral_r')
    axs[1].set_xlabel(r"Measured soil moisture ($cm^3/cm^3$)", fontsize=18)
    axs[1].set_ylabel(r"Soil moisture retrieval based on VV backscatter ($cm^3/cm^3$)", fontsize=18)
    r = ss.pearsonr(measured_sm.flatten(), inversion_sm_vv.flatten())[0]
    R2 = r2_score(measured_sm.flatten(), inversion_sm_vv.flatten())
    RMSE = sqrt(mean_squared_error(measured_sm.flatten(), inversion_sm_vv.flatten()))
    MAE = mean_absolute_error(measured_sm.flatten(), inversion_sm_vv.flatten())
    MSE = mean_squared_error(measured_sm.flatten(), inversion_sm_vv.flatten())
    evaluation_indicator = ' (d)  R:{:.3f}  R2:{:.3f}  RMSE:{:.3f}  MAE:{:.3f}  MSE:{:.3f}'.format(r, R2, RMSE, MAE, MSE)
    axs[1].text(0.005, 0.975, evaluation_indicator, transform=axs[1].transAxes, fontdict={'size': '15'})
    fig.tight_layout()
    # plt.show()
    plt.savefig(picture_dir + '02 Zribi改进的CD反演法' + '.png')


def improved_cd_model(ndvi_path, sm_path, vv_path, vh_path, picture_dir):
    sm_df = pd.read_excel(sm_path, index_col=0, header=0)
    vv_2017_2021_df = pd.read_excel(vv_path, index_col=0, header=0)  # vv 2017-2021年的数据
    vh_2017_2021_df = pd.read_excel(vh_path, index_col=0, header=0)  # vh 2017-2021年的数据
    ndvi_2017_2021_df = pd.read_excel(ndvi_path, index_col=0, header=0)  # vv 2017-2021年的数据
    site_name = sm_df.index
    date_time_2019 = [name for name in sm_df.columns if '2019' in name]
    date_time_2020 = [name for name in sm_df.columns if '2020' in name]
    sm_2019 = np.array(sm_df.loc[:, date_time_2019].values)  # 获取2019年sm数据
    sm_2020 = np.array(sm_df.loc[:, date_time_2020].values)  # 获取2020年sm数据
    vv_2019 = np.array(vv_2017_2021_df.loc[:, date_time_2019].values)  # 获取2019年vv极化数据
    vv_2020 = np.array(vv_2017_2021_df.loc[:, date_time_2020].values)  # 获取2020年vv极化数据
    vh_2019 = np.array(vh_2017_2021_df.loc[:, date_time_2019].values)  # 获取2019年vh极化数据
    vh_2020 = np.array(vh_2017_2021_df.loc[:, date_time_2020].values)  # 获取2020年vh极化数据
    ndvi_2019 = np.array(ndvi_2017_2021_df.loc[:, date_time_2019].values)  # 获取2019年ndvi数据
    ndvi_2020 = np.array(ndvi_2017_2021_df.loc[:, date_time_2019].values)  # 获取2020年ndvi数据

    # 基于NDVI分段，0.25, 0.5, 1.0，获取各区间的最大和最小土壤湿度
    intervals = 3
    min_vv_2019 = np.zeros([sm_df.shape[0], intervals])
    max_vv_2019 = np.zeros([sm_df.shape[0], intervals])
    min_vv_2020 = np.zeros([sm_df.shape[0], intervals])
    max_vv_2020 = np.zeros([sm_df.shape[0], intervals])

    min_vh_2019 = np.zeros([sm_df.shape[0], intervals])
    max_vh_2019 = np.zeros([sm_df.shape[0], intervals])
    min_vh_2020 = np.zeros([sm_df.shape[0], intervals])
    max_vh_2020 = np.zeros([sm_df.shape[0], intervals])

    min_sm_2019 = np.zeros([sm_df.shape[0], intervals])
    max_sm_2019 = np.zeros([sm_df.shape[0], intervals])
    min_sm_2020 = np.zeros([sm_df.shape[0], intervals])
    max_sm_2020 = np.zeros([sm_df.shape[0], intervals])

    # 2019年 各站点 各NDVI分段区间 VV和VH极化最大值最小值
    for site in range(sm_df.shape[0]):
        vv_temp_01 = np.zeros((0,))
        vv_temp_02 = np.zeros((0,))
        vv_temp_03 = np.zeros((0,))
        vh_temp_01 = np.zeros((0,))
        vh_temp_02 = np.zeros((0,))
        vh_temp_03 = np.zeros((0,))
        sm_temp_01 = np.zeros((0,))
        sm_temp_02 = np.zeros((0,))
        sm_temp_03 = np.zeros((0,))
        for date in range(len(date_time_2019)):
            if ndvi_2019[site, date] <= 0.25:
                vv_temp_01 = np.append(vv_temp_01, vv_2019[site, date])  # 追加值
                vh_temp_01 = np.append(vh_temp_01, vh_2019[site, date])
                sm_temp_01 = np.append(sm_temp_01, sm_2019[site, date])
            elif ndvi_2019[site, date] <= 0.5:
                vv_temp_02 = np.append(vv_temp_02, vv_2019[site, date])  # 追加值
                vh_temp_02 = np.append(vh_temp_02, vh_2019[site, date])
                sm_temp_02 = np.append(sm_temp_02, sm_2019[site, date])
            elif ndvi_2019[site, date] <= 1.0:
                vv_temp_03 = np.append(vv_temp_03, vv_2019[site, date])
                vh_temp_03 = np.append(vh_temp_03, vh_2019[site, date])
                sm_temp_03 = np.append(sm_temp_03, sm_2019[site, date])

        if vv_temp_01.any():
            min_vv_2019[site, 0] = np.min(vv_temp_01)
            max_vv_2019[site, 0] = np.max(vv_temp_01)
        if vh_temp_01.any():
            min_vh_2019[site, 0] = np.min(vh_temp_01)
            max_vh_2019[site, 0] = np.max(vh_temp_01)
        if sm_temp_01.any():
            min_sm_2019[site, 0] = np.min(sm_temp_01)
            max_sm_2019[site, 0] = np.max(sm_temp_01)
        if vv_temp_02.any():
            min_vv_2019[site, 1] = np.min(vv_temp_02)
            max_vv_2019[site, 1] = np.max(vv_temp_02)
        if vh_temp_02.any():
            min_vh_2019[site, 1] = np.min(vh_temp_02)
            max_vh_2019[site, 1] = np.max(vh_temp_02)
        if sm_temp_02.any():
            min_sm_2019[site, 1] = np.min(sm_temp_02)
            max_sm_2019[site, 1] = np.max(sm_temp_02)
        if vv_temp_03.any():
            min_vv_2019[site, 2] = np.min(vv_temp_03)
            max_vv_2019[site, 2] = np.max(vv_temp_03)
        if vh_temp_03.any():
            min_vh_2019[site, 2] = np.min(vh_temp_03)
            max_vh_2019[site, 2] = np.max(vh_temp_03)
        if sm_temp_03.any():
            min_sm_2019[site, 2] = np.min(sm_temp_03)
            max_sm_2019[site, 2] = np.max(sm_temp_03)

    # 2020年 各站点 各NDVI分段区间 VV和VH极化最大值最小值
    for site in range(sm_df.shape[0]):
        vv_temp_01 = np.zeros((0,))
        vv_temp_02 = np.zeros((0,))
        vv_temp_03 = np.zeros((0,))
        vh_temp_01 = np.zeros((0,))
        vh_temp_02 = np.zeros((0,))
        vh_temp_03 = np.zeros((0,))
        sm_temp_01 = np.zeros((0,))
        sm_temp_02 = np.zeros((0,))
        sm_temp_03 = np.zeros((0,))
        for date in range(len(date_time_2020)):
            if ndvi_2020[site, date] <= 0.25:
                vv_temp_01 = np.append(vv_temp_01, vv_2020[site, date])  # 追加值
                vh_temp_01 = np.append(vh_temp_01, vh_2020[site, date])
                sm_temp_01 = np.append(sm_temp_01, sm_2020[site, date])
            elif ndvi_2020[site, date] <= 0.5:
                vv_temp_02 = np.append(vv_temp_02, vv_2020[site, date])  # 追加值
                vh_temp_02 = np.append(vh_temp_02, vh_2020[site, date])
                sm_temp_02 = np.append(sm_temp_02, sm_2020[site, date])
            elif ndvi_2020[site, date] <= 1.0:
                vv_temp_03 = np.append(vv_temp_03, vv_2020[site, date])
                vh_temp_03 = np.append(vh_temp_03, vh_2020[site, date])
                sm_temp_03 = np.append(sm_temp_03, sm_2020[site, date])

        if vv_temp_01.any():
            min_vv_2020[site, 0] = np.min(vv_temp_01)
            max_vv_2020[site, 0] = np.max(vv_temp_01)
        if vh_temp_01.any():
            min_vh_2020[site, 0] = np.min(vh_temp_01)
            max_vh_2020[site, 0] = np.max(vh_temp_01)
        if sm_temp_01.any():
            min_sm_2020[site, 0] = np.min(sm_temp_01)
            max_sm_2020[site, 0] = np.max(sm_temp_01)
        if vv_temp_02.any():
            min_vv_2020[site, 1] = np.min(vv_temp_02)
            max_vv_2020[site, 1] = np.max(vv_temp_02)
        if vh_temp_02.any():
            min_vh_2020[site, 1] = np.min(vh_temp_02)
            max_vh_2020[site, 1] = np.max(vh_temp_02)
        if sm_temp_02.any():
            min_sm_2020[site, 1] = np.min(sm_temp_02)
            max_sm_2020[site, 1] = np.max(sm_temp_02)
        if vv_temp_03.any():
            min_vv_2020[site, 2] = np.min(vv_temp_03)
            max_vv_2020[site, 2] = np.max(vv_temp_03)
        if vh_temp_03.any():
            min_vh_2020[site, 2] = np.min(vh_temp_03)
            max_vh_2020[site, 2] = np.max(vh_temp_03)
        if sm_temp_03.any():
            min_sm_2020[site, 2] = np.min(sm_temp_03)
            max_sm_2020[site, 2] = np.max(sm_temp_03)

    # 反演土壤湿度
    inversion_sm_2019_vv = np.zeros([sm_df.shape[0], len(date_time_2019)])
    inversion_sm_2019_vh = np.zeros([sm_df.shape[0], len(date_time_2019)])
    inversion_sm_2020_vv = np.zeros([sm_df.shape[0], len(date_time_2020)])
    inversion_sm_2020_vh = np.zeros([sm_df.shape[0], len(date_time_2020)])

    # 反演2019年土壤湿度
    for site in range(sm_df.shape[0]):
        for date in range(len(date_time_2019)):
            if ndvi_2019[site, date] <= 0.25:
                ndvi_index = 0
                now_vv = vv_2019[site, date]
                min_vv = min_vv_2019[site, ndvi_index]
                max_vv = max_vv_2019[site, ndvi_index]
                now_vh = vh_2019[site, date]
                min_vh = min_vh_2019[site, ndvi_index]
                max_vh = max_vh_2019[site, ndvi_index]
                min_sm = min_sm_2019[site, ndvi_index]
                max_sm = max_sm_2019[site, ndvi_index]
                inversion_sm_2019_vv[site, date] = (now_vv - min_vv) / (max_vv - min_vv) * (max_sm - min_sm) + min_sm
                inversion_sm_2019_vh[site, date] = (now_vh - min_vh) / (max_vh - min_vh) * (max_sm - min_sm) + min_sm
            elif ndvi_2019[site, date] <= 0.5:
                ndvi_index = 1
                now_vv = vv_2019[site, date]
                min_vv = min_vv_2019[site, ndvi_index]
                max_vv = max_vv_2019[site, ndvi_index]
                now_vh = vh_2019[site, date]
                min_vh = min_vh_2019[site, ndvi_index]
                max_vh = max_vh_2019[site, ndvi_index]
                min_sm = min_sm_2019[site, ndvi_index]
                max_sm = max_sm_2019[site, ndvi_index]
                inversion_sm_2019_vv[site, date] = (now_vv - min_vv) / (max_vv - min_vv) * (max_sm - min_sm) + min_sm
                inversion_sm_2019_vh[site, date] = (now_vh - min_vh) / (max_vh - min_vh) * (max_sm - min_sm) + min_sm
            elif ndvi_2019[site, date] <= 1.0:
                ndvi_index = 2
                now_vv = vv_2019[site, date]
                min_vv = min_vv_2019[site, ndvi_index]
                max_vv = max_vv_2019[site, ndvi_index]
                now_vh = vh_2019[site, date]
                min_vh = min_vh_2019[site, ndvi_index]
                max_vh = max_vh_2019[site, ndvi_index]
                min_sm = min_sm_2019[site, ndvi_index]
                max_sm = max_sm_2019[site, ndvi_index]
                inversion_sm_2019_vv[site, date] = (now_vv - min_vv) / (max_vv - min_vv) * (max_sm - min_sm) + min_sm
                inversion_sm_2019_vh[site, date] = (now_vh - min_vh) / (max_vh - min_vh) * (max_sm - min_sm) + min_sm

        # 反演2020年土壤湿度
        for site in range(sm_df.shape[0]):
            for date in range(len(date_time_2020)):
                if ndvi_2020[site, date] <= 0.25:
                    ndvi_index = 0
                    now_vv = vv_2020[site, date]
                    min_vv = min_vv_2020[site, ndvi_index]
                    max_vv = max_vv_2020[site, ndvi_index]
                    now_vh = vh_2020[site, date]
                    min_vh = min_vh_2020[site, ndvi_index]
                    max_vh = max_vh_2020[site, ndvi_index]
                    min_sm = min_sm_2020[site, ndvi_index]
                    max_sm = max_sm_2020[site, ndvi_index]
                    inversion_sm_2020_vv[site, date] = (now_vv - min_vv) / (max_vv - min_vv) * (
                        max_sm - min_sm) + min_sm
                    inversion_sm_2020_vh[site, date] = (now_vh - min_vh) / (max_vh - min_vh) * (
                        max_sm - min_sm) + min_sm
                elif ndvi_2020[site, date] <= 0.5:
                    ndvi_index = 1
                    now_vv = vv_2020[site, date]
                    min_vv = min_vv_2020[site, ndvi_index]
                    max_vv = max_vv_2020[site, ndvi_index]
                    now_vh = vh_2020[site, date]
                    min_vh = min_vh_2020[site, ndvi_index]
                    max_vh = max_vh_2020[site, ndvi_index]
                    min_sm = min_sm_2020[site, ndvi_index]
                    max_sm = max_sm_2020[site, ndvi_index]
                    inversion_sm_2020_vv[site, date] = (now_vv - min_vv) / (max_vv - min_vv) * (
                        max_sm - min_sm) + min_sm
                    inversion_sm_2020_vh[site, date] = (now_vh - min_vh) / (max_vh - min_vh) * (
                        max_sm - min_sm) + min_sm
                elif ndvi_2020[site, date] <= 1.0:
                    ndvi_index = 2
                    now_vv = vv_2020[site, date]
                    min_vv = min_vv_2020[site, ndvi_index]
                    max_vv = max_vv_2020[site, ndvi_index]
                    now_vh = vh_2020[site, date]
                    min_vh = min_vh_2020[site, ndvi_index]
                    max_vh = max_vh_2020[site, ndvi_index]
                    min_sm = min_sm_2020[site, ndvi_index]
                    max_sm = max_sm_2020[site, ndvi_index]
                    inversion_sm_2020_vv[site, date] = (now_vv - min_vv) / (max_vv - min_vv) * (
                        max_sm - min_sm) + min_sm
                    inversion_sm_2020_vh[site, date] = (now_vh - min_vh) / (max_vh - min_vh) * (
                        max_sm - min_sm) + min_sm

    inversion_sm_vv = np.hstack((inversion_sm_2019_vv, inversion_sm_2020_vv))  # 拼接数据
    inversion_sm_vh = np.hstack((inversion_sm_2019_vh, inversion_sm_2020_vh))  # 拼接数据
    measured_sm = np.hstack((sm_2019, sm_2020))

    return inversion_sm_vv, inversion_sm_vh, measured_sm

    plt.rcParams['figure.figsize'] = (19.2, 9.2)  # 绘图分辨率
    fig, axs = plt.subplots(1, 2)
    for ax in axs.ravel():
        ax.set_xticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
        ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
        ax.set_xlim(0.0, 0.52)
        ax.set_ylim(0.0, 0.52)
        ax.plot([0.0, 0.52], [0.0, 0.52], ls="--", c=".3", linewidth=1.0, alpha=0.8)
        ax.tick_params(labelsize=13)
    # 绘制子图1
    x = measured_sm.flatten()
    y = inversion_sm_vh.flatten()
    xy = np.vstack([x, y])  # 将两个维度的数据叠加
    z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[0].scatter(x, y, c=z, s=30, cmap='Spectral_r')
    axs[0].set_xlabel(r"Measured soil moisture ($cm^3/cm^3$)", fontsize=18)
    axs[0].set_ylabel(r"Soil moisture retrieval based on VH backscatter ($cm^3/cm^3$)", fontsize=18)
    r = ss.pearsonr(measured_sm.flatten(), inversion_sm_vh.flatten())[0]
    R2 = r2_score(measured_sm.flatten(), inversion_sm_vh.flatten())
    RMSE = sqrt(mean_squared_error(measured_sm.flatten(), inversion_sm_vh.flatten()))
    MAE = mean_absolute_error(measured_sm.flatten(), inversion_sm_vh.flatten())
    MSE = mean_squared_error(measured_sm.flatten(), inversion_sm_vh.flatten())
    evaluation_indicator = ' (e)  R:{:.3f}  R2:{:.3f}  RMSE:{:.3f}  MAE:{:.3f}  MSE:{:.3f}'.format(r, R2, RMSE, MAE,
                                                                                                   MSE)
    axs[0].text(0.005, 0.975, evaluation_indicator, transform=axs[0].transAxes, fontdict={'size': '15'})
    # 绘制子图2
    x = measured_sm.flatten()
    y = inversion_sm_vv.flatten()
    xy = np.vstack([x, y])  # 将两个维度的数据叠加
    z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[1].scatter(x, y, c=z, s=30, cmap='Spectral_r')
    axs[1].set_xlabel(r"Measured soil moisture ($cm^3/cm^3$)", fontsize=18)
    axs[1].set_ylabel(r"Soil moisture retrieval based on VV backscatter ($cm^3/cm^3$)", fontsize=18)
    r = ss.pearsonr(measured_sm.flatten(), inversion_sm_vv.flatten())[0]
    R2 = r2_score(measured_sm.flatten(), inversion_sm_vv.flatten())
    RMSE = sqrt(mean_squared_error(measured_sm.flatten(), inversion_sm_vv.flatten()))
    MAE = mean_absolute_error(measured_sm.flatten(), inversion_sm_vv.flatten())
    MSE = mean_squared_error(measured_sm.flatten(), inversion_sm_vv.flatten())
    evaluation_indicator = ' (f)  R:{:.3f}  R2:{:.3f}  RMSE:{:.3f}  MAE:{:.3f}  MSE:{:.3f}'.format(r, R2, RMSE, MAE,
                                                                                                   MSE)
    axs[1].text(0.005, 0.975, evaluation_indicator, transform=axs[1].transAxes, fontdict={'size': '15'})
    fig.tight_layout()
    # plt.show()
    plt.savefig(picture_dir + '03 本文改进的CD反演法' + '.png')


def alpha_model(sm_path, vv_path, picture_dir):
    pass


def plot_scatter(inversion_sm_vv_1, inversion_sm_vh_1, measured_sm_1,
                 inversion_sm_vv_2, inversion_sm_vh_2, measured_sm_2,
                 inversion_sm_vv_3, inversion_sm_vh_3, measured_sm_3,
                 pic_dir):
    plt.rcParams['figure.figsize'] = (18, 24)  # 绘图分辨率
    fig, axs = plt.subplots(3, 2)
    for ax in axs.ravel():
        ax.set_xticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
        ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
        ax.set_xlim(0.0, 0.52)
        ax.set_ylim(0.0, 0.52)
        ax.plot([0.0, 0.52], [0.0, 0.52], ls="--", c=".3", linewidth=1.0, alpha=0.8)
        ax.tick_params(labelsize=14)

    # 绘制子图1
    x = measured_sm_1.flatten()
    y = inversion_sm_vh_1.flatten()
    xy = np.vstack([x, y])  # 将两个维度的数据叠加
    z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[0][0].scatter(x, y, c=z, s=30, cmap='Spectral_r')
    axs[0][0].set_xlabel(r"Measured soil moisture ($cm^3/cm^3$)", fontsize=18)
    axs[0][0].set_ylabel(r"Soil moisture retrieval based on VH backscatter ($cm^3/cm^3$)", fontsize=18)
    r = ss.pearsonr(measured_sm_1.flatten(), inversion_sm_vh_1.flatten())[0]
    R2 = r2_score(measured_sm_1.flatten(), inversion_sm_vh_1.flatten())
    RMSE = sqrt(mean_squared_error(measured_sm_1.flatten(), inversion_sm_vh_1.flatten()))
    MAE = mean_absolute_error(measured_sm_1.flatten(), inversion_sm_vh_1.flatten())
    MSE = mean_squared_error(measured_sm_1.flatten(), inversion_sm_vh_1.flatten())
    evaluation_indicator = '(a)  R:{:.3f}  R2:{:.3f}  RMSE:{:.3f}  MAE:{:.3f}  MSE:{:.3f}'.format(r, R2, RMSE, MAE,
                                                                                                   MSE)
    axs[0][0].text(0.01, 0.96, evaluation_indicator, transform=axs[0][0].transAxes, fontdict={'size': '16'})
    # 绘制子图2
    x = measured_sm_1.flatten()
    y = inversion_sm_vv_1.flatten()
    xy = np.vstack([x, y])  # 将两个维度的数据叠加
    z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[0][1].scatter(x, y, c=z, s=30, cmap='Spectral_r')
    axs[0][1].set_xlabel(r"Measured soil moisture ($cm^3/cm^3$)", fontsize=18)
    axs[0][1].set_ylabel(r"Soil moisture retrieval based on VV backscatter ($cm^3/cm^3$)", fontsize=18)
    r = ss.pearsonr(measured_sm_1.flatten(), inversion_sm_vv_1.flatten())[0]
    R2 = r2_score(measured_sm_1.flatten(), inversion_sm_vv_1.flatten())
    RMSE = sqrt(mean_squared_error(measured_sm_1.flatten(), inversion_sm_vv_1.flatten()))
    MAE = mean_absolute_error(measured_sm_1.flatten(), inversion_sm_vv_1.flatten())
    MSE = mean_squared_error(measured_sm_1.flatten(), inversion_sm_vv_1.flatten())
    evaluation_indicator = '(b)  R:{:.3f}  R2:{:.3f}  RMSE:{:.3f}  MAE:{:.3f}  MSE:{:.3f}'.format(r, R2, RMSE, MAE,
                                                                MSE)
    axs[0][1].text(0.01, 0.96, evaluation_indicator, transform=axs[0][1].transAxes, fontdict={'size': '16'})

    # 绘制子图3
    x = measured_sm_2.flatten()
    y = inversion_sm_vh_2.flatten()
    xy = np.vstack([x, y])  # 将两个维度的数据叠加
    z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[1][0].scatter(x, y, c=z, s=30, cmap='Spectral_r')
    axs[1][0].set_xlabel(r"Measured soil moisture ($cm^3/cm^3$)", fontsize=18)
    axs[1][0].set_ylabel(r"Soil moisture retrieval based on VH backscatter ($cm^3/cm^3$)", fontsize=18)
    r = ss.pearsonr(measured_sm_2.flatten(), inversion_sm_vh_2.flatten())[0]
    R2 = r2_score(measured_sm_2.flatten(), inversion_sm_vh_2.flatten())
    RMSE = sqrt(mean_squared_error(measured_sm_2.flatten(), inversion_sm_vh_2.flatten()))
    MAE = mean_absolute_error(measured_sm_2.flatten(), inversion_sm_vh_2.flatten())
    MSE = mean_squared_error(measured_sm_2.flatten(), inversion_sm_vh_2.flatten())
    evaluation_indicator = '(c)  R:{:.3f}  R2:{:.3f}  RMSE:{:.3f}  MAE:{:.3f}  MSE:{:.3f}'.format(r, R2, RMSE, MAE,
                                                                                                   MSE)
    axs[1][0].text(0.01, 0.96, evaluation_indicator, transform=axs[1][0].transAxes, fontdict={'size': '16'})
    # 绘制子图4
    x = measured_sm_2.flatten()
    y = inversion_sm_vv_2.flatten()
    xy = np.vstack([x, y])  # 将两个维度的数据叠加
    z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[1][1].scatter(x, y, c=z, s=30, cmap='Spectral_r')
    axs[1][1].set_xlabel(r"Measured soil moisture ($cm^3/cm^3$)", fontsize=18)
    axs[1][1].set_ylabel(r"Soil moisture retrieval based on VV backscatter ($cm^3/cm^3$)", fontsize=18)
    r = ss.pearsonr(measured_sm_2.flatten(), inversion_sm_vv_2.flatten())[0]
    R2 = r2_score(measured_sm_2.flatten(), inversion_sm_vv_2.flatten())
    RMSE = sqrt(mean_squared_error(measured_sm_2.flatten(), inversion_sm_vv_2.flatten()))
    MAE = mean_absolute_error(measured_sm_2.flatten(), inversion_sm_vv_2.flatten())
    MSE = mean_squared_error(measured_sm_2.flatten(), inversion_sm_vv_2.flatten())
    evaluation_indicator = '(d)  R:{:.3f}  R2:{:.3f}  RMSE:{:.3f}  MAE:{:.3f}  MSE:{:.3f}'.format(r, R2, RMSE, MAE,
                                                                                                   MSE)
    axs[1][1].text(0.01, 0.96, evaluation_indicator, transform=axs[1][1].transAxes, fontdict={'size': '16'})

    # 绘制子图5
    x = measured_sm_3.flatten()
    y = inversion_sm_vh_3.flatten()
    xy = np.vstack([x, y])  # 将两个维度的数据叠加
    z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[2][0].scatter(x, y, c=z, s=30, cmap='Spectral_r')
    axs[2][0].set_xlabel(r"Measured soil moisture ($cm^3/cm^3$)", fontsize=18)
    axs[2][0].set_ylabel(r"Soil moisture retrieval based on VH backscatter ($cm^3/cm^3$)", fontsize=18)
    r = ss.pearsonr(measured_sm_3.flatten(), inversion_sm_vh_3.flatten())[0]
    R2 = r2_score(measured_sm_3.flatten(), inversion_sm_vh_3.flatten())
    RMSE = sqrt(mean_squared_error(measured_sm_3.flatten(), inversion_sm_vh_3.flatten()))
    MAE = mean_absolute_error(measured_sm_3.flatten(), inversion_sm_vh_3.flatten())
    MSE = mean_squared_error(measured_sm_3.flatten(), inversion_sm_vh_3.flatten())
    evaluation_indicator = '(e)  R:{:.3f}  R2:{:.3f}  RMSE:{:.3f}  MAE:{:.3f}  MSE:{:.3f}'.format(r, R2, RMSE, MAE,
                                                                                                   MSE)
    axs[2][0].text(0.01, 0.96, evaluation_indicator, transform=axs[2][0].transAxes, fontdict={'size': '16'})
    # 绘制子图6
    x = measured_sm_3.flatten()
    y = inversion_sm_vv_3.flatten()
    xy = np.vstack([x, y])  # 将两个维度的数据叠加
    z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    axs[2][1].scatter(x, y, c=z, s=30, cmap='Spectral_r')
    axs[2][1].set_xlabel(r"Measured soil moisture ($cm^3/cm^3$)", fontsize=18)
    axs[2][1].set_ylabel(r"Soil moisture retrieval based on VV backscatter ($cm^3/cm^3$)", fontsize=18)
    r = ss.pearsonr(measured_sm_3.flatten(), inversion_sm_vv_3.flatten())[0]
    R2 = r2_score(measured_sm_3.flatten(), inversion_sm_vv_3.flatten())
    RMSE = sqrt(mean_squared_error(measured_sm_3.flatten(), inversion_sm_vv_3.flatten()))
    MAE = mean_absolute_error(measured_sm_3.flatten(), inversion_sm_vv_3.flatten())
    MSE = mean_squared_error(measured_sm_3.flatten(), inversion_sm_vv_3.flatten())
    evaluation_indicator = '(f)  R:{:.3f}  R2:{:.3f}  RMSE:{:.3f}  MAE:{:.3f}  MSE:{:.3f}'.format(r, R2, RMSE, MAE,
                                                                                                   MSE)
    axs[2][1].text(0.01, 0.96, evaluation_indicator, transform=axs[2][1].transAxes, fontdict={'size': '16'})

    fig.tight_layout()
    # plt.show()
    plt.savefig(pic_dir + '各模型对比图' + '.png')


if __name__ == '__main__':
    sm_path = r'E:\04 Method\01 Table\sm_2019_2020_sort.xlsx'
    ts_path = r'E:\04 Method\01 Table\ts_2019_2020_sort.xlsx'
    vv_path = r'E:\04 Method\01 Table\vv_2017_2021_sort.xlsx'
    vh_path = r'E:\04 Method\01 Table\vh_2017_2021_sort.xlsx'
    ndvi_path = r'E:\04 Method\01 Table\ndvi_2017_2021_sort.xlsx'

    # Wagner提出的CD反演法
    pic_dir = r'E:\04 Method\02 Picture\03 变化检测法/'
    inversion_sm_vv_1, inversion_sm_vh_1, measured_sm_1 = wagner_cd_model(sm_path, vv_path, vh_path, pic_dir)  # 不校正，各站点使用全年的最大最小后向散射&土壤水分

    # Zribi改进的CD反演法
    inversion_sm_vv_2, inversion_sm_vh_2, measured_sm_2 = zribi_cd_model(ndvi_path, sm_path, vv_path, vh_path, pic_dir)  # 基于NDVI分段，只校正最大最小后向散射

    # 本文改进的CD反演法
    inversion_sm_vv_3, inversion_sm_vh_3, measured_sm_3 = improved_cd_model(ndvi_path, sm_path, vv_path, vh_path, pic_dir)  # 基于NDVI分段，使用区间后向散射&土壤水分的最大最小值

    plot_scatter(inversion_sm_vv_1, inversion_sm_vh_1, measured_sm_1,
                 inversion_sm_vv_2, inversion_sm_vh_2, measured_sm_2,
                 inversion_sm_vv_3, inversion_sm_vh_3, measured_sm_3,
                 pic_dir)

    # Alpha模型反演法
    # alpha_model(sm_path, vv_path, vh_path, pic_dir)

    os.startfile(pic_dir)  # 打开文件夹