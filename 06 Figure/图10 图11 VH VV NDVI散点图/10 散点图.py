# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import rcParams

def ndvi_vv_scatter(ndvi_path, vv_path, picture_dir):
    ndvi_2017_2021_df = pd.read_excel(ndvi_path, index_col=0, header=0)  # ndvi 2017和2021年的数据
    column_name = ndvi_2017_2021_df.columns
    date_time = [row_name for row_name in column_name if ('2019' in row_name or '2020' in row_name)]
    site_name = ndvi_2017_2021_df.index
    vv_2017_2021_df = pd.read_excel(vv_path, index_col=0, header=0)  # vv 2017-2021年的数据
    ndvi_list = ndvi_2017_2021_df[date_time].values  # 获取2019-2020年ndvi数值
    vv_list = vv_2017_2021_df[date_time].values  # 获取2019-2020年vv数值

    vv_025 = np.zeros([0])
    vv_050 = np.zeros([0])
    vv_100 = np.zeros([0])

    # 分段获取最大值和最小值
    for row in range(vv_list.shape[0]):
        for column in range(vv_list.shape[1]):
            if site_name[row] == 'M1':
                if 0.25 >= ndvi_list[row][column] >= 0:
                    vv_num = vv_list[row][column]
                    vv_025 = np.append(vv_025, vv_num)
                elif 0.25 < ndvi_list[row][column] <= 0.5:
                    vv_num = vv_list[row][column]
                    vv_050 = np.append(vv_050, vv_num)
                elif 0.5 < ndvi_list[row][column] <= 1:
                    vv_num = vv_list[row][column]
                    vv_100 = np.append(vv_100, vv_num)

    # plt.rcParams['text.usetex'] = True
    font_title = {  # 用 dict 单独指定 title 样式
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 32,
        'usetex': True,
    }

    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    config = {"font.family": 'serif', "mathtext.fontset": 'stix', "font.serif": ['SimSun']}
    rcParams.update(config)
    plt.rc('font', family='Times New Roman')
    for row in range(ndvi_2017_2021_df.shape[0]):
        if site_name[row] == 'M1':
            fig, ax = plt.subplots()
            lw = 2
            line_1 = plt.plot([0, 0.25], [np.max(vv_025), np.max(vv_025)],  color='red', linewidth=lw, label='Maximum VV polarization backscattering coefficient')
            line_2 = plt.plot([0, 0.25], [np.min(vv_025), np.min(vv_025)], color='blue', linewidth=lw, label='Minimum VV polarization backscattering coefficient')
            plt.plot([0.25, 0.5], [np.max(vv_050), np.max(vv_050)], color='red', linewidth=lw)
            plt.plot([0.25, 0.5], [np.min(vv_050), np.min(vv_050)], color='blue', linewidth=lw)
            plt.plot([0.5, 1.0], [np.max(vv_100), np.max(vv_100)], color='red', linewidth=lw)
            plt.plot([0.5, 1.0], [np.min(vv_100), np.min(vv_100)], color='blue', linewidth=lw)
            lns = line_1 + line_2
            labels = [l.get_label() for l in lns]
            plt.legend(lns, labels, fontsize=16)
            ax.scatter(ndvi_list[row, :], vv_list[row, :], color='tab:orange')
            ax.set_xlabel(r"NDVI", fontsize=26, labelpad=15)
            ax.set_ylabel(r"VV polarization backscattering coefficient ($\mathrm{linear}$)", fontsize=26, labelpad=15)

            plt.text(0.125, 0.052, r'$\sigma_{x,y,}\scriptscriptstyle\mathrm{_I}\displaystyle\mathrm{_{,max}}$', fontdict={'family': 'Times New Roman', 'size': 28, 'color': 'black', 'usetex': True}, ha='center', va='center')
            plt.text(0.125, 0.006, r'$\sigma_{x,y,}\scriptscriptstyle\mathrm{_I}\displaystyle\mathrm{_{,min}}$', fontdict={'family': 'Times New Roman', 'size': 28, 'color': 'black', 'usetex': True}, ha='center', va='center')
            plt.text(0.375, 0.046, r'$\sigma_{x,y,}\scriptscriptstyle\mathrm{_{II}}\displaystyle\mathrm{_{,max}}$', fontdict={'family': 'Times New Roman', 'size': 28, 'color': 'black', 'usetex': True}, ha='center', va='center')
            plt.text(0.375, 0.0078, r'$\sigma_{x,y,}\scriptscriptstyle\mathrm{_{II}}\displaystyle\mathrm{_{,min}}$', fontdict={'family': 'Times New Roman', 'size': 28, 'color': 'black', 'usetex': True}, ha='center', va='center')
            plt.text(0.75, 0.076, r'$\sigma_{x,y,}\scriptscriptstyle\mathrm{_{III}}\displaystyle\mathrm{_{,max}}$', fontdict={'family': 'Times New Roman', 'size': 28, 'color': 'black', 'usetex': True}, ha='center', va='center')
            plt.text(0.75, 0.014, r'$\sigma_{x,y,}\scriptscriptstyle\mathrm{_{III}}\displaystyle\mathrm{_{,min}}$', fontdict={'family': 'Times New Roman', 'size': 28, 'color': 'black', 'usetex': True}, ha='center', va='center')

            xticks = np.linspace(0, 1.0, 21)
            plt.xticks(xticks)
            plt.yticks([-0.005, 0, 0.025, 0.05, 0.075, 0.1, 0.105])
            plt.xlim(0.0, 1.0)
            plt.ylim(-0.0045, 0.1045)
            plt.tick_params(labelsize=16)
            # plt.grid(alpha=0.15, ls='--')
            plt.plot([0.25, 0.25], [-0.0045, 0.1045], linestyle=':', color='black', alpha=0.5)
            plt.plot([0.5, 0.5], [-0.0045, 0.1045], linestyle=':', color='black', alpha=0.5)
            fig.tight_layout()
            # plt.show()
            plt.savefig(picture_dir + 'ndvi vv ' + site_name[row] + '.jpg', dpi=300)
            plt.close()
            print(picture_dir + 'ndvi vv ' + site_name[row] + '.jpg' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹


def ndvi_vh_scatter(ndvi_path, vh_path, picture_dir):
    ndvi_2017_2021_df = pd.read_excel(ndvi_path, index_col=0, header=0)  # ndvi 2017和2021年的数据
    column_name = ndvi_2017_2021_df.columns
    date_time = [row_name for row_name in column_name if ('2019' in row_name or '2020' in row_name)]
    site_name = ndvi_2017_2021_df.index
    vh_2017_2021_df = pd.read_excel(vh_path, index_col=0, header=0)  # vv 2017-2021年的数据
    ndvi_list = ndvi_2017_2021_df[date_time].values  # 获取2019-2020年ndvi数值
    vh_list = vh_2017_2021_df[date_time].values  # 获取2019-2020年vv数值

    vh_025 = np.zeros([0])
    vh_050 = np.zeros([0])
    vh_100 = np.zeros([0])

    # 分段获取最大值和最小值
    for row in range(vh_list.shape[0]):
        for column in range(vh_list.shape[1]):
            if site_name[row] == 'M1':
                if 0.25 >= ndvi_list[row][column] >= 0:
                    vh_num = vh_list[row][column]
                    vh_025 = np.append(vh_025, vh_num)
                elif 0.25 < ndvi_list[row][column] <= 0.5:
                    vh_num = vh_list[row][column]
                    vh_050 = np.append(vh_050, vh_num)
                elif 0.5 < ndvi_list[row][column] <= 1:
                    vh_num = vh_list[row][column]
                    vh_100 = np.append(vh_100, vh_num)

    # plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    config = {"font.family": 'serif', "mathtext.fontset": 'stix', "font.serif": ['SimSun']}
    rcParams.update(config)
    plt.rc('font', family='Times New Roman')
    for row in range(ndvi_2017_2021_df.shape[0]):
        if site_name[row] == 'M1':
            fig, ax = plt.subplots()
            lw = 2
            line_1 = plt.plot([0, 0.25], [np.max(vh_025), np.max(vh_025)], color='red', linewidth=lw, label='Maximum VH polarization backscattering coefficient')
            line_2 = plt.plot([0, 0.25], [np.min(vh_025), np.min(vh_025)], color='blue', linewidth=lw, label='Minimum VH polarization backscattering coefficient')
            plt.plot([0.25, 0.5], [np.max(vh_050), np.max(vh_050)], color='red', linewidth=lw)
            plt.plot([0.25, 0.5], [np.min(vh_050), np.min(vh_050)], color='blue', linewidth=lw)
            plt.plot([0.5, 1.0], [np.max(vh_100), np.max(vh_100)], color='red', linewidth=lw)
            plt.plot([0.5, 1.0], [np.min(vh_100), np.min(vh_100)], color='blue', linewidth=lw)
            lns = line_1 + line_2
            labels = [l.get_label() for l in lns]
            plt.legend(lns, labels, fontsize=16)
            ax.scatter(ndvi_list[row, :], vh_list[row, :], color='tab:pink')
            ax.set_xlabel(r"NDVI", fontsize=26, labelpad=15)
            ax.set_ylabel(r"VH polarization backscattering coefficient ($\mathrm{linear}$)", fontsize=26, labelpad=15)

            plt.text(0.125, 0.003, r'$\sigma_{x,y,}\scriptscriptstyle\mathrm{_I}\displaystyle\mathrm{_{,max}}$', fontdict={'family': 'Times New Roman', 'size': 28, 'color': 'black', 'usetex': True}, ha='center', va='center')
            plt.text(0.125, 0.0001, r'$\sigma_{x,y,}\scriptscriptstyle\mathrm{_I}\displaystyle\mathrm{_{,min}}$', fontdict={'family': 'Times New Roman', 'size': 28, 'color': 'black', 'usetex': True}, ha='center', va='center')
            plt.text(0.375, 0.008, r'$\sigma_{x,y,}\scriptscriptstyle\mathrm{_{II}}\displaystyle\mathrm{_{,max}}$', fontdict={'family': 'Times New Roman', 'size': 28, 'color': 'black', 'usetex': True}, ha='center', va='center')
            plt.text(0.375, 0.00031, r'$\sigma_{x,y,}\scriptscriptstyle\mathrm{_{II}}\displaystyle\mathrm{_{,min}}$', fontdict={'family': 'Times New Roman', 'size': 28, 'color': 'black', 'usetex': True}, ha='center', va='center')
            plt.text(0.75, 0.0081, r'$\sigma_{x,y,}\scriptscriptstyle\mathrm{_{III}}\displaystyle\mathrm{_{,max}}$', fontdict={'family': 'Times New Roman', 'size': 28, 'color': 'black', 'usetex': True}, ha='center', va='center')
            plt.text(0.75, 0.0011, r'$\sigma_{x,y,}\scriptscriptstyle\mathrm{_{III}}\displaystyle\mathrm{_{,min}}$', fontdict={'family': 'Times New Roman', 'size': 28, 'color': 'black', 'usetex': True}, ha='center', va='center')

            xticks = np.linspace(0, 1.0, 21)
            plt.xticks(xticks)
            plt.yticks([-0.001, 0, 0.005, 0.01, 0.015, 0.016])
            plt.xlim(0.0, 1.0)
            plt.ylim(-0.0008, 0.0158)
            plt.tick_params(labelsize=16)
            # plt.grid(alpha=0.15, ls='--')
            plt.plot([0.25, 0.25], [-0.0008, 0.0158], linestyle=':', color='black', alpha=0.5)
            plt.plot([0.5, 0.5], [-0.0008, 0.0158], linestyle=':', color='black', alpha=0.5)
            fig.tight_layout()
            # plt.show()
            plt.savefig(picture_dir + 'ndvi vh ' + site_name[row] + '.jpg', dpi=300)
            plt.close()
            print(picture_dir + 'ndvi vh ' + site_name[row] + '.jpg' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹

if __name__ == '__main__':
    sm_path = r'D:\04 Method\01 Table\sm_2019_2020_sort.xlsx'
    ts_path = r'D:\04 Method\01 Table\ts_2019_2020_sort.xlsx'
    vv_path = r'D:\04 Method\01 Table\vv_2017_2021_sort.xlsx'
    vh_path = r'D:\04 Method\01 Table\vh_2017_2021_sort.xlsx'
    ndvi_path = r'D:\04 Method\01 Table\ndvi_2017_2021_sort.xlsx'

    picture_dir = r'D:\05 Essay\小论文\论文图\图10 图11 VH VV NDVI散点图/'  # NDVI和vv极化散点图 2017-2021年
    ndvi_vv_scatter(ndvi_path, vv_path, picture_dir)
    ndvi_vh_scatter(ndvi_path, vh_path, picture_dir)
