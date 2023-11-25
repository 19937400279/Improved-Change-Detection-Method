# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import rcParams


def ndvi_sm_scatter(ndvi_path, sm_path, picture_dir):
    ndvi_2017_2021_df = pd.read_excel(ndvi_path, index_col=0, header=0)  # sm2019和2020年的数据
    site_name = ndvi_2017_2021_df.index
    sm_df = pd.read_excel(sm_path, index_col=0, header=0)  # vv2017-2021年的数据
    date_time = sm_df.columns
    ndvi_df = ndvi_2017_2021_df.loc[:, date_time]  # 获取2019和2020年的vv数据
    ndvi_list = ndvi_df.values
    sm_list = sm_df.values  # 获取sm数值

    sm_025 = np.zeros([0])
    sm_050 = np.zeros([0])
    sm_100 = np.zeros([0])

    # 分段获取最大值和最小值
    for row in range(sm_list.shape[0]):
        for column in range(sm_list.shape[1]):
            if site_name[row] == 'M1':
                if 0.25 >= ndvi_list[row][column] >= 0:
                    sm_num = sm_list[row][column]
                    sm_025 = np.append(sm_025, sm_num)
                elif 0.25 < ndvi_list[row][column] <= 0.5:
                    sm_num = sm_list[row][column]
                    sm_050 = np.append(sm_050, sm_num)
                elif 0.5 < ndvi_list[row][column] <= 1:
                    sm_num = sm_list[row][column]
                    sm_100 = np.append(sm_100, sm_num)

    config = {"font.family": 'serif', "mathtext.fontset": 'stix', "font.serif": ['SimSun']}
    rcParams.update(config)
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    plt.rc('font', family='Times New Roman')
    for row in range(ndvi_df.shape[0]):
        if site_name[row] == "M1":
            fig, ax = plt.subplots()
            lw = 2
            line_1 = plt.plot([0, 0.25], [np.max(sm_025), np.max(sm_025)], color='red', linewidth=lw, label='Maximum soil moisture')
            line_2 = plt.plot([0, 0.25], [np.min(sm_025), np.min(sm_025)], color='blue', linewidth=lw, label='Minimum soil moisture')
            plt.plot([0.25, 0.5], [np.max(sm_050), np.max(sm_050)], color='red', linewidth=lw)
            plt.plot([0.25, 0.5], [np.min(sm_050), np.min(sm_050)], color='blue', linewidth=lw)
            plt.plot([0.5, 1.0], [np.max(sm_100), np.max(sm_100)], color='red', linewidth=lw)
            plt.plot([0.5, 1.0], [np.min(sm_100), np.min(sm_100)], color='blue', linewidth=lw)
            lns = line_1 + line_2
            labels = [l.get_label() for l in lns]
            plt.legend(lns, labels, fontsize=20)
            ax.scatter(ndvi_list[row, :], sm_list[row, :], color='tab:green')
            # plt.title('2019 and 2020 year NDVI and soil moisture at ' + site_name[row], fontsize=16)
            ax.set_xlabel(r"NDVI", fontsize=26, labelpad=15)
            ax.set_ylabel(r"Measured soil moisture $\mathrm{({cm}^3/{cm}^3)}$", fontsize=26, labelpad=15)

            plt.text(0.125, 0.255, r'${Mv}_{x,y,}\scriptscriptstyle\mathrm{_I}\displaystyle\mathrm{_{,max}}$', fontdict={'family': 'Times New Roman', 'size': 22, 'color': 'black', 'usetex': True},
                     ha='center', va='center')
            plt.text(0.125, 0.09, r'${Mv}_{x,y,}\scriptscriptstyle\mathrm{_I}\displaystyle\mathrm{_{,min}}$', fontdict={'family': 'Times New Roman', 'size': 22, 'color': 'black', 'usetex': True},
                     ha='center', va='center')
            plt.text(0.375, 0.29, r'${Mv}_{x,y,}\scriptscriptstyle\mathrm{_{II}}\displaystyle\mathrm{_{,max}}$', fontdict={'family': 'Times New Roman', 'size': 22, 'color': 'black', 'usetex': True},
                     ha='center', va='center')
            plt.text(0.375, 0.13, r'${Mv}_{x,y,}\scriptscriptstyle\mathrm{_{II}}\displaystyle\mathrm{_{,min}}$', fontdict={'family': 'Times New Roman', 'size': 22, 'color': 'black', 'usetex': True},
                     ha='center', va='center')
            plt.text(0.75, 0.30, r'${Mv}_{x,y,}\scriptscriptstyle\mathrm{_{III}}\displaystyle\mathrm{_{,max}}$', fontdict={'family': 'Times New Roman', 'size': 22, 'color': 'black', 'usetex': True},
                     ha='center', va='center')
            plt.text(0.75, 0.135, r'${Mv}_{x,y,}\scriptscriptstyle\mathrm{_{III}}\displaystyle\mathrm{_{,min}}$', fontdict={'family': 'Times New Roman', 'size': 22, 'color': 'black', 'usetex': True},
                     ha='center', va='center')

            xticks = np.linspace(0, 1.0, 21)
            plt.xticks(xticks)
            plt.yticks([-0.02, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.52])
            plt.xlim(-0.0, 1.0)
            plt.ylim(-0.018, 0.518)
            plt.tick_params(labelsize=16)
            fig.tight_layout()
            # plt.grid(alpha=0.15, ls='--')
            plt.plot([0.25, 0.25], [-0.018, 0.518], linestyle=':', color='black', alpha=0.5)
            plt.plot([0.5, 0.5], [-0.018, 0.518], linestyle=':', color='black', alpha=0.5)
            # plt.show()
            plt.savefig(picture_dir + 'ndvi sm ' + site_name[row] + '.jpg', dpi=300)
            plt.close()
            print(picture_dir + 'ndvi sm ' + site_name[row] + '.jpg' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹


if __name__ == '__main__':
    sm_path = r'D:\04 Method\01 Table\sm_2019_2020_sort.xlsx'
    ts_path = r'D:\04 Method\01 Table\ts_2019_2020_sort.xlsx'
    vv_path = r'D:\04 Method\01 Table\vv_2017_2021_sort.xlsx'
    ndvi_path = r'D:\04 Method\01 Table\ndvi_2017_2021_sort.xlsx'
    picture_dir = r'D:\05 Essay\小论文\论文图\图9 散点图/'  # NDVI和SM
    ndvi_sm_scatter(ndvi_path, sm_path, picture_dir)
