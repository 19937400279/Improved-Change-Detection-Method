# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import numpy as np




def ndvi_sm_scatter(ndvi_path, sm_path, picture_dir):
    config = {"font.family": 'serif', "mathtext.fontset": 'stix', "font.serif": ['SimSun']}
    rcParams.update(config)
    ndvi_2017_2021_df = pd.read_excel(ndvi_path, index_col=0, header=0)  # sm2019和2020年的数据
    site_name = ndvi_2017_2021_df.index
    sm_df = pd.read_excel(sm_path, index_col=0, header=0)  # vv2017-2021年的数据
    date_time = sm_df.columns
    ndvi_df = ndvi_2017_2021_df.loc[:, date_time]  # 获取2019和2020年的vv数据
    ndvi_list = ndvi_df.values
    sm_list = sm_df.values  # 获取sm数值
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 绘图分辨率
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots()
    ax.scatter(ndvi_list, sm_list, color='tab:green')
    ax.set_xlabel(r"NDVI", fontsize=26, labelpad=10)
    ax.set_ylabel(r"Measured soil moisture $(\mathrm{cm}^3/\mathrm{cm}^3)$", fontdict={'family': 'Times New Roman'}, fontsize=26, labelpad=10)

    plt.xticks([-0.02, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.02])
    plt.yticks([-0.02, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.52])
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 0.51)
    plt.tick_params(labelsize=16)
    # plt.grid(alpha=0.25, ls='--')

    # red_line_x = [0.0114, 0.0954, 0.1931, 0.4241, 0.6376, 0.7812]
    # red_line_y = [0.2865, 0.3239, 0.3677, 0.4244, 0.4542, 0.4694]
    # blue_line_x = [0, ]
    # blue_line_y = [0.0446, ]

    line_1 = plt.plot([0.95], [0.355], color='red', label='Maximum soil moisture')
    line_2 = plt.plot([0.95], [0.355], color='blue', label='Minimum soil moisture')

    lns = line_1 + line_2
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, fontsize=20)

    fig.tight_layout()
    # plt.show()
    plt.savefig(picture_dir + 'ndvi sm all' + '.jpg', dpi=300)
    plt.close()
    print(picture_dir + 'ndvi sm all' + '.jpg' + '  图片保存成功')


if __name__ == '__main__':
    sm_path = r'D:\04 Method\01 Table\sm_2019_2020_sort.xlsx'
    ts_path = r'D:\04 Method\01 Table\ts_2019_2020_sort.xlsx'
    vv_path = r'D:\04 Method\01 Table\vv_2017_2021_sort.xlsx'
    ndvi_path = r'D:\04 Method\01 Table\ndvi_2017_2021_sort.xlsx'

    picture_dir = r'D:\05 Essay\小论文\论文图\图6 NDVI SM 散点图/'  # NDVI和SM
    ndvi_sm_scatter(ndvi_path, sm_path, picture_dir)
