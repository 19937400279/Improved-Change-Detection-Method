# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import rcParams


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
    for row in range(ndvi_df.shape[0]):
        if site_name[row] == "M1":
            fig, ax = plt.subplots()
            ax.scatter(ndvi_list[row, :], sm_list[row, :], color='tab:green')
            # plt.title('2019 and 2020 year NDVI and soil moisture at ' + site_name[row], fontsize=16)
            ax.set_xlabel(r"NDVI", fontsize=26, labelpad=15)
            ax.set_ylabel(r"Measured soil moisture $(\mathrm{cm}^3/\mathrm{cm}^3)$", fontsize=26, labelpad=15)
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
            print(picture_dir + 'ndvi sm ' + site_name[row] + '.png' + '  图片保存成功')
    os.startfile(picture_dir)  # 打开文件夹


if __name__ == '__main__':
    sm_path = r'D:\04 Method\01 Table\sm_2019_2020_sort.xlsx'
    ts_path = r'D:\04 Method\01 Table\ts_2019_2020_sort.xlsx'
    vv_path = r'D:\04 Method\01 Table\vv_2017_2021_sort.xlsx'
    ndvi_path = r'D:\04 Method\01 Table\ndvi_2017_2021_sort.xlsx'

    picture_dir = r'D:\05 Essay\小论文\论文图\图8 散点图/'  # NDVI和SM
    ndvi_sm_scatter(ndvi_path, sm_path, picture_dir)




