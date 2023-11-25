import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams

def box_2019():
    # 2019年
    SM_DataFrame = pd.read_excel(io=r'D:\Research\07 Excel\SM 5cm 2019-2020.xlsx')
    VV_DataFrame = pd.read_excel(io=r'D:\Research\07 Excel\vv 2019-2020.xlsx')
    SM_DataFrame = SM_DataFrame.T
    new_header = SM_DataFrame.iloc[0]  # 将第一行作为新的列名
    SM_DataFrame = SM_DataFrame[1:]  # 删除第一行
    SM_DataFrame.columns = new_header  # 将列名设置为新的列名
    SM_DataFrame = SM_DataFrame[:29]
    VV_DataFrame = VV_DataFrame.T
    new_header = VV_DataFrame.iloc[0]  # 将第一行作为新的列名
    VV_DataFrame = VV_DataFrame[1:]  # 删除第一行
    VV_DataFrame.columns = new_header  # 将列名设置为新的列名
    VV_DataFrame = VV_DataFrame[:29]
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 1920x1080
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax.grid(alpha=0.7, linestyle=':')
    bplot1 = ax.boxplot(x=SM_DataFrame.values, labels=SM_DataFrame.columns, patch_artist=True)
    ax.set_title('Box plot of soil moisture at each station in 2019', fontsize=14, labelpad=10)
    ax.set_xlabel('Site Name', fontsize=12, )
    ax.set_ylabel(r'Soil Moisture ($\,cm^{3}\,/cm^{3}$)', fontsize=12, labelpad=10)
    ax[1].grid(alpha=0.7, linestyle=':')
    bplot2 = ax[1].boxplot(x=VV_DataFrame.values, labels=VV_DataFrame.columns, patch_artist=True)
    ax[1].set_title('Box plot of VV backscatter coefficient at each station in 2019', fontsize=14)
    ax[1].set_xlabel('Site Name', fontsize=12)
    ax[1].set_ylabel(r'Sigma VV ($\,linear$)', fontsize=12)
    colors = ['pink'] * 26
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    colors = ['lightblue'] * 26
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    plt.tight_layout()
    plt.savefig('Box plot of soil moisture and VV backscatter coefficient at each station in 2019' + '.jpg', dpi=300)
    plt.show()
    plt.close()


def box_2020():
    # 2020年
    SM_DataFrame = pd.read_excel(io=r'D:\Research\07 Excel\SM 5cm 2019-2020.xlsx')
    VV_DataFrame = pd.read_excel(io=r'D:\Research\07 Excel\vv 2019-2020.xlsx')
    SM_DataFrame = SM_DataFrame.T
    new_header = SM_DataFrame.iloc[0]  # 将第一行作为新的列名
    SM_DataFrame = SM_DataFrame[1:]  # 删除第一行
    SM_DataFrame.columns = new_header  # 将列名设置为新的列名
    SM_DataFrame = SM_DataFrame[29:]
    VV_DataFrame = VV_DataFrame.T
    new_header = VV_DataFrame.iloc[0]  # 将第一行作为新的列名
    VV_DataFrame = VV_DataFrame[1:]  # 删除第一行
    VV_DataFrame.columns = new_header  # 将列名设置为新的列名
    VV_DataFrame = VV_DataFrame[29:]
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 1920x1080
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax.grid(alpha=0.7, linestyle=':')
    bplot1 = ax.boxplot(x=SM_DataFrame.values, labels=SM_DataFrame.columns, patch_artist=True)
    ax.set_title('Box plot of soil moisture at each station in 2020', fontsize=14)
    ax.set_xlabel('Site Name', fontsize=12, labelpad=10)
    ax.set_ylabel(r'Soil Moisture ($\,cm^{3}\,/cm^{3}$)', fontsize=12, labelpad=10)
    ax[1].grid(alpha=0.7, linestyle=':')
    bplot2 = ax[1].boxplot(x=VV_DataFrame.values, labels=VV_DataFrame.columns, patch_artist=True)
    ax[1].set_title('Box plot of VV backscatter coefficient at each station in 2020', fontsize=14)
    ax[1].set_xlabel('Site Name', fontsize=12, labelpad=10)
    ax[1].set_ylabel(r'Sigma VV ($\,linear$)', fontsize=12, labelpad=10)
    colors = ['pink'] * 26
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    colors = ['lightblue'] * 26
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    plt.tight_layout()
    plt.savefig('Box plot of soil moisture and VV backscatter coefficient at each station in 2020' + '.jpg', dpi=300)
    # plt.show()
    plt.close()

def box_2019_2020(sm_path, pic_path):
    # 2019-2020年
    config = {"font.family": 'serif', "mathtext.fontset": 'stix', "font.serif": ['SimSun']}
    rcParams.update(config)
    sm_df = pd.read_excel(sm_path, header=0, index_col=0).T
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 1920x1080
    plt.rc('font', family='Times New Roman')
    fig, ax = plt.subplots()
    # ax.grid(alpha=0.7, linestyle=':')
    bplot1 = ax.boxplot(x=sm_df.values, labels=sm_df.columns, patch_artist=True, medianprops=dict(color='tab:red', linewidth=1))
    ax.set_xlabel('Observation stations', fontsize=26, labelpad=15)
    ax.set_ylabel(r'Measured soil moisture $(\mathrm{cm}^3/\mathrm{cm}^3)$', fontsize=26, labelpad=15)
    plt.yticks([-0.02, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.52])
    plt.ylim(-0.018, 0.518)
    colors = ['lightblue'] * 26
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    # plt.show()
    plt.savefig(pic_path + 'Box plot of soil moisture at all stations from 2019 to 2020 year.jpg', dpi=300)
    plt.close()


if __name__ == '__main__':
    sm_path = r'D:\04 Method\01 Table\sm_2019_2020_sort.xlsx'
    pic_path = r'D:\05 Essay\小论文\论文图\图7 箱线图/'
    box_2019_2020(sm_path, pic_path)


