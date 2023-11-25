# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)

def sm_line_chart(sm_2019_2020_excel_path, save_path):
    del_files(save_path)
    print(save_path + '  旧图片已删除')
    color_list = sns.color_palette("hls", 26)
    sm_df = pd.read_excel(sm_2019_2020_excel_path, index_col=0, header=0)
    date_time = sm_df.columns
    site_name = sm_df.index
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 1920x1080
    fig, ax = plt.subplots()
    for row in range(sm_df.shape[0]):
        ax.plot(date_time, sm_df.iloc[row], color=color_list[row], label=site_name[row])
    ax.set_xlabel(r"Date", fontsize=12)
    ax.set_ylabel(r"Soil Moisture ($cm^3/cm^3$)", fontsize=12)
    plt.xticks(rotation=45, fontsize=10, horizontalalignment="right", rotation_mode='anchor')
    plt.title('2019 and 2020 year soil moisture at all sites', fontsize=16)
    fig.tight_layout()
    plt.legend()
    # plt.show()
    plt.savefig(save_path + 'sm all' + '.png')
    plt.close()
    print(save_path + 'sm all' + '.png' + '  图片保存成功')
    for row in range(sm_df.shape[0]):
        fig, ax = plt.subplots()
        ax.plot(date_time, sm_df.iloc[row], color='black', label=site_name[row])
        ax.set_xlabel(r"Date", fontsize=12)
        ax.set_ylabel(r"Soil Moisture ($cm^3/cm^3$)", fontsize=12)
        plt.xticks(rotation=45, fontsize=10, horizontalalignment="right", rotation_mode='anchor')
        plt.title('2019 and 2020 year soil moisture at site ' + site_name[row], fontsize=16)
        fig.tight_layout()
        # plt.show()
        plt.savefig(save_path + 'sm ' + site_name[row] + '.png')
        plt.close()
        print(save_path + 'sm ' + site_name[row] + '.png' + '  图片保存成功')


def ts_line_chart(ts_2019_2020_excel_path, save_path):
    del_files(save_path)
    print(save_path + '  旧图片已删除')
    color_list = sns.color_palette("hls", 26)
    ts_df = pd.read_excel(ts_2019_2020_excel_path, index_col=0, header=0)
    date_time = ts_df.columns
    site_name = ts_df.index
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 1920x1080
    fig, ax = plt.subplots()
    for row in range(ts_df.shape[0]):
        ax.plot(date_time, ts_df.iloc[row], color=color_list[row], label='ts'+site_name[row])
    ax.set_xlabel(r"Date", fontsize=12)
    ax.set_ylabel(r"Soil Temperature ($℃$)", fontsize=12)
    plt.xticks(rotation=45, fontsize=10, horizontalalignment="right", rotation_mode='anchor')
    plt.title('2019 and 2020 year soil temperature at all sites', fontsize=16)
    fig.tight_layout()
    plt.legend()
    # plt.show()
    plt.savefig(save_path + 'ts all' + '.png')
    plt.close()
    print(save_path + 'ts all' + '.png' + '  图片保存成功')
    for row in range(ts_df.shape[0]):
        fig, ax = plt.subplots()
        ax.plot(date_time, ts_df.iloc[row], color='black', label='ts'+site_name[row])
        ax.set_xlabel(r"Date", fontsize=12)
        ax.set_ylabel(r"Soil Temperature ($℃$)", fontsize=12)
        plt.xticks(rotation=45, fontsize=10, horizontalalignment="right", rotation_mode='anchor')
        plt.title('2019 and 2020 year soil temperature at site ' + site_name[row], fontsize=16)
        fig.tight_layout()
        # plt.show()
        plt.savefig(save_path + 'ts ' + site_name[row] + '.png')
        plt.close()
        print(save_path + 'ts ' + site_name[row] + '.png' + '  图片保存成功')


def vv_line_chart(vv_2019_2020_excel_path, save_path):
    del_files(save_path)
    print(save_path + '  旧图片已删除')
    color_list = sns.color_palette("hls", 26)
    vv_df = pd.read_excel(vv_2019_2020_excel_path, index_col=0, header=0)
    date_time = vv_df.columns
    site_name = vv_df.index
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 1920x1080
    fig, ax = plt.subplots()
    for row in range(vv_df.shape[0]):
        ax.plot(date_time, vv_df.iloc[row], color=color_list[row], label='vv'+site_name[row])
    ax.set_xlabel(r"Date", fontsize=12)
    ax.set_ylabel(r"vv polarization backscattering coefficient", fontsize=12)
    plt.xticks(rotation=90, fontsize=10)
    plt.title('2017 and 2021 year vv polarization backscattering coefficient at all sites', fontsize=16)
    plt.legend()
    fig.tight_layout()
    # plt.show()
    plt.savefig(save_path + 'vv all' + '.png')
    plt.close()
    print(save_path + 'vv all' + '.png' + '  图片保存成功')
    for row in range(vv_df.shape[0]):
        fig, ax = plt.subplots()
        ax.plot(date_time, vv_df.iloc[row], color='black', label='vv'+site_name[row])
        ax.set_xlabel(r"Date", fontsize=12)
        ax.set_ylabel(r"vv polarization backscattering coefficient", fontsize=12)
        plt.xticks(rotation=90, fontsize=10)
        plt.title('2017 and 2021 year vv polarization backscattering coefficient at site ' + site_name[row], fontsize=16)
        fig.tight_layout()
        # plt.show()
        plt.savefig(save_path + 'vv ' + site_name[row] + '.png')
        plt.close()
        print(save_path + 'vv ' + site_name[row] + '.png' + '  图片保存成功')


def vh_line_chart(vh_2019_2020_excel_path, save_path):
    del_files(save_path)
    print(save_path + '  旧图片已删除')
    color_list = sns.color_palette("hls", 26)
    vh_df = pd.read_excel(vh_2019_2020_excel_path, index_col=0, header=0)
    date_time = vh_df.columns
    site_name = vh_df.index
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 1920x1080
    fig, ax = plt.subplots()
    for row in range(vh_df.shape[0]):
        ax.plot(date_time, vh_df.iloc[row], color=color_list[row], label='vh '+site_name[row])
    ax.set_xlabel(r"Date", fontsize=12)
    ax.set_ylabel(r"vh polarization backscattering coefficient", fontsize=12)
    plt.xticks(rotation=90, fontsize=10)
    plt.title('2017 and 2021 year vh polarization backscattering coefficient at all sites', fontsize=16)
    plt.legend()
    fig.tight_layout()
    # plt.show()
    plt.savefig(save_path + 'vh all' + '.png')
    plt.close()
    print(save_path + 'vh all' + '.png' + '  图片保存成功')
    for row in range(vh_df.shape[0]):
        fig, ax = plt.subplots()
        ax.plot(date_time, vh_df.iloc[row], color='black', label='vh '+site_name[row])
        ax.set_xlabel(r"Date", fontsize=12)
        ax.set_ylabel(r"vh polarization backscattering coefficient", fontsize=12)
        plt.xticks(rotation=90, fontsize=10)
        plt.title('2017 and 2021 year vh polarization backscattering coefficient at site ' + site_name[row], fontsize=16)
        fig.tight_layout()
        # plt.show()
        plt.savefig(save_path + 'vh ' + site_name[row] + '.png')
        plt.close()
        print(save_path + 'vh ' + site_name[row] + '.png' + '  图片保存成功')


def ndvi_line_chart(ndvi_2019_2020_excel_path, save_path):
    del_files(save_path)
    print(save_path + '  旧图片已删除')
    color_list = sns.color_palette("hls", 26)
    # color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ndvi_df = pd.read_excel(ndvi_2019_2020_excel_path, index_col=0, header=0)
    date_time = ndvi_df.columns
    site_name = ndvi_df.index
    plt.rcParams['figure.figsize'] = (19.2, 10.8)  # 1920x1080
    fig, ax = plt.subplots()
    for row in range(ndvi_df.shape[0]):
        ax.plot(date_time, ndvi_df.iloc[row], color=color_list[row], label='NDVI '+site_name[row])
    ax.set_xlabel(r"Date", fontsize=12)
    ax.set_ylabel(r"NDVI", fontsize=12)
    plt.xticks(rotation=90, fontsize=10)
    plt.title('2017 and 2021 year NDVI at all sites', fontsize=16)
    plt.legend()
    fig.tight_layout()
    # plt.show()
    plt.savefig(save_path + 'ndvi all' + '.png')
    plt.close()
    print(save_path + 'ndvi all' + '.png' + '  图片保存成功')
    for row in range(ndvi_df.shape[0]):
        fig, ax = plt.subplots()
        ax.plot(date_time, ndvi_df.iloc[row], color='black', label='NDVI '+site_name[row])
        ax.set_xlabel(r"Date", fontsize=12)
        ax.set_ylabel(r"NDVI", fontsize=12)
        plt.xticks(rotation=90, fontsize=10)
        plt.title('2017 and 2021 year NDVI at site ' + site_name[row], fontsize=16)
        plt.legend()
        fig.tight_layout()
        # plt.show()
        plt.savefig(save_path + 'ndvi ' + site_name[row] + '.png')
        plt.close()
        print(save_path + 'ndvi ' + site_name[row] + '.png' + '  图片保存成功')

if __name__ == '__main__':
    sm_2019_2020_excel_path = r'E:\04 Method\01 Table\sm_2019_2020_sort.xlsx'
    sm_save_path = r'E:\04 Method\02 Picture\01 折线图\sm/'
    sm_line_chart(sm_2019_2020_excel_path, sm_save_path)

    ts_2019_2020_excel_path = r'E:\04 Method\01 Table\ts_2019_2020_sort.xlsx'
    ts_save_path = r'E:\04 Method\02 Picture\01 折线图\ts/'
    ts_line_chart(ts_2019_2020_excel_path, ts_save_path)

    vv_2017_2021_excel_path = r'E:\04 Method\01 Table\vv_2017_2021_sort.xlsx'
    vv_save_path = r'E:\04 Method\02 Picture\01 折线图\vv/'
    vv_line_chart(vv_2017_2021_excel_path, vv_save_path)

    vh_2017_2021_excel_path = r'E:\04 Method\01 Table\vh_2017_2021_sort.xlsx'
    vh_save_path = r'E:\04 Method\02 Picture\01 折线图\vh/'
    vh_line_chart(vh_2017_2021_excel_path, vh_save_path)

    ndvi_2017_2021_excel_path = r'E:\04 Method\01 Table\ndvi_2017_2021_sort.xlsx'
    ndvi_save_path = r'E:\04 Method\02 Picture\01 折线图\ndvi/'
    ndvi_line_chart(ndvi_2017_2021_excel_path, ndvi_save_path)

