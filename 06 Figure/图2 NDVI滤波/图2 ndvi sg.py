# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import seaborn as sns


def ndvi_line_chart(ndvi_2017_2021, ndvi_sg_2017_2021, picture_path):
    ndvi_2017_2018_df = pd.read_excel(ndvi_2017_2021, index_col=0, header=0)
    ndvi_sg_2017_2018_df = pd.read_excel(ndvi_sg_2017_2021, index_col=0, header=0)

    date_time = [date for date in ndvi_2017_2018_df.columns if ('2019' in date or '2020' in date)]
    site_name = ndvi_2017_2018_df.index

    new_date_time = []
    for date in date_time:
        new_date_time.append(date[0:4]+'/'+date[4:6]+'/'+date[6:8])

    ndvi_df = ndvi_2017_2018_df.loc[:, date_time]
    ndvi_sg_df = ndvi_sg_2017_2018_df.loc[:, date_time]
    plt.rcParams['figure.figsize'] = (15.5, 9)  # 1920x1080
    plt.rc('font', family='Times New Roman')

    date_time = ['2019001','2019009','2019017','2019025','2019033','2019041','2019049','2019057','2019065','2019073','2019081','2019089','2019097','2019105','2019113','2019121','2019129','2019137','2019145','2019153','2019161','2019169','2019177','2019185','2019193','2019201','2019209','2019217','2019225','2019233','2019241','2019249','2019257','2019265','2019273','2019281','2019289','2019297','2019305','2019313','2019321','2019329','2019337','2019345','2019353','2019361','2020001','2020009','2020017','2020025','2020033','2020041','2020049','2020057','2020065','2020073','2020081','2020089','2020097','2020105','2020113','2020121','2020129','2020137','2020145','2020153','2020161','2020169','2020177','2020185','2020193','2020201','2020209','2020217','2020225','2020233','2020241','2020249','2020257','2020265','2020273','2020281','2020289','2020297','2020305','2020313','2020321','2020329','2020337','2020345','2020353', '2020361']


    for row in range(ndvi_df.shape[0]):
        if 'M1' == site_name[row]:
            fig, ax1 = plt.subplots()
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(8))
            line_1 = ax1.plot(date_time, ndvi_df.iloc[row], color='tab:red', label='Original NDVI')
            ax1.set_xlabel("Day Of Year", fontsize=22, labelpad=10)
            ax1.set_ylabel("Original NDVI", fontsize=22, color='tab:red', labelpad=12)
            ax1.tick_params(axis='y', labelcolor='tab:red')
            # plt.xticks(rotation=90, fontsize=14)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.yticks([-0.05, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.05])
            plt.ylim(-0.048, 1.048)
            plt.grid(alpha=0.3, axis='y')
            ax2 = ax1.twinx()
            line_2 = ax2.plot(date_time, ndvi_sg_df.iloc[row], color='tab:blue', label='Filtered NDVI')
            ax2.set_ylabel("Filtered NDVI", fontsize=22, color='tab:blue', labelpad=18)
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            plt.yticks([-0.05, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.05])
            plt.ylim(-0.048, 1.048)
            plt.yticks(fontsize=18)
            lines = line_1 + line_2
            labels = [label.get_label() for label in lines]
            plt.legend(lines, labels, fontsize=20)
            fig.tight_layout()
            # plt.grid(alpha=0.3)
            # plt.show()
            plt.savefig(picture_path + 'Original NDVI and Filtered NDVI at All Stations.jpg', dpi=300)
            plt.close()
            print('Original NDVI and Filtered NDVI at Station ' + site_name[row] + '.jpg' + '  图片保存成功')



if __name__ == '__main__':
    ndvi_2017_2021 = r'D:\03 Modis\03 Table\NDVI 2017-2021 All.xlsx'
    ndvi_sg_2017_2021 = r'D:\03 Modis\03 Table\NDVI SG 2017-2021 All.xlsx'
    picture_path = r'D:\05 Essay\小论文\论文图\图2 NDVI滤波/'

    ndvi_line_chart(ndvi_2017_2021, ndvi_sg_2017_2021, picture_path)


