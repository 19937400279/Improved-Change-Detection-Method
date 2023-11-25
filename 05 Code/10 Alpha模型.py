# 雷达入射角、
# VV极化转alpha
# 构建欠定方程组
# 获取边界条件求解欠定方程组
# 创建LUT超找表 获得土壤湿度
import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear, fsolve
import matplotlib.pyplot as plt
import os
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde
import scipy.stats as ss
import math


def dobson(freq, temp, sm, vsand, vclay, bd):
    '''
    Dobson模型 土壤湿度转介电常数
    :param freq: 雷达中心频率，单位 GHz
    :param temp: 土壤温度，单位摄氏度 ℃
    :param sm: 土壤体积含水量 cm3/cm3
    :param sand: 土壤沙土含量 %
    :param clay: 土壤黏土含量 %
    :param bd: 土壤容重 g/cm3
    :return: 土壤介电常数
    '''
    alpha = 0.65
    sd = 2.66
    dcs = (1.01 + 0.44 * sd) ** 2 - 0.062
    dc0 = 0.008854
    dcw0 = 88.045 - 0.4147 * temp + 6.295e-4 * (temp ** 2) + 1.075e-5 * (temp ** 3)
    tpt = 0.11109 - 3.824e-3 * temp + 6.938e-5 * temp ** 2 - 5.096e-7 * temp ** 3
    dcwinf = 4.9
    if freq >= 1.4:
        sigma = -1.645 + 1.939 * bd - 0.0225622 * vsand + 0.01594 * vclay
    else:
        sigma = 0.0467 + 0.2204 * bd - 0.004111 * vsand + 0.006614 * vclay
    dcwi = (tpt * freq * (dcw0 - dcwinf)) / (1 + (tpt * freq) ** 2) + sigma * (1.0 - (bd / sd)) / (8.0 * math.atan(1.0) * dc0 * freq * sm)
    dcwr = dcwinf + ((dcw0 - dcwinf) / (1 + (tpt * freq) ** 2))
    betai = 1.33797 - 0.00603 * vsand - 0.00166 * vclay
    betar = 1.2748 - 0.00519 * vsand - 0.00152 * vclay
    dcsi = (sm ** (betai / alpha)) * dcwi
    dcsr = (1.0 + (bd / sd) * ((dcs ** alpha) - 1.0) + (sm ** betar) * (dcwr ** alpha) - sm) ** (1 / alpha)
    return dcsr


def cal_sdc(st, sm, ts):
    '''
    基于dobson模型计算土壤介电常数
    :return: 2019年和2020年土壤介电常数
    '''

    column_name_2019 = [name for name in sm.columns if '2019' in name]
    column_name_2020 = [name for name in sm.columns if '2020' in name]

    # 土壤湿度转介电常数
    # 遍历每个站点的土壤湿度
    a = 1.0458
    b = 0.0022
    st = st.values
    sm_2019 = (sm.loc[:, column_name_2019] * a + b).values  # 校正2019年土壤水分
    sm_2020 = (sm.loc[:, column_name_2020] * a + b).values  # 校正2020年土壤水分
    ts_2019 = (ts.loc[:, column_name_2019]).values
    ts_2020 = (ts.loc[:, column_name_2020]).values

    sdc_2019 = np.zeros(sm_2019.shape)
    for site in range(sm_2019.shape[0]):
        for column in range(sm_2019.shape[1]):
            freq = 5.405
            temp = ts_2019[site, column]
            sm = sm_2019[site, column]
            vsand = st[site, 0]
            vclay = st[site, 2]
            bd = st[site, 3]
            sdc_2019[site, column] = dobson(freq, temp, sm, vsand, vclay, bd)

    sdc_2020 = np.zeros(sm_2020.shape)
    for site in range(sm_2020.shape[0]):
        for column in range(sm_2020.shape[1]):
            freq = 5.405
            temp = ts_2020[site, column]
            sm = sm_2020[site, column]
            vsand = st[site, 0]
            vclay = st[site, 2]
            bd = st[site, 3]
            sdc_2020[site, column] = dobson(freq, temp, sm, vsand, vclay, bd)
    return sdc_2019, sdc_2020


def cal_alpha(sdc_2019, sdc_2020, ang):
    column_name_2019 = [name for name in ang.columns if '2019' in name]
    column_name_2020 = [name for name in ang.columns if '2020' in name]

    ang_2019 = (ang.loc[:, column_name_2019] / 180 * np.pi).values
    ang_2020 = (ang.loc[:, column_name_2020] / 180 * np.pi).values

    alpha_vv_2019 = np.zeros(ang_2019.shape)
    for site in range(ang_2019.shape[0]):
        for column in range(ang_2019.shape[1]):
            dc = sdc_2019[site, column]
            angle = ang_2019[site, column]
            alpha_vv_2019[site, column] = (dc - 1) * (np.sin(angle) ** 2 - dc * (1 + np.sin(angle) ** 2)) / (dc * np.cos(angle) + np.sqrt(dc - np.sin(angle) ** 2)) ** 2

    alpha_vv_2020 = np.zeros(ang_2020.shape)
    for site in range(ang_2020.shape[0]):
        for column in range(ang_2020.shape[1]):
            dc = sdc_2020[site, column]
            angle = ang_2020[site, column]
            alpha_vv_2020[site, column] = (dc - 1) * (np.sin(angle) ** 2 - dc * (1 + np.sin(angle) ** 2)) / (dc * np.cos(angle) + np.sqrt(dc - np.sin(angle) ** 2)) ** 2

    return alpha_vv_2019, alpha_vv_2020


def creat_coefficient_matrix(sm, vv):
    column_name_2019 = [name for name in sm.columns if '2019' in name]
    column_name_2020 = [name for name in sm.columns if '2020' in name]
    a = 1.0458
    b = 0.0022
    sm_2019 = (sm.loc[:, column_name_2019] * a + b).values  # 校正2019年土壤水分
    sm_2020 = (sm.loc[:, column_name_2020] * a + b).values  # 校正2020年土壤水分
    vv_2019 = vv.loc[:, column_name_2019].values
    vv_2020 = vv.loc[:, column_name_2020].values

    # 获取边界条件 即最小和最大的alpha值 将每年分为3段，每段求解 4个月一段   1,2,3,4   5,6,7,8   9,10,11,12
    interval_1_2019 = [name for name in column_name_2019 if ('201901' in name or '201902' in name or '201903' in name or '201904' in name)]
    interval_2_2019 = [name for name in column_name_2019 if ('201905' in name or '201906' in name or '201907' in name or '201908' in name)]
    interval_3_2019 = [name for name in column_name_2019 if ('201909' in name or '201910' in name or '201911' in name or '201912' in name)]
    min_interval_1_2019 = np.zeros([alpha_vv_2019.shape[0]])
    min_interval_2_2019 = np.zeros([alpha_vv_2019.shape[0]])
    min_interval_3_2019 = np.zeros([alpha_vv_2019.shape[0]])
    max_interval_1_2019 = np.zeros([alpha_vv_2019.shape[0]])
    max_interval_2_2019 = np.zeros([alpha_vv_2019.shape[0]])
    max_interval_3_2019 = np.zeros([alpha_vv_2019.shape[0]])
    for site in range(alpha_vv_2019.shape[0]):
        min_interval_1_2019[site] = min(alpha_vv_2019[site, 0:len(interval_1_2019)])
        max_interval_1_2019[site] = max(alpha_vv_2019[site, 0:len(interval_1_2019)])
        min_interval_2_2019[site] = min(alpha_vv_2019[site, len(interval_1_2019):len(interval_1_2019) + len(interval_2_2019)])
        max_interval_2_2019[site] = max(alpha_vv_2019[site, len(interval_1_2019):len(interval_1_2019) + len(interval_2_2019)])
        min_interval_3_2019[site] = min(alpha_vv_2019[site, len(interval_1_2019) + len(interval_2_2019):len(interval_1_2019) + len(interval_2_2019) + len(interval_3_2019)])
        max_interval_3_2019[site] = max(alpha_vv_2019[site, len(interval_1_2019) + len(interval_2_2019):len(interval_1_2019) + len(interval_2_2019) + len(interval_3_2019)])
    interval_1_2020 = [name for name in column_name_2020 if ('202001' in name or '202002' in name or '202003' in name or '202004' in name)]
    interval_2_2020 = [name for name in column_name_2020 if ('202005' in name or '202006' in name or '202007' in name or '202008' in name)]
    interval_3_2020 = [name for name in column_name_2020 if ('202009' in name or '202010' in name or '202011' in name or '202012' in name)]
    min_interval_1_2020 = np.zeros([alpha_vv_2020.shape[0]])
    min_interval_2_2020 = np.zeros([alpha_vv_2020.shape[0]])
    min_interval_3_2020 = np.zeros([alpha_vv_2020.shape[0]])
    max_interval_1_2020 = np.zeros([alpha_vv_2020.shape[0]])
    max_interval_2_2020 = np.zeros([alpha_vv_2020.shape[0]])
    max_interval_3_2020 = np.zeros([alpha_vv_2020.shape[0]])
    for site in range(alpha_vv_2019.shape[0]):
        min_interval_1_2020[site] = min(alpha_vv_2020[site, 0:len(interval_1_2020)])
        max_interval_1_2020[site] = max(alpha_vv_2020[site, 0:len(interval_1_2020)])
        min_interval_2_2020[site] = min(alpha_vv_2020[site, len(interval_1_2020):len(interval_1_2020) + len(interval_2_2020)])
        max_interval_2_2020[site] = max(alpha_vv_2020[site, len(interval_1_2020):len(interval_1_2020) + len(interval_2_2020)])
        min_interval_3_2020[site] = min(alpha_vv_2020[site, len(interval_1_2020) + len(interval_2_2020):len(interval_1_2020) + len(interval_2_2020) + len(interval_3_2020)])
        max_interval_3_2020[site] = max(alpha_vv_2020[site, len(interval_1_2020) + len(interval_2_2020):len(interval_1_2020) + len(interval_2_2020) + len(interval_3_2020)])

    # 创建系数方程组
    # 每4个月创建一个欠定方程组 使用各自的边界约束条件进行求解 (n-1)x(n)
    coefficient_matrix_2019_interval_1 = np.zeros([sm_2019.shape[0], len(interval_1_2019) - 1, len(interval_1_2019)])
    coefficient_matrix_2019_interval_2 = np.zeros([sm_2019.shape[0], len(interval_2_2019) - 1, len(interval_2_2019)])
    coefficient_matrix_2019_interval_3 = np.zeros([sm_2019.shape[0], len(interval_3_2019) - 1, len(interval_3_2019)])
    for site in range(sm_2019.shape[0]):
        for column in range(0, len(interval_1_2019) - 1):
            coefficient_matrix_2019_interval_1[site, column, column] = -np.sqrt(vv_2019[site, column + 1] / vv_2019[site, column])
            coefficient_matrix_2019_interval_1[site, column, column + 1] = 1
        for column in range(len(interval_1_2019), len(interval_2_2019) + len(interval_1_2019) - 1):
            coefficient_matrix_2019_interval_2[site, column - len(interval_1_2019), column - len(interval_1_2019)] = -np.sqrt(vv_2019[site, column + 1] / vv_2019[site, column])
            coefficient_matrix_2019_interval_2[site, column - len(interval_1_2019), column - len(interval_1_2019) + 1] = 1
        for column in range(len(interval_2_2019) + len(interval_1_2019), len(interval_3_2019) + len(interval_2_2019) + len(interval_1_2019) - 1):
            coefficient_matrix_2019_interval_3[site, column - len(interval_1_2019) - len(interval_2_2019), column - len(interval_1_2019) - len(interval_2_2019)] = -np.sqrt(
                vv_2019[site, column + 1] / vv_2019[site, column])
            coefficient_matrix_2019_interval_3[site, column - len(interval_1_2019) - len(interval_2_2019), column - len(interval_1_2019) - len(interval_2_2019) + 1] = 1

    coefficient_matrix_2020_interval_1 = np.zeros([sm_2020.shape[0], len(interval_1_2020) - 1, len(interval_1_2020)])
    coefficient_matrix_2020_interval_2 = np.zeros([sm_2020.shape[0], len(interval_2_2020) - 1, len(interval_2_2020)])
    coefficient_matrix_2020_interval_3 = np.zeros([sm_2020.shape[0], len(interval_3_2020) - 1, len(interval_3_2020)])
    for site in range(sm_2020.shape[0]):
        for column in range(0, len(interval_1_2020) - 1):
            coefficient_matrix_2020_interval_1[site, column, column] = -np.sqrt(vv_2020[site, column + 1] / vv_2020[site, column])
            coefficient_matrix_2020_interval_1[site, column, column + 1] = 1
        for column in range(len(interval_1_2020), len(interval_2_2020) + len(interval_1_2020) - 1):
            coefficient_matrix_2020_interval_2[site, column - len(interval_1_2020), column - len(interval_1_2020)] = -np.sqrt(vv_2020[site, column + 1] / vv_2020[site, column])
            coefficient_matrix_2020_interval_2[site, column - len(interval_1_2020), column - len(interval_1_2020) + 1] = 1
        for column in range(len(interval_1_2020) + len(interval_2_2020), len(interval_3_2020) + len(interval_2_2020) + len(interval_1_2020) - 1):
            coefficient_matrix_2020_interval_3[site, column - len(interval_1_2020) - len(interval_2_2020), column - len(interval_1_2020) - len(interval_2_2020)] = -np.sqrt(
                vv_2020[site, column + 1] / vv_2020[site, column])
            coefficient_matrix_2020_interval_3[site, column - len(interval_1_2020) - len(interval_2_2020), column - len(interval_1_2020) - len(interval_2_2020) + 1] = 1

    # 求解欠定方程组
    # 2019
    solved_alpha_2019_interval_1 = np.zeros([sm_2019.shape[0], len(interval_1_2019)])
    solved_alpha_2019_interval_2 = np.zeros([sm_2019.shape[0], len(interval_2_2019)])
    solved_alpha_2019_interval_3 = np.zeros([sm_2019.shape[0], len(interval_3_2019)])
    for site in range(sm_2019.shape[0]):
        # 求解第一段
        A = coefficient_matrix_2019_interval_1[site][:][:]
        b = np.zeros(np.size(coefficient_matrix_2019_interval_1[site][:][:], 0))  # 创建(n-1)x1的0向量
        lb = np.zeros(np.size(coefficient_matrix_2019_interval_1[site][:][:], 1)) + min_interval_1_2019[site]
        ub = np.zeros(np.size(coefficient_matrix_2019_interval_1[site][:][:], 1)) + max_interval_1_2019[site]
        res = lsq_linear(A, b, bounds=(lb, ub))
        solved_alpha_2019_interval_1[site, :] = res.x
        # solved_alpha_2019_interval_1[site, :] = lsq_linear(A, b, bounds=(lb, ub))
        # 求解第二段
        A = coefficient_matrix_2019_interval_2[site][:][:]
        b = np.zeros(np.size(coefficient_matrix_2019_interval_2[site][:][:], 0))  # 创建(n-1)x1的0向量
        lb = np.zeros(np.size(coefficient_matrix_2019_interval_2[site][:][:], 1)) + min_interval_2_2019[site]
        ub = np.zeros(np.size(coefficient_matrix_2019_interval_2[site][:][:], 1)) + max_interval_2_2019[site]
        res = lsq_linear(A, b, bounds=(lb, ub))
        solved_alpha_2019_interval_2[site, :] = res.x
        # 求解第三段
        A = coefficient_matrix_2019_interval_3[site][:][:]
        b = np.zeros(np.size(coefficient_matrix_2019_interval_3[site][:][:], 0))  # 创建(n-1)x1的0向量
        lb = np.zeros(np.size(coefficient_matrix_2019_interval_3[site][:][:], 1)) + min_interval_3_2019[site]
        ub = np.zeros(np.size(coefficient_matrix_2019_interval_3[site][:][:], 1)) + max_interval_3_2019[site]
        res = lsq_linear(A, b, bounds=(lb, ub))
        solved_alpha_2019_interval_3[site, :] = res.x
    solved_alpha_2019 = np.hstack((solved_alpha_2019_interval_1, solved_alpha_2019_interval_2, solved_alpha_2019_interval_3))  # 合并所有的解

    # 2020
    solved_alpha_2020_interval_1 = np.zeros([sm_2020.shape[0], len(interval_1_2020)])
    solved_alpha_2020_interval_2 = np.zeros([sm_2020.shape[0], len(interval_2_2020)])
    solved_alpha_2020_interval_3 = np.zeros([sm_2020.shape[0], len(interval_3_2020)])
    for site in range(sm_2020.shape[0]):
        # 求解第一段
        A = coefficient_matrix_2020_interval_1[site][:][:]
        b = np.zeros(np.size(coefficient_matrix_2020_interval_1[site][:][:], 0))  # 创建(n-1)x1的0向量
        lb = np.zeros(np.size(coefficient_matrix_2020_interval_1[site][:][:], 1)) + min_interval_1_2020[site]
        ub = np.zeros(np.size(coefficient_matrix_2020_interval_1[site][:][:], 1)) + max_interval_1_2020[site]
        res = lsq_linear(A, b, bounds=(lb, ub))
        solved_alpha_2020_interval_1[site, :] = res.x
        # 求解第二段
        A = coefficient_matrix_2020_interval_2[site][:][:]
        b = np.zeros(np.size(coefficient_matrix_2020_interval_2[site][:][:], 0))  # 创建(n-1)x1的0向量
        lb = np.zeros(np.size(coefficient_matrix_2020_interval_2[site][:][:], 1)) + min_interval_2_2020[site]
        ub = np.zeros(np.size(coefficient_matrix_2020_interval_2[site][:][:], 1)) + max_interval_2_2020[site]
        res = lsq_linear(A, b, bounds=(lb, ub))
        solved_alpha_2020_interval_2[site, :] = res.x
        # 求解第三段
        A = coefficient_matrix_2020_interval_3[site][:][:]
        b = np.zeros(np.size(coefficient_matrix_2020_interval_3[site][:][:], 0))  # 创建(n-1)x1的0向量
        lb = np.zeros(np.size(coefficient_matrix_2020_interval_3[site][:][:], 1)) + min_interval_3_2020[site]
        ub = np.zeros(np.size(coefficient_matrix_2020_interval_3[site][:][:], 1)) + max_interval_3_2020[site]
        res = lsq_linear(A, b, bounds=(lb, ub))
        solved_alpha_2020_interval_3[site, :] = res.x
    solved_alpha_2020 = np.hstack((solved_alpha_2020_interval_1, solved_alpha_2020_interval_2, solved_alpha_2020_interval_3))
    return solved_alpha_2019, solved_alpha_2020


def lut(st, sm, ts):
    column_name_2019 = [name for name in sm.columns if '2019' in name]
    column_name_2020 = [name for name in sm.columns if '2020' in name]
    a = 1.0458
    b = 0.0022
    st = st.values
    sm_2019 = (sm.loc[:, column_name_2019] * a + b).values  # 校正2019年土壤水分
    sm_2020 = (sm.loc[:, column_name_2020] * a + b).values  # 校正2020年土壤水分
    ts_2019 = (ts.loc[:, column_name_2019]).values
    ts_2020 = (ts.loc[:, column_name_2020]).values

    min_sm = 0.01
    max_sm = 0.50
    step_sm = 0.001
    lut_2019 = np.zeros([sm_2019.shape[0], sm_2019.shape[1], len(np.arange(min_sm, max_sm, step_sm))])
    for site in range(sm_2019.shape[0]):
        for column in range(sm_2019.shape[1]):
            for sm_num in range(len(np.arange(min_sm, max_sm, step_sm))):
                freq = 5.405
                temp = ts_2019[site, column]
                sm = min_sm + sm_num * step_sm
                vsand = st[site, 0]
                vclay = st[site, 2]
                bd = st[site, 3]
                lut_2019[site, column, sm_num] = dobson(freq, temp, sm, vsand, vclay, bd)

    lut_2020 = np.zeros([sm_2020.shape[0], sm_2020.shape[1], len(np.arange(min_sm, max_sm, step_sm))])
    for site in range(sm_2020.shape[0]):
        for column in range(sm_2020.shape[1]):
            for sm_num in range(len(np.arange(min_sm, max_sm, step_sm))):
                freq = 5.405
                temp = ts_2020[site, column]
                sm = min_sm + sm_num * step_sm
                vsand = st[site, 0]
                vclay = st[site, 2]
                bd = st[site, 3]
                lut_2020[site, column, sm_num] = dobson(freq, temp, sm, vsand, vclay, bd)
    return lut_2019, lut_2020


def solved_sm(solved_alpha_2019, solved_alpha_2020, lut_2019, lut_2020):
    # 遍历查找表，获取土壤水分
    min_sm = 0.01
    step_sm = 0.001
    solved_sm_2019 = np.zeros(solved_alpha_2019.shape)
    for site in range(lut_2019.shape[0]):
        for column in range(lut_2019.shape[1]):
            min_err = float("inf")
            err = 0
            for sm_num in range(lut_2019.shape[2]):
                err = np.abs(lut_2019[site, column, sm_num] - solved_alpha_2019[site, column])
                if err < min_err:
                    min_err = err
                    solved_sm_2019[site, column] = min_sm + step_sm * sm_num
    solved_sm_2020 = np.zeros(solved_alpha_2020.shape)
    for site in range(lut_2020.shape[0]):
        for column in range(lut_2020.shape[1]):
            min_err = float("inf")
            err = 0
            for sm_num in range(lut_2020.shape[2]):
                err = np.abs(lut_2020[site, column, sm_num] - solved_alpha_2020[site, column])
                if err < min_err:
                    min_err = err
                    solved_sm_2020[site, column] = min_sm + step_sm * sm_num
    return solved_sm_2019, solved_sm_2020


def func(x, params):
    angle, alpha = params
    return ((x - 1) * (np.sin(angle) ** 2 - x * (1 + np.sin(angle) ** 2))) / ((x * np.cos(angle) + np.sqrt(x - np.sin(angle) ** 2)) ** 2) - alpha


def solved_dc(solved_alpha_2019, solved_alpha_2020, ang):
    column_name_2019 = [name for name in ang.columns if '2019' in name]
    column_name_2020 = [name for name in ang.columns if '2020' in name]
    ang_2019 = (ang.loc[:, column_name_2019] / 180 * np.pi).values
    ang_2020 = (ang.loc[:, column_name_2020] / 180 * np.pi).values
    solved_dc_2019 = np.zeros([ang_2019.shape[0], ang_2019.shape[1]])
    for site in range(ang_2019.shape[0]):
        for column in range(ang_2020.shape[1]):
            ang = ang_2019[site, column]
            alpha = solved_alpha_2019[site, column]
            params = [ang, alpha]
            root = fsolve(func, 5, args=params)
            solved_dc_2019[site, column] = root
    solved_dc_2020 = np.zeros([ang_2020.shape[0], ang_2020.shape[1]])
    for site in range(ang_2020.shape[0]):
        for column in range(ang_2020.shape[1]):
            ang = ang_2020[site, column]
            alpha = solved_alpha_2020[site, column]
            params = [ang, alpha]
            root = fsolve(func, 5, args=params)
            solved_dc_2020[site, column] = root
    return solved_dc_2019, solved_dc_2020


if __name__ == '__main__':
    st_path = r'E:\04 Method\01 Table\st_2019_2020_sort.xlsx'  # 土壤质地与容重
    sm_path = r'E:\04 Method\01 Table\sm_2019_2020_sort.xlsx'  # 土壤湿度
    ts_path = r'E:\04 Method\01 Table\ts_2019_2020_sort.xlsx'  # 土壤温度
    vv_path = r'E:\04 Method\01 Table\vv_2017_2021_sort.xlsx'  # vv极化后向散射系数
    vh_path = r'E:\04 Method\01 Table\vh_2017_2021_sort.xlsx'  # vh极化后向散射系数
    ang_path = r'E:\04 Method\01 Table\angle_2017_2021_sort.xlsx'  # 雷达入射角°（需要转弧度/180*pi）
    ndvi_path = r'E:\04 Method\01 Table\ndvi_2017_2021_sort.xlsx'  # 归一化植被指数

    st = pd.read_excel(st_path, index_col=0, header=0)
    sm = pd.read_excel(sm_path, index_col=0, header=0)
    ts = pd.read_excel(ts_path, index_col=0, header=0)
    vv = pd.read_excel(vv_path, index_col=0, header=0)
    vh = pd.read_excel(vh_path, index_col=0, header=0)
    ang = pd.read_excel(ang_path, index_col=0, header=0)
    ndvi = pd.read_excel(ndvi_path, index_col=0, header=0)

    sdc_2019, sdc_2020 = cal_sdc(st, sm, ts)  # 计算2019年和2020年土壤介电常数
    alpha_vv_2019, alpha_vv_2020 = cal_alpha(sdc_2019, sdc_2020, ang)  # 根据土壤介电常数计算alpha值
    solved_alpha_2019, solved_alpha_2020 = creat_coefficient_matrix(sm, vv)  # 根据vv后向散射系数创建欠定方程组
    solved_dc_2019, solved_dc_2020 = solved_dc(solved_alpha_2019, solved_alpha_2020, ang)  # alpha求解介电常数
    lut_2019, lut_2020 = lut(st, sm, ts)  # 创建查找表
    solved_sm_2019, solved_sm_2020 = solved_sm(solved_dc_2019, solved_dc_2020, lut_2019, lut_2020)  # 得到土壤湿度

    measured_sm = sm.values
    solved_sm = np.hstack((solved_sm_2019, solved_sm_2020))

    plt.rcParams['figure.figsize'] = (9, 8.5)  # 绘图分辨率
    fig, ax = plt.subplots(1, 1)
    # for ax in axs.ravel():
    ax.set_xticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    ax.set_xlim(0.0, 0.53)
    ax.set_ylim(0.0, 0.53)
    ax.plot([0.0, 0.53], [0.0, 0.53], ls="--", c=".3", linewidth=1.0, alpha=0.8)
    ax.tick_params(labelsize=14)

    # 绘制子图1
    x = measured_sm.flatten()
    y = solved_sm.flatten()
    xy = np.vstack([x, y])  # 将两个维度的数据叠加
    z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(x, y, c=z, s=30, cmap='Spectral_r')
    ax.set_xlabel(r"Measured soil moisture ($cm^3/cm^3$)", fontsize=18)
    ax.set_ylabel(r"Soil moisture retrieval based on VH backscatter ($cm^3/cm^3$)", fontsize=18)
    r = ss.pearsonr(measured_sm.flatten(), solved_sm.flatten())[0]
    R2 = r2_score(measured_sm.flatten(), solved_sm.flatten())
    RMSE = sqrt(mean_squared_error(measured_sm.flatten(), solved_sm.flatten()))
    MAE = mean_absolute_error(measured_sm.flatten(), solved_sm.flatten())
    MSE = mean_squared_error(measured_sm.flatten(), solved_sm.flatten())
    evaluation_indicator = ' R:{:.3f}  R2:{:.3f}  RMSE:{:.3f}  MAE:{:.3f}  MSE:{:.3f}'.format(r, R2, RMSE, MAE, MSE)
    ax.text(0.01, 0.96, evaluation_indicator, transform=ax.transAxes, fontdict={'size': '15'})
    fig.tight_layout()
    plt.savefig(r'E:\04 Method\02 Picture\03 变化检测法/' + '04 alpha模型' + '.png')
    plt.show()





