import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt


def draw1():
    config = {
        "font.family": 'Times New Roman',
        "font.size": 110,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    plt.figure(figsize=(56, 36))

    nodes = ['5', '10', '20']
    nodes_num = np.arange(len(nodes))

    # # mae
    # plt.ylabel('MAE')
    # plt.ylim(4000, 15000)
    # ARIMA = np.array([9338, 11819, 12024])
    # SVR = np.array([6885, 8328, 9056])
    # FNN = np.array([6208, 7891, 8129])
    # LSTM = np.array([5619, 7277, 7623])
    # GRU = np.array([5548, 7345, 7875])
    # CNN_GRU = np.array([5561, 7057, 7654])
    # CNN_LSTM = np.array([5682, 7562, 7615])
    # AGGRU = np.array([4675, 6357, 6706])

    # # rmse
    # plt.ylabel('RMSE')
    # plt.ylim(8000, 23000)
    # ARIMA = np.array([16604, 19160, 19622])
    # SVR = np.array([12482, 13875, 15194])
    # FNN = np.array([11776, 13385, 13982])
    # LSTM = np.array([10876, 12690, 13298])
    # GRU = np.array([10710, 12789, 13630])
    # CNN_GRU = np.array([10879, 12344, 13308])
    # CNN_LSTM = np.array([10994, 12900, 13241])
    # AGGRU = np.array([9032, 11020, 11603])

    # # mape
    # plt.ylabel('MAPE')
    # plt.ylim(0.1, 0.3)
    # ARIMA = np.array([0.24685, 0.21372, 0.19627])
    # SVR = np.array([0.19668, 0.17708, 0.15025])
    # FNN = np.array([0.17509, 0.15940, 0.13538])
    # LSTM = np.array([0.17363, 0.15399, 0.13017])
    # GRU = np.array([0.17251, 0.15682, 0.13332])
    # CNN_GRU = np.array([0.17642, 0.15401, 0.13239])
    # CNN_LSTM = np.array([0.17714, 0.15860, 0.13151])
    # AGGRU = np.array([0.15896, 0.14237, 0.11953])

    # smape
    plt.ylabel('SMAPE')
    plt.ylim(0.02, 0.046)
    ARIMA = np.array([0.03814, 0.03915, 0.03982])
    SVR = np.array([0.02903, 0.03027, 0.03307])
    FNN = np.array([0.02407, 0.02615, 0.02674])
    LSTM = np.array([0.02342, 0.02501, 0.02547])
    GRU = np.array([0.02332, 0.02532, 0.02623])
    CNN_GRU = np.array([0.02409, 0.02486, 0.02589])
    CNN_LSTM = np.array([0.02401, 0.02581, 0.02582])
    AGGRU = np.array([0.02175, 0.02324, 0.02368])

    width = 0.05
    ARIMA_start = nodes_num
    SVR_start = nodes_num + width
    FNN_start = nodes_num + width * 2
    LSTM_start = nodes_num + width * 3
    GRU_start = nodes_num + width * 4
    CNN_LSTM_start = nodes_num + width * 5
    CNN_GRU_start = nodes_num + width * 6
    AGGRU_start = nodes_num + width * 7

    plt.bar(ARIMA_start, ARIMA, width=width, label='ARIMA', color='lightskyblue')
    plt.bar(SVR_start + 0.02, SVR, width=width, label='SVR', color='navajowhite')
    plt.bar(FNN_start + 0.04, FNN, width=width, label='FNN', color='mediumspringgreen')
    plt.bar(LSTM_start + 0.06, LSTM, width=width, label='LSTM', color='gold')
    plt.bar(GRU_start + 0.08, GRU, width=width, label='GRU', color='deepskyblue')
    plt.bar(CNN_LSTM_start + 0.10, CNN_LSTM, width=width, label='CNN-LSTM', color='thistle')
    plt.bar(CNN_GRU_start + 0.12, CNN_GRU, width=width, label='CNN-GRU', color='violet')
    plt.bar(AGGRU_start + 0.14, AGGRU, width=width, label='AGGRU', color='red')

    plt.xlabel('Number of houses')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='both', width=4, length=20)

    plt.xticks(GRU_start, ['5', '10', '20'])
    plt.legend(fontsize=100, loc='upper left', frameon=False, ncol=3)
    plt.savefig('figure.png', bbox_inches='tight', pad_inches=0.3)
    plt.tight_layout()
    plt.close()


def draw2():
    config = {
        "font.family": 'Times New Roman',
        "font.size": 110,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    plt.figure(figsize=(56, 36))

    nodes = ['25', '50', '100']
    nodes_num = np.arange(len(nodes))

    # # mae
    # plt.ylabel('MAE')
    # plt.ylim(500, 1800)
    # ARIMA = np.array([1453, 1192, 1157])
    # SVR = np.array([864, 721, 700])
    # FNN = np.array([784, 681, 652])
    # LSTM = np.array([783, 686, 666])
    # GRU = np.array([774, 684, 664])
    # CNN_GRU = np.array([783, 677, 645])
    # CNN_LSTM = np.array([770, 667, 642])
    # AGGRU = np.array([717, 640, 609])

    # # rmse
    # plt.ylabel('RMSE')
    # plt.ylim(1200, 3800)
    # ARIMA = np.array([3071, 2476, 2326])
    # SVR = np.array([2012, 1690, 1557])
    # FNN = np.array([1888, 1613, 1466])
    # LSTM = np.array([1856, 1582, 1463])
    # GRU = np.array([1837, 1596, 1453])
    # CNN_GRU = np.array([1866, 1568, 1436])
    # CNN_LSTM = np.array([1836, 1572, 1437])
    # AGGRU = np.array([1709, 1517, 1369])

    # # mape
    # plt.ylabel('MAPE')
    # plt.ylim(0.1, 0.32)
    # ARIMA = np.array([0.26158, 0.24153, 0.24593])
    # SVR = np.array([0.16589, 0.15209, 0.14445])
    # FNN = np.array([0.15425, 0.14511, 0.13656])
    # LSTM = np.array([0.15579, 0.14814, 0.13974])
    # GRU = np.array([0.15508, 0.14532, 0.14122])
    # CNN_GRU = np.array([0.15594, 0.14529, 0.13696])
    # CNN_LSTM = np.array([0.15489, 0.14273, 0.13652])
    # AGGRU = np.array([0.14573, 0.13814, 0.13009])

    # smape
    plt.ylabel('SMAPE')
    plt.ylim(0.02, 0.07)
    ARIMA = np.array([0.05683, 0.05432, 0.05524])
    SVR = np.array([0.03589, 0.03419, 0.03364])
    FNN = np.array([0.03227, 0.03242, 0.03152])
    LSTM = np.array([0.03259, 0.03273, 0.03216])
    GRU = np.array([0.03194, 0.03275, 0.03223])
    CNN_GRU = np.array([0.03267, 0.03233, 0.03137])
    CNN_LSTM = np.array([0.03243, 0.03193, 0.03117])
    AGGRU = np.array([0.03098, 0.03102, 0.02977])

    width = 0.05
    ARIMA_start = nodes_num
    SVR_start = nodes_num + width
    FNN_start = nodes_num + width * 2
    LSTM_start = nodes_num + width * 3
    GRU_start = nodes_num + width * 4
    CNN_LSTM_start = nodes_num + width * 5
    CNN_GRU_start = nodes_num + width * 6
    AGGRU_start = nodes_num + width * 7

    plt.bar(ARIMA_start, ARIMA, width=width, label='ARIMA', color='lightskyblue')
    plt.bar(SVR_start + 0.02, SVR, width=width, label='SVR', color='navajowhite')
    plt.bar(FNN_start + 0.04, FNN, width=width, label='FNN', color='mediumspringgreen')
    plt.bar(LSTM_start + 0.06, LSTM, width=width, label='LSTM', color='gold')
    plt.bar(GRU_start + 0.08, GRU, width=width, label='GRU', color='deepskyblue')
    plt.bar(CNN_LSTM_start + 0.10, CNN_LSTM, width=width, label='CNN-LSTM', color='thistle')
    plt.bar(CNN_GRU_start + 0.12, CNN_GRU, width=width, label='CNN-GRU', color='violet')
    plt.bar(AGGRU_start + 0.14, AGGRU, width=width, label='AGGRU', color='red')

    plt.xlabel('Number of houses')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='both', width=4, length=20)

    plt.xticks(GRU_start, nodes)
    plt.legend(fontsize=100, loc='upper left', frameon=False, ncol=3)
    plt.savefig('figure.png', bbox_inches='tight', pad_inches=0.3)
    plt.tight_layout()
    plt.close()


if __name__ == '__main__':
    draw2()
