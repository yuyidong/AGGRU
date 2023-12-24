import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def main():
    path = 'data/GEFCom2012.xlsx'
    timestep = 24
    num_nodes = 5

    data = pd.read_excel(path)
    for i in data.columns:
        data = data[data[i] > 0]
    data = data.values
    data = data[:500]
    dataX, dataY = [], []
    for index in range(len(data) - timestep):
        dataX.append(data[index:index + timestep, :num_nodes])
        dataY.append(data[index + timestep, :num_nodes])
    dataX, dataY = np.array(dataX), np.array(dataY)

    x_tran, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.25)

    print('starting...')
    mae = 0
    for i in range(num_nodes):
        clf = SVR(kernel='linear', C=1.25)
        clf.fit(x_tran[:, :, i], y_train[:, i])
        y_hat = clf.predict(x_test[:, :, i])
        mae += mean_absolute_error(y_test[:, i], y_hat)
        print(mae)
    print(mae / num_nodes)


if __name__ == '__main__':
    main()
