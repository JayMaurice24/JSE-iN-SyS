import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

def size_data(window_size, horizon, dataset):
    numSamples = len(dataset)
    # print(numSamples)
    xWindow = np.arange(0, window_size, 1)
    yHorizon = np.arange(window_size, window_size+horizon, 1)
    # print(xWindow)
    # print(yHorizon)
    x = []
    y = []
    for i in range(0, numSamples-yHorizon[len(yHorizon)-1], 1):
        xEntry = []
        yEntry = []
        for j in range(0, window_size, 1):
            xEntry.append(dataset[i + j])
        for k in range(0, len(yHorizon), 1):
            yEntry.append(dataset[i + yHorizon[k]])

        # print(xEntry)
        # print(yEntry)
        x.append(xEntry)
        y.append(yEntry)
        # print(x)
        # print(y)
    # print(len(x))
    # print(len(y))
    # print(x[0])
    # print(y[0])
    # print(x[1])
    # print(y[1])
    return x, y


def read_company_data(path, company = "GROWPNT"):
    df = pd.read_csv(path)
    companyData = df[company]
    # print(companyData)
    return companyData


def split_data(x_dataset, y_dataset, train_size = 0.8, test_size = 0.1, val_size = 0.1):
    numSamples = len(x_dataset)
    # print(numSamples)
    numTest = round(numSamples * test_size)
    numValidate = round(numSamples * val_size)
    numTrain = round(numSamples * train_size)
    xTrain, yTrain = x_dataset[:numTrain], y_dataset[:numTrain]
    xVal, yVal = (
        x_dataset[numTrain: numTrain + numValidate],
        y_dataset[numTrain: numTrain + numValidate],
    )
    xTest, yTest = x_dataset[-numTest:], y_dataset[-numTest:]
    # print(len(xTrain), len(yTrain))
    # print(len(xVal), len(yVal))
    # print(len(xTest), len(yTest))
    # print((xTrain[0]), (yTrain[0]))
    # print((xVal[0]), (yVal[0]))
    # print((xTest[0]), (yTest[0]))
    return xTrain, yTrain, xVal, yVal, xTest, yTest


def partition_data(num_partitions, dataset):
    numSamples = len(dataset)
    # print(numSamples)
    partitionSize = round(numSamples/num_partitions)
    # print(partitionSize)
    partedData = []
    runningTotal = 0
    for i in range(0, num_partitions - 1, 1):
        temp = dataset[i*partitionSize: (partitionSize + i*partitionSize)]
        # print(len(temp))
        runningTotal += len(temp)
        partedData.append(temp)
    remaining = numSamples - ((num_partitions - 1) * partitionSize)
    # print(remaining)
    temp = dataset[-remaining:]
    runningTotal += len(temp)
    partedData.append(temp)
    # print(len(partedData))
    # print(runningTotal)
    # print(len(partedData))
    return partedData


# path = "JSE_clean.csv"
# comp = read_company_data(path)
#
# partitionedComp = partition_data(5, comp)
#
# data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# xData, yData = size_data(120, 4, comp)
# xTraining, yTraining, xVal, yVal, xTest, yTest = split_data(xData, yData, 0.7, 0.3, 0)


