import csv
import random
from math import ceil
from pathlib import Path
from typing import List, Tuple, Union, cast

from nn import Neuron

xType = float
yType = int
dataType = List[Union[List[xType], int]]


def readData(dataFile: Path) -> List[dataType]:
    data: List[dataType] = list()
    with open(dataFile, "r") as csvFile:
        reader = csv.reader(csvFile, delimiter=",")
        for row in reader:
            x = list(map(float, row[:4]))
            y = int(row[4])
            data.append([x, y])
    return data


def xySplit(data: List[dataType]) -> Tuple[List[List[xType]], List[yType]]:
    X: List[List[xType]] = list()
    Y: List[yType] = list()
    for dp in data:
        x = cast(List[xType], dp[0])
        y = cast(yType, dp[1])
        X.append(x)
        Y.append(y)
    return X, Y


def trainTestSplit(
    data: List[dataType], split=0.3
) -> Tuple[List[dataType], List[dataType]]:
    val = ceil(len(data) * split)
    return data[:-val], data[-val:]


def prepareData(
    dataFile: Path, split=0.3
) -> Tuple[List[List[xType]], List[yType], List[List[xType]], List[yType]]:
    data = readData(dataFile)
    random.shuffle(data)
    train, test = trainTestSplit(data, split)
    trainX, trainY = xySplit(train)
    testX, testY = xySplit(test)
    return trainX, trainY, testX, testY


if __name__ == "__main__":
    dataFile = Path("./data_banknote_authentication.csv")
    # shuffle data
    trainX, trainY, testX, testY = prepareData(dataFile)

    assert len(trainX) > 0
    perceptron = Neuron(len(trainX[0]))
    print(perceptron._nInp)
