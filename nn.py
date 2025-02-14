import math
from random import gauss
from typing import List, Optional


class Neuron:
    def __init__(self, nFeatures: int):
        assert nFeatures > 1
        self._nInp = nFeatures
        # Random weight initialization using Gaussian distibution
        # with mean 0, and std deviation 1
        self._weights: List[float] = self._initWeights()

    def fit(
        self, trainX: List[List[float]], trainY: List[int], lr=0.1, epochs=50
    ):
        lenX = len(trainX)
        assert lenX > 0
        assert lr > 0.0
        assert epochs > 1
        assert len(trainX[0]) == self._nInp
        assert lenX == len(trainY)
        # Using stochastic gradient descent instead of batch gradient descent
        for i in range(epochs):
            if i % 50 == 0:
                print(f"-------------- Epoch {i + 1} -------------")
            epochError = 0.0
            for datapoint, yActual in zip(trainX, trainY):
                s = self.dot(datapoint, self._weights[:-1]) + self._weights[-1]
                # y = self._sigmoid(s)
                predicted = 1 if s >= 0.0 else 0
                error = self._mse(yActual, predicted)
                epochError += error
                self._updateWeights(datapoint, s, predicted, yActual, lr)

            avgEpochError = epochError / lenX
            if i % 50 == 0:
                print(f"Avg. Epoch error is { avgEpochError }")
        print("\n\n------------------- Finished Training ------------")

    def predict(self, X: List[float]):
        s = self.dot(X, self._weights[:-1]) + self._weights[-1]
        # y = self._sigmoid(s)
        predicted = 1 if s >= 0.0 else 0
        return predicted

    def getAcccuracy(self, testX: List[List[float]], testY: List[int]):
        assert len(testX) == len(testY)
        nCorrect = 0
        for x, y in zip(testX, testY):
            prediction = self.predict(x)
            if prediction == y:
                nCorrect += 1
        return (nCorrect * 100.0) / len(testY)

    def _updateWeights(
        self,
        X: List[float],
        s: float,
        # y: float,
        pred: int,
        actual: int,
        lr: float,
    ):
        dEdY = self._mse_derivative(actual, pred)
        # dYdS = self._sigmoid_derivative(s)
        dYdS = 1.0
        for i, (Xi, Wi) in enumerate(zip(X, self._weights)):
            dSdWi = Xi
            dEdWi = dEdY * dYdS * dSdWi
            self._weights[i] += dEdWi * lr
        self._weights[-1] += dEdY * dYdS * lr

    def _initWeights(self) -> List[float]:
        return [gauss(0.0, 1.0) for _ in range(self._nInp + 1)]

    def resetWeights(self):
        self._weights = self._initWeights()

    @staticmethod
    def _sigmoid(X: float) -> float:
        # return 1.0 / (1 + math.exp(-X))
        # Making sigmoid numerically stable now
        if X >= 0:
            z = math.exp(-X)
            return 1 / (1 + z)
        else:
            z = math.exp(X)
            return z / (1 + z)

    def _sigmoid_derivative(self, X: float) -> float:
        f = self._sigmoid(X)
        return f * (1.0 - f)

    @staticmethod
    def _mse(actual: int, predicted: int) -> float:
        return 0.5 * math.pow(actual - predicted, 2)

    @staticmethod
    def _mse_derivative(actual: int, predicted: int) -> int:
        return actual - predicted

    @staticmethod
    def dot(X: List[float], W: List[float]) -> float:
        assert len(X) == len(W)
        return sum(x * y for x, y in zip(X, W))


if __name__ == "__main__":
    X = [[1.0, 0.0], [1.0, 1.0], [0, 1.0], [0.0, 0.0]]
    Y = [0, 1, 0, 0]
    perceptron = Neuron(2)
    print(perceptron._weights)
    perceptron.fit(X, Y, epochs=500, lr=0.01)
    print(perceptron._weights)
    for x in X:
        print(f"{x}\t{perceptron.predict(x)}")

