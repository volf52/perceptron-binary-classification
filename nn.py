from random import gauss
from typing import List, Optional


class Neuron:
    def __init__(self, nFeatures: int):
        self._nInp = nFeatures
        # Random weight initialization using Gaussian distibution
        # with mean 0, and std deviation 1
        self._weights: List[float] = [
            gauss(0.0, 1.0) for _ in range(nFeatures)
        ]
        self._preActivation: Optional[float] = None
        self._y: Optional[float] = None
        self._inputs: Optional[List[float]] = None

    def fit(self, trainX: List[List[float]], trainY: List[int]):
        raise NotImplementedError

    def predict(self, X: List[float]):
        raise NotImplementedError

    def getAcccuracy(self, testX: List[List[float]], testY: List[int]):
        raise NotImplementedError

    def _updateWeights_(self):
        raise NotImplementedError

    @staticmethod
    def _sigmoid_(X: float):
        raise NotImplementedError

    @staticmethod
    def _sigmoid_derivative_(X: float):
        raise NotImplementedError


if __name__ == "__main__":
    perceptron = Neuron(10)
    print(perceptron._weights)
