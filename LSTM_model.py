from alpha_model import AlphaModel
import numpy as np

class LSTMModel(AlphaModel):
    def __init__(self):
        self.model = 0

    def train(self, X, Y):
        assert len(X)==len(Y)

    def predict(self, X):
        return 0