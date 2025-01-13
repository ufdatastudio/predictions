import sklearn

from abc import ABC, abstractmethod

from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

class Models(ABC):
    """Abstract implementation of a model. Each specified model inherits from this base class.

    Methods decorated with @abstractmethod must be implemented and if not, the interpreter will throw an error. Methods not decorated will be shared by all other classes that inherit from Model.
    """

    def __name__(self):
        return "ML MODEL BASE"
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def train_model(self):
        pass

    # @abstractmethod
    # def train_eval_metric(self):
    #     pass

    # @abstractmethod
    # def test_eval_metric(self):
    #     pass

class PerceptronModel(Models):
    def train_model(self, X_train, X_test, y_train, y_test): 

        technique = Perceptron(tol=1e-3, random_state=0)
        technique.fit(X_train, y_train)
        y_train_predictions = technique.predict(X_train)
        y_test_predictions = technique.predict(X_test)


        # train_metrics = train_eval_metric(y_train, y_train_predictions)
        # test_metrics = test_eval_metric(y_test, y_test_predictions)

        return y_train_predictions, y_test_predictions
