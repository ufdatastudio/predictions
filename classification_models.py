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
    def __name__(self):
        return "Perceptron Model"
    
    def train_model(self, X_train, X_test, y_train, y_test): 

        technique = Perceptron(tol=1e-3, random_state=0) # instantiate the model
        technique.fit(X_train, y_train) # train the model on the training data; sklearn intializes the weights and bias randomly
        y_train_predictions = technique.predict(X_train)
        y_test_predictions = technique.predict(X_test)


        # train_metrics = train_eval_metric(y_train, y_train_predictions)
        # test_metrics = test_eval_metric(y_test, y_test_predictions)

        return y_train_predictions, y_test_predictions

class EvaluationMetric:

    def eval_accuracy(self, y_true, y_prediction):
        return sklearn.metrics.accuracy_score(y_true, y_prediction)

    def eval_precision(self, y_true, y_prediction):
        return sklearn.metrics.precision_score(y_true, y_prediction)

    def eval_recall(self, y_true, y_prediction):
        return sklearn.metrics.recall_score(y_true, y_prediction)

    def eval_f1_score(self, y_true, y_prediction):
        return sklearn.metrics.f1_score(y_true, y_prediction)

    def eval_metrics(self, y_train_true, y_train_predictions):
        accuracy = self.eval_accuracy(y_train_true, y_train_predictions)
        precision = self.eval_precision(y_train_true, y_train_predictions)
        recall = self.eval_recall(y_train_true, y_train_predictions)
        f1 = self.eval_f1_score(y_train_true, y_train_predictions)

        metrics_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        return metrics_dict