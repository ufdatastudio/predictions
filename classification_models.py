import sklearn

from abc import ABC, abstractmethod

from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from data_processing import DataProcessing

class SkLearnModelFactory(ABC):
    """Abstract implementation of a model. Each specified model inherits from this base class.

    Methods decorated with @abstractmethod must be implemented and if not, the interpreter will throw an error. Methods not decorated will be shared by all other classes that inherit from Model.
    """

    def __name__(self):
        return "ML MODEL BASE"
    
    def __init__(self):
        self.classifer = None
    
    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def get_model_name(self):
        return self.__name__()
    
    def select_model(model_name: str):
        """Select a model based on the provided model name
        
        Parameters:
        -----------
        model_name: `str`
            The name of the model to select
        
        Returns:
        --------
        object
            An instance of the selected model
        """
        models = {
            "perceptron": SkLearnPerceptronModel(),
            "sgdclassifier": SkLearnSGDClassifier(),
            "logistic regression": SkLearnLogisticRegression(),
            # Add other models here as needed
        }

        if model_name in models:
            return models[model_name]
        else:
            raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")

class SkLearnPerceptronModel(SkLearnModelFactory):      

    def __init__(self):
        super().__init__()  

    def __name__(self):
        return "Perceptron Model"
    
    def train_model(self, X, y):

        self.classifer = Perceptron() # instantiate the model
        self.classifer.fit(X, y) # train the model on the training data; sklearn intializes the weights and bias randomly
    
    def predict(self, X_test, to_series: bool = True):
        """Predict the test data using the trained model
        
        Parameters:
        -----------
        X_test: `pd.DataFrame`
            A DataFrame containing the test data to predict
        
        to_series: `bool`
            A boolean value to convert the predictions to a pd.Series
        """
        predictions = self.classifer.predict(X_test)
        if to_series:
            return DataProcessing.array_to_df(predictions)
        return predictions

class SkLearnSGDClassifier(SkLearnModelFactory):      

    def __init__(self):
        super().__init__()  

    def __name__(self):
        return "SDGClassifier Model"
    
    def train_model(self, X, y):

        self.classifer = SGDClassifier() # instantiate the model
        self.classifer.fit(X, y) # train the model on the training data; sklearn intializes the weights and bias randomly
    
    def predict(self, X_test, to_series: bool = True):
        """Predict the test data using the trained model
        
        Parameters:
        -----------
        X_test: `pd.DataFrame`
            A DataFrame containing the test data to predict
        
        to_series: `bool`
            A boolean value to convert the predictions to a pd.Series
        """
        predictions = self.classifer.predict(X_test)
        if to_series:
            return DataProcessing.array_to_df(predictions)
        return predictions

class SkLearnLogisticRegression(SkLearnModelFactory):      

    def __init__(self):
        super().__init__()  

    def __name__(self):
        return "Logistic Regression Model"
    
    def train_model(self, X, y):

        self.classifer = LogisticRegression() # instantiate the model
        self.classifer.fit(X, y) # train the model on the training data; sklearn intializes the weights and bias randomly
    
    def predict(self, X_test, to_series: bool = True):
        """Predict the test data using the trained model
        
        Parameters:
        -----------
        X_test: `pd.DataFrame`
            A DataFrame containing the test data to predict
        
        to_series: `bool`
            A boolean value to convert the predictions to a pd.Series
        """
        predictions = self.classifer.predict(X_test)
        if to_series:
            return DataProcessing.array_to_df(predictions)
        return predictions


class EvaluationMetric:

    def eval_accuracy(self, y_true, y_prediction):
        return accuracy_score(y_true, y_prediction)

    def eval_precision(self, y_true, y_prediction):
        return precision_score(y_true, y_prediction)

    def eval_recall(self, y_true, y_prediction):
        return recall_score(y_true, y_prediction)

    def eval_f1_score(self, y_true, y_prediction):
        return f1_score(y_true, y_prediction)
    
    def custom_evaluation_metrics(self, y_true, y_prediction):
        """Evaluate the model using accuracy and precision"""

        accuracy = self.eval_accuracy(y_true, y_prediction)
        precision = self.eval_precision(y_true, y_prediction)
        recall = self.eval_recall(y_true, y_prediction)
        f1 = self.eval_f1_score(y_true, y_prediction)

        metrics_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        return metrics_dict

    def eval_classification_report(self, y_true, y_prediction):
        print(classification_report(y_true, y_prediction))


def train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a model.
    
    Parameters:
    -----------
    model_name: `str`
        The name of the model to be trained and evaluated.
    X_train: `pd.DataFrame`
        The training features.
    y_train: `pd.Series`
        The training labels.
    X_test: `pd.DataFrame`
        The test features.
    y_test: `pd.Series`
        The test labels.
    
    Returns:
    --------
    dict
        A dictionary containing the model name and its evaluation metrics.
    pd.Series
        The predictions made by the model.
    """
    model = SkLearnModelFactory.select_model(model_name)
    model.train_model(X_train, y_train)
    predictions = model.predict(X_test)
    
    eval_metric = EvaluationMetric()
    metrics = eval_metric.custom_evaluation_metrics(y_true=y_test, y_prediction=predictions)
    metrics['Model'] = model.get_model_name()
    
    return metrics, predictions