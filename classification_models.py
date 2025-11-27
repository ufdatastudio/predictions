import sklearn

from abc import ABC, abstractmethod

from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegression, RidgeClassifier, LinearRegression, ElasticNet
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from data_processing import DataProcessing

class SkLearnModelFactory(ABC):
    """Abstract implementation of a model. Each specified model inherits from this base class.

    Methods decorated with @abstractmethod must be implemented and if not, the interpreter will throw an error. Methods not decorated will be shared by all other classes that inherit from Model.
    """

    @abstractmethod
    def __name__(self):
        return "ML MODEL BASE"
    
    def get_model_name(self):
        return self.__name__()

    def __init__(self):
        self.classifer = None
    
    @abstractmethod
    def train_model(self):
        pass

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
            "sgd_classifier": SkLearnSGDClassifier(),
            "logistic_regression": SkLearnLogisticRegression(),
            "ridge_classifier": SkLearnRidgeClassifier(),
            "linear_regression": SkLearnLinearRegression(),
            "elastic_net": SkLearnElasticNet(),
            "decision_tree_classifier": SkLearnDecisionTreeClassifier(),
            "random_forest_classifier": SkLearnRandomForestClassifier(),
            "gradient_boosting_classifier": SkLearnGradientBoostingClassifier(),
            "support_vector_machine_classifier": SkLearnSVC()
        }

        if model_name in models:
            return models[model_name]
        else:
            raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")

class SkLearnPerceptronModel(SkLearnModelFactory):      

    def __init__(self):
        super().__init__()  

    def __name__(self):
        return "Perceptron"
    
    def train_model(self, X, y):

        self.classifer = Perceptron() # instantiate the model
        self.classifer.fit(X, y) # train the model on the training data; sklearn intializes the weights and bias randomly

class SkLearnSGDClassifier(SkLearnModelFactory):      

    def __init__(self):
        super().__init__()  

    def __name__(self):
        return "SDG Classifier"
    
    def train_model(self, X, y):

        self.classifer = SGDClassifier() # instantiate the model
        self.classifer.fit(X, y) # train the model on the training data; sklearn intializes the weights and bias randomly

class SkLearnLogisticRegression(SkLearnModelFactory):      

    def __init__(self):
        super().__init__()  

    def __name__(self):
        return "Logistic Regression"
    
    def train_model(self, X, y):

        self.classifer = LogisticRegression() # instantiate the model
        self.classifer.fit(X, y) # train the model on the training data; sklearn intializes the weights and bias randomly

class SkLearnRidgeClassifier(SkLearnModelFactory):      

    def __init__(self):
        super().__init__()  

    def __name__(self):
        return "Ridge Classifier"
    
    def train_model(self, X, y):

        self.classifer = RidgeClassifier() # instantiate the model
        self.classifer.fit(X, y) # train the model on the training data; sklearn intializes the weights and bias randomly

class SkLearnLinearRegression(SkLearnModelFactory):      

    def __init__(self):
        super().__init__()  

    def __name__(self):
        return "Linear Regression"
    
    def train_model(self, X, y):

        self.classifer = LinearRegression() # instantiate the model
        self.classifer.fit(X, y) # train the model on the training data; sklearn intializes the weights and bias randomly

class SkLearnElasticNet(SkLearnModelFactory):      

    def __init__(self):
        super().__init__()  

    def __name__(self):
        return "Elastic Net"
    
    def train_model(self, X, y):

        self.classifer = ElasticNet() # instantiate the model
        self.classifer.fit(X, y) # train the model on the training data; sklearn intializes the weights and bias randomly

class SkLearnDecisionTreeClassifier(SkLearnModelFactory):      

    def __init__(self):
        super().__init__()  

    def __name__(self):
        return "Decision Tree"
    
    def train_model(self, X, y):

        self.classifer = DecisionTreeClassifier() # instantiate the model
        self.classifer.fit(X, y) # train the model on the training data; sklearn intializes the weights and bias randomly

class SkLearnRandomForestClassifier(SkLearnModelFactory):      

    def __init__(self):
        super().__init__()  

    def __name__(self):
        return "Random Forest"
    
    def train_model(self, X, y):

        self.classifer = RandomForestClassifier() # instantiate the model
        self.classifer.fit(X, y) # train the model on the training data; sklearn intializes the weights and bias randomly

class SkLearnGradientBoostingClassifier(SkLearnModelFactory):      

    def __init__(self):
        super().__init__()  

    def __name__(self):
        return "Gradient Boosting Machine"
    
    def train_model(self, X, y):

        self.classifer = GradientBoostingClassifier() # instantiate the model
        self.classifer.fit(X, y) # train the model on the training data; sklearn intializes the weights and bias randomly

class SkLearnSVC(SkLearnModelFactory):      

    def __init__(self):
        super().__init__()  

    def __name__(self):
        return "Support Vector Machine"
    
    def train_model(self, X, y):

        self.classifer = SVC() # instantiate the model
        self.classifer.fit(X, y) # train the model on the training data; sklearn intializes the weights and bias randomly
        