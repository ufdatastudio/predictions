import sklearn

from abc import ABC, abstractmethod

from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegression, RidgeClassifier, LinearRegression, ElasticNet
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

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
    
    def __init__(self, random_state=42):
        self.classifer = None
        self.random_state = random_state
    
    @abstractmethod
    def train_model(self):
        pass
    
    def predict(self, X_test, to_series: bool = True):
        predictions = self.classifer.predict(X_test)
        if to_series:
            return DataProcessing.array_to_df(predictions)
        return predictions
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for X_test.
        
        Parameters
        ----------
        X_test : array-like
            Test samples
        
        Returns
        -------
        np.ndarray
            Probability estimates of shape (n_samples, n_classes)
        
        Notes
        -----
        Some models (LinearRegression, ElasticNet, RidgeClassifier) don't support
        predict_proba. This method will raise an error for those models.
        """
        if not hasattr(self.classifer, 'predict_proba'):
            raise AttributeError(
                f"{self.__name__()} does not support predict_proba. "
                f"LIME requires probability predictions."
            )
        return self.classifer.predict_proba(X_test)
    
    def get_score(self, X, y):
        """Mean accuracy."""
        return self.classifer.score(X, y)

    @staticmethod
    def select_model(model_name: str, random_state=42):
        """Select a model with specified random state."""
        models = {
            "perceptron": SkLearnPerceptronModel(random_state),
            "sgd_classifier": SkLearnSGDClassifier(random_state),
            "logistic_regression": SkLearnLogisticRegression(random_state),
            "ridge_classifier": SkLearnRidgeClassifier(random_state),
            "linear_regression": SkLearnLinearRegression(random_state),
            "elastic_net": SkLearnElasticNet(random_state),
            "decision_tree_classifier": SkLearnDecisionTreeClassifier(random_state),
            "random_forest_classifier": SkLearnRandomForestClassifier(random_state),
            "gradient_boosting_classifier": SkLearnGradientBoostingClassifier(random_state),
            "support_vector_machine_classifier": SkLearnSVC(random_state),
            "x_gradient_boosting_classifier": CustomXGBClassifier(random_state)
        }
        if model_name in models:
            return models[model_name]
        else:
            raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")

# Models that NEED random_state:
class SkLearnSGDClassifier(SkLearnModelFactory):      
    def __init__(self, random_state=42):
        super().__init__(random_state)
    
    def __name__(self):
        return "SDG Classifier"
    
    def train_model(self, X, y):
        self.classifer = SGDClassifier(random_state=self.random_state, loss='log_loss')
        self.classifer.fit(X, y)

class SkLearnLogisticRegression(SkLearnModelFactory):      
    def __init__(self, random_state=42):
        super().__init__(random_state)
    
    def __name__(self):
        return "Logistic Regression"
    
    def train_model(self, X, y):
        self.classifer = LogisticRegression(random_state=self.random_state, max_iter=1000)
        self.classifer.fit(X, y)

class SkLearnDecisionTreeClassifier(SkLearnModelFactory):      
    def __init__(self, random_state=42):
        super().__init__(random_state)
    
    def __name__(self):
        return "Decision Tree"
    
    def train_model(self, X, y):
        self.classifer = DecisionTreeClassifier(random_state=self.random_state)
        self.classifer.fit(X, y)

class SkLearnRandomForestClassifier(SkLearnModelFactory):      
    def __init__(self, random_state=42):
        super().__init__(random_state)
    
    def __name__(self):
        return "Random Forest"
    
    def train_model(self, X, y):
        self.classifer = RandomForestClassifier(random_state=self.random_state)
        self.classifer.fit(X, y)

class SkLearnGradientBoostingClassifier(SkLearnModelFactory):      
    def __init__(self, random_state=42):
        super().__init__(random_state)
    
    def __name__(self):
        return "Gradient Boosting Machine"
    
    def train_model(self, X, y):
        self.classifer = GradientBoostingClassifier(random_state=self.random_state)
        self.classifer.fit(X, y)

class CustomXGBClassifier(SkLearnModelFactory):      
    def __init__(self, random_state=42):
        super().__init__(random_state)
    
    def __name__(self):
        return "X Gradient Boosting Machine"
    
    def train_model(self, X, y):
        self.classifer = XGBClassifier(random_state=self.random_state, probability=True)
        self.classifer.fit(X, y)

# Models that DON'T need random_state (deterministic):
class SkLearnPerceptronModel(SkLearnModelFactory):      
    def __init__(self, random_state=42):
        super().__init__(random_state)
    
    def __name__(self):
        return "Perceptron"
    
    def train_model(self, X, y):
        self.classifer = Perceptron(random_state=self.random_state)  # Has random_state parameter
        self.classifer.fit(X, y)

class SkLearnRidgeClassifier(SkLearnModelFactory):      
    def __init__(self, random_state=42):
        super().__init__(random_state)
    
    def __name__(self):
        return "Ridge Classifier"
    
    def train_model(self, X, y):
        self.classifer = RidgeClassifier(random_state=self.random_state)  # Has random_state parameter
        self.classifer.fit(X, y)

# LinearRegression and ElasticNet - keep as-is (deterministic, no random_state param)
class SkLearnLinearRegression(SkLearnModelFactory):      
    def __init__(self, random_state=42):
        super().__init__(random_state)
    
    def __name__(self):
        return "Linear Regression"
    
    def train_model(self, X, y):
        self.classifer = LinearRegression()  # No random_state parameter
        self.classifer.fit(X, y)

class SkLearnElasticNet(SkLearnModelFactory):      
    def __init__(self, random_state=42):
        super().__init__(random_state)
    
    def __name__(self):
        return "Elastic Net"
    
    def train_model(self, X, y):
        self.classifer = ElasticNet(random_state=self.random_state)  # Has random_state parameter
        self.classifer.fit(X, y)

class SkLearnSVC(SkLearnModelFactory):      
    def __init__(self, random_state=42):
        super().__init__(random_state)
    
    def __name__(self):
        return "Support Vector Machine"
    
    def train_model(self, X, y):
        self.classifer = SVC(kernel='linear', C=1, random_state=self.random_state,  probability=True)  # Has random_state parameter
        return self.classifer.fit(X, y)

    def get_score(self, X, y):
        return self.classifer.score(X, y)