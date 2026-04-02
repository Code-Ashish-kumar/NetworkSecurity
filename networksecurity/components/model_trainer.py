import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact

from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.main_utils.utils import load_object,save_object

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
import mlflow
from datetime import datetime

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def track_mlflow(self, best_model, best_model_name, classification_train_metric, classification_test_metric):
        with mlflow.start_run(run_name=f"Model Trainer - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"):
            # Log model name and type
            mlflow.log_param("model_name", best_model_name)
            mlflow.log_param("model_type", str(type(best_model).__name__))
            
            # Log train metrics
            mlflow.log_metric("train_f1_score", classification_train_metric.f1_score)
            mlflow.log_metric("train_precision_score", classification_train_metric.precision_score)
            mlflow.log_metric("train_recall_score", classification_train_metric.recall_score)
            
            # Log test metrics
            mlflow.log_metric("test_f1_score", classification_test_metric.f1_score)
            mlflow.log_metric("test_precision_score", classification_test_metric.precision_score)
            mlflow.log_metric("test_recall_score", classification_test_metric.recall_score)
            
            # Log the model
            mlflow.sklearn.log_model(best_model, artifact_path="model")
            
            logging.info(f"MLflow run logged successfully with run_id: {mlflow.active_run().info.run_id}")

    def train_model(self,x_train,y_train,x_test,y_test):
        '''
        Trains the model on x and y data and return the trained model object
        '''
        try:
            models = {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
            }

            params = {
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    # "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "AdaBoost": {
                    # "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Logistic Regression": {
                    # "C": [0.1, 0.01, 0.05, 0.001]
                },
                "Decision Tree": {
                    "criterion": ['gini', 'entropy', 'log_loss']
                },
                "K-Neighbors Classifier": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    # "weights": ['uniform', 'distance']
                }
            }
            
            model_report : dict = evaluate_models(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                                  models=models,params=params)
            
            best_model_score = max(model_report.values())
            best_model_name = [model_name for model_name,model_score in model_report.items() if model_score == best_model_score][0]
            best_model = models[best_model_name]

            best_model.fit(x_train,y_train)

            y_train_pred = best_model.predict(x_train)
            classification_train_metric = get_classification_score(y_true=y_train,y_pred=y_train_pred)

            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test,y_pred=y_test_pred)

            ## Track the experiment with MLFlow (single run with both train and test metrics)
            self.track_mlflow(best_model=best_model, best_model_name=best_model_name, 
                            classification_train_metric=classification_train_metric,
                            classification_test_metric=classification_test_metric)

            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            Network_Model = NetworkModel(preprocessor=preprocessor,model=best_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=Network_Model)

            ## Model Trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                        train_metric_artifact=classification_train_metric,
                                                        test_metric_artifact=classification_test_metric)
            
            logging.info(f"Model trainer artifact : {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model trainer component")
            
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading transformed train and test array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # splitting input and target feature from train and test array
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            model_trainer_artifact = self.train_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)