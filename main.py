from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.entity.config_entity import ModelTrainerConfig

import sys
import mlflow
import os

if __name__ == "__main__":
    try:
        # Configure MLflow tracking
        mlflow.set_tracking_uri("file:mlruns")  # Local SQLite backend
        mlflow.set_experiment("Network_Security_Model")

        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiate the data ingestion")
        data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")
        print(data_ingestion_artifacts)

        logging.info("Initiate the data validation")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifacts, data_validation_config)
        data_validation_artifacts = data_validation.initiate_data_validation()
        logging.info("Data validation completed")   
        print(data_validation_artifacts)

        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config, data_validation_artifacts)
        logging.info("Initiate the data transformation")
        data_transformation_artifacts = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed")
        print(data_transformation_artifacts)

        logging.info("Initiate the model trainer")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifacts)
        model_trainer_artifacts = model_trainer.initiate_model_trainer()
        logging.info("Model trainer completed")
    except Exception as e:
        raise NetworkSecurityException(e,sys)