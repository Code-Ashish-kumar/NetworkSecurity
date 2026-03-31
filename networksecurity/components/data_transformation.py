import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constants.training_pipeline import TARGET_COLUMN
from networksecurity.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from networksecurity.entity.artifact_entity import DataValidationArtifact
from networksecurity.entity.artifact_entity import DataTransformationArtifact

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.utils.main_utils.utils import save_numpy_array_data
from networksecurity.utils.main_utils.utils import save_object

class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig,
                 data_validation_artifact:DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame :
        try:
            return pd.read_csv(file_path)  
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def get_data_transformer_object(cls) -> Pipeline:
        '''
        It initialize KNNImputer object with parameters specified in the training pipeline constants file 
        and return a pipeline object with imputer as step

        Args :
            cls : class method reference(DataTransformation)

        Returns :
            A pipeline object
        '''
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            pipeline = Pipeline([("imputer", imputer)])
            return pipeline
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            # reading validated train and test file
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # training dataframe
            input_feature_train_df = train_df.drop(columns=TARGET_COLUMN)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1,0)

            #test dataframe
            input_feature_test_df = test_df.drop(columns=TARGET_COLUMN)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1,0)

            preprocessor_object = self.get_data_transformer_object()
            preprocessor_object.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            train_arr = np.c_[transformed_input_train_feature , target_feature_train_df.to_numpy()]
            test_arr = np.c_[transformed_input_test_feature , target_feature_test_df.to_numpy()]

            #save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,test_arr)  
            #save preprocessor object
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor_object)

            # prepare artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path
            )
            return data_transformation_artifact
        
            logging.info("Data transformation completed successfully")
        except Exception as e:
            raise NetworkSecurityException(e,sys)