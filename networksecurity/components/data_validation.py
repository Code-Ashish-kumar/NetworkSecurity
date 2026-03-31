from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.entity.artifact_entity import DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.logging.logger import logging
from scipy.stats import ks_2samp
import pandas as pd
import os , sys
from networksecurity.utils.main_utils.utils import read_yaml_file
from networksecurity.utils.main_utils.utils import write_yaml_file

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame :
        return pd.read_csv(file_path)
    
    def validate_number_of_columns(self , dataframe : pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self.schema_config["columns"])
            logging.info(f"Required number of columns : {number_of_columns}")
            logging.info(f"Dataframe has columns : {dataframe.shape[1]}")
            if dataframe.shape[1] == number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def check_numeric_columns_exist(self , dataframe : pd.DataFrame) -> bool:
        try:
            numeric_columns = self.schema_config["numeric_columns"]
            logging.info(f"Required numeric columns : {numeric_columns}")
            dataframe_columns = dataframe.columns.to_list()
            logging.info(f"Dataframe columns : {dataframe_columns}")
            for column in numeric_columns:
                if column not in dataframe_columns:
                    logging.info(f"Column : {column} is not present in dataframe")
                    return False
            return True
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def detect_data_drift(self , base_df : pd.DataFrame , current_df : pd.DataFrame , threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                p_value = ks_2samp(d1,d2).pvalue
                isFound = False
                if p_value < threshold:
                    logging.info(f"Data drift detected in column : {column} with p-value : {p_value}")
                    status = False
                    isFound = True
                else:
                    logging.info(f"No data drift detected in column : {column} with p-value : {p_value}")
                report.update({column : {
                    "p_value" : p_value,
                    "drift_found" : isFound
                }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            #create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def initiate_data_validation(self) -> DataValidationArtifact :
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            ## read the data from train and test
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            ## validate number of columns
            status = self.validate_number_of_columns(train_df)
            if not status:
                raise Exception(f"Train dataframe does not have required number of columns")

            status = self.validate_number_of_columns(test_df)
            if not status:
                raise Exception(f"Test dataframe does not have required number of columns")
            
            ## check if numeric columns are present
            # status = self.check_numeric_columns_exist(train_df)
            # if not status:
            #     raise Exception(f"Train dataframe does not have required numeric columns")

            # status = self.check_numeric_columns_exist(test_df)
            # if not status:
            #     raise Exception(f"Test dataframe does not have required numeric columns")

            ## lets check data drift
            # we will use ks_2samp test to check data drift
            status = self.detect_data_drift(base_df=train_df,current_df=test_df)
            if not status:
                raise Exception(f"Data drift detected in train and test dataframe")

            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)
           
            train_df.to_csv(self.data_validation_config.valid_train_file_path,index=False,header=True)

            test_df.to_csv(self.data_validation_config.valid_test_file_path,index=False,header=True)        
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)