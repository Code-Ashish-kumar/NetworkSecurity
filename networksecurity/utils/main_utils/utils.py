from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os,sys
import numpy as np
# import dill
import pickle

def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path , 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)

def write_yaml_file(file_path:str , content:object , replace:bool=False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'w') as yaml_file:
            yaml.dump(content,yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def save_numpy_array_data(file_path:str , array:np.array):
    '''
    save numpy array data to file
    file_path : str location of file to save
    array : np.array data to save
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    

def save_object(file_path:str , obj:object):
    '''
    file_path : str location of file to save
    obj : object to save
    '''
    try:
        logging.info(f"Saving object file : {file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
        logging.info(f"Object file saved successfully and exited from save_object function of utils")
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def load_object(file_path:str) -> object:
    '''
    file_path : str location of file to load
    return : object loaded from file
    '''
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at path : {file_path}")
        logging.info(f"Loading object file : {file_path}")
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def load_numpy_array_data(file_path:str) -> np.array:
    '''
    file_path : str location of file to load
    return : np.array data loaded from file
    '''
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at path : {file_path}")
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models:dict,params:dict) -> dict:
    '''
    Trains and evaluates given models on training and testing data and return the report containing metric scores for each model
    '''
    try:
        report = {}
        for model_name,model in models.items():
            para = params[model_name]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test,y_test_pred)
            train_model_score = r2_score(y_train,y_train_pred)
            report[model_name] = test_model_score

        return report
    except Exception as e:
        raise NetworkSecurityException(e,sys)