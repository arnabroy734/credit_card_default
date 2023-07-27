from path.path import UPLOADED_DATA, PREDICTION_INPUT
import os
from logs.logging import LOGGER
import logging
import pandas as pd

class Validation:
    """
    Description: 
    
    This class performs the following validation
    1. Check the existence of the file in the designated directory
    2. Try to open the file
    3. Check the column names sequentially
    4. Check data type
    5. Check null values/ missing values
    
    If any of the above step fails - 
    1. Fails the validation
    2. Write proper logs
    """
    def __init__(self):
        self.train_columns = ['LIMIT_BAL', 'SEX', 'EDUCATION','MARRIAGE','AGE', 'PAY_1', 'PAY_2', 'PAY_3',
                        'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 
                        'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'default']
        
        self.prediction_columns = ['LIMIT_BAL', 'SEX', 'EDUCATION','MARRIAGE','AGE', 'PAY_1', 'PAY_2', 'PAY_3',
                        'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 
                        'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']



    def validate_train(self):
        """
        This method returns True of all validation steps are successful, otherwise returns False
        """
        # Write logs for validation start
        LOGGER.log_validation(message="Starting validation of training data", level=logging.INFO)
        try:
            self.check_file(UPLOADED_DATA)
            data = self.open_file(UPLOADED_DATA)
            self.check_columns(data, self.train_columns)
            self.check_dtype(data)
            self.check_null(data)
            LOGGER.log_validation(message="End of successful training data validation\n\n", level=logging.INFO)
            return True
        except:
            return False
    
    def validate_prediction(self):
        """
        Description: This method validated the input data from user
        Input : None
        Output: 
            (status, error)  - status is either True or False and error is the error message in validation
        """
        LOGGER.log_validation(message="Starting validation of prediction input data", level=logging.INFO)
        try:
            self.check_file(PREDICTION_INPUT)
        except:
            return (False, "File prediction_input.xls not uploaded")
        
        try:
            data = self.open_file(PREDICTION_INPUT)
        except:
            return (False, "File prediction_input.xls cannot be read")
        
        try:
            self.check_columns(data, self.train_columns)
        except:
            return (False, "Please check the column names of your input file")
        
        try:
            self.check_dtype(data)
        except:
            return (False, "Please check data type of your input file")
        
        try:
            self.check_null(data)
        except:
            return (False, "Some values are missing in your input file")
        
        LOGGER.log_validation(message="End of successful prediction input data validation\n\n", level=logging.INFO)

        return (True, "Prediction data validation successful")
        
    
    # Step - 1
    def check_file(self, filename):
        if os.path.exists(filename):
            LOGGER.log_validation(message=f"File {filename} exists", level=logging.INFO)
        else:
            LOGGER.log_validation(message=f"File {filename} does not exist\n\n", level=logging.ERROR)
            raise Exception()
    
    # Step -2
    def open_file(self, filename):
        try:
            data = pd.read_excel(filename,  index_col='ID')
            LOGGER.log_validation(message=f"File {filename} opened successfully", level=logging.INFO)
            return data
        except Exception as e:
            LOGGER.log_validation(message=f"File {filename} cannot be opened {e}\n\n", level=logging.ERROR)
            raise Exception()
    
    # Step - 3
    def check_columns(self, data, columns):
        for idx, column in enumerate(data.columns):
            if column != columns[idx]:
                LOGGER.log_validation(message=f"Column name {column} mismatch\n\n", level=logging.ERROR)
                raise Exception()
        LOGGER.log_validation(message="Column name validation successful", level=logging.INFO)
    
    # Step - 4
    def check_dtype(self, data):
        for column in data.columns:
            if data[column].dtype == 'O':
                LOGGER.log_validation(message=f"Datatype mismatch in column {column}\n\n", level=logging.ERROR)
                raise Exception()
        LOGGER.log_validation(message="Datatype validation successful", level=logging.INFO)
    
    # Step - 5
    def check_null(self, data):
        for column in data.columns:
            if data[column].isna().sum() != 0:
                LOGGER.log_validation(message=f"Missing value found in {column}\n\n", level=logging.ERROR)
                raise Exception()
        LOGGER.log_validation(message="Missing value validation successful", level=logging.INFO)

        