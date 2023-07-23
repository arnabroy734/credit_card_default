from path.path import UPLOADED_DATA
import os
from logs.logging import LOGGER
import logging
import pandas as pd

class TrainDataValidation:
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
        self.columns = ['LIMIT_BAL', 'SEX', 'EDUCATION','MARRIAGE','AGE', 'PAY_1', 'PAY_2', 'PAY_3',
                        'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 
                        'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'default'
                        ]


    def validate(self):

        # Write logs for validation start
        LOGGER.log_validation(message="Starting validation of training data", level=logging.INFO)

        # Step - 1
        if os.path.exists(UPLOADED_DATA):
            LOGGER.log_validation(message=f"File {UPLOADED_DATA} exists", level=logging.INFO)
        else:
            LOGGER.log_validation(message=f"File {UPLOADED_DATA} does not exist\n\n", level=logging.ERROR)
            raise Exception()
        
        # Step 2
        try:
            data = pd.read_excel(UPLOADED_DATA,  index_col='ID')
            LOGGER.log_validation(message="File opened successfully", level=logging.INFO)
        except Exception as e:
            LOGGER.log_validation(message=f"File cannot be opened {e}\n\n", level=logging.ERROR)
            raise Exception()
        
        # Step 3:
        for idx, column in enumerate(data.columns):
            if column != self.columns[idx]:
                LOGGER.log_validation(message=f"Column name {column} mismatch\n\n", level=logging.ERROR)
                raise Exception()
        LOGGER.log_validation(message="Column name validation successful", level=logging.INFO)

        # Step 4
        for column in data.columns:
            if data[column].dtype == 'O':
                LOGGER.log_validation(message=f"Datatype mismatch in column {column}\n\n", level=logging.ERROR)
                raise Exception()
        LOGGER.log_validation(message="Datatype validation successful", level=logging.INFO)

        # Step 5:
        for column in data.columns:
            if data[column].isna().sum() != 0:
                LOGGER.log_validation(message=f"Missing value found in {column}\n\n", level=logging.ERROR)
                raise Exception()
        LOGGER.log_validation(message="Missing value validation successful", level=logging.INFO)

        LOGGER.log_validation(message="End of successful training data validation\n\n", level=logging.INFO)