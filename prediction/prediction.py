from validation.validation import Validation
from preprocessing.preprocessing import Preprocessor
from path.path import PREPROCESSOR, BEST_MODEL, PREDICTION_OUTPUT, PREDICTION_INPUT 
import pickle
import pandas as pd
import numpy as np

class PredictionPipeline:
    """
    Description:
    This class does the following steps:
    1. Validate the data uploaded by user
    2. De-serialise the preprocesor
    3. Preprocess the input data
    3. Read the preprocessed data
    4. De-serialise the saved model
    5. Do the prediction
    6. Save the prediction file
    """
    def predict(self):
        """
        Input: None
        Output: 
            (status, message) - if status is False there is something wrong so send a error message 
                                if status is True prediction is successful, so send success message
        """
        # Validation
        status, message = Validation().validate_prediction()
        if status == False:
            return (False, message)

        # Preprocessor: de-serialisation
        try:
            with open(PREPROCESSOR, 'rb') as f:
                preprocessor = pickle.load(f)
                f.close()
        except:
            return (False, "Saved preprocessor model cannot be de-serialised")
        
        # Preprocessing
        status, data_preprocessed = preprocessor.prediction_preprocess()
        if  status == False:
            return (False, "Something went wrong during preprocessing the data")
        
        
        # De-serialise model
        try:
            with open(BEST_MODEL, 'rb') as f:
                model = pickle.load(f)
                f.close()
        except:
            return (False, "Model cannot be loaded")
        
        # Do prediction and save
        try:
            y_pred = model.predict(data_preprocessed)
            data = pd.read_excel(PREDICTION_INPUT, index_col='ID')
            data['default'] = y_pred
            data.to_csv(PREDICTION_OUTPUT)
            total_customers = data.shape[0]
            total_defaulters = np.count_nonzero(y_pred)
        except Exception as e:
            return (False, f"Something went worng during final prediction stage - {e}")
        
        
        return (True, f"Prediction successful. Out of {total_customers} customers {total_defaulters} are going to default next month.")

        
