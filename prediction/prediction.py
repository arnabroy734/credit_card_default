from validation.validation import Validation
from preprocessing.preprocessing import Preprocessor
from path.path import PREPROCESSOR, BEST_MODEL, PREDICTION_OUTPUT 
import pickle
import pandas as pd

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
        status, mesage = Validation().validate_prediction()
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
        status, data = preprocessor.prediction_preprocess()
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
            y_pred = model.predict(data)
            data['default'] = y_pred
            data.to_csv(PREDICTION_OUTPUT)
        except Exception as e:
            return (False, f"Something went worng during final prediction stage - {e}")
        
        return (True, "Prediction successful")

        
