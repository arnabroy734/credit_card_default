from validation.validation import Validation
from preprocessing.preprocessing import Preprocessor
from training.tune_model import find_best_model
from prediction.prediction import PredictionPipeline

if __name__=="__main__":
    # if Validation().validate_train():
    #     print("Validation of train data successful")
    # else:
    #     print("Validation of train data failed")   

    # try:
    #     Preprocessor().train_preprocess()
    # except:
    #     print("Preprocessing Failed")

    # try:
    #     find_best_model()
    # except Exception as e:
    #     print(f"Training failed {e}")
    staus, message = PredictionPipeline().predict()
    print(message)