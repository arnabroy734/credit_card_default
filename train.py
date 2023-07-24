from validation.validation import TrainDataValidation
from preprocessing.preprocessing import Preprocessor
from training.tune_model import find_best_model

if __name__=="__main__":
    # try:
    #     TrainDataValidation().validate()
    # except:
    #     print("Validation faild")

    # try:
    #     Preprocessor().train_preprocess()
    # except:
    #     print("Preprocessing Failed")

    try:
        find_best_model()
    except Exception as e:
        print(f"Training failed {e}")