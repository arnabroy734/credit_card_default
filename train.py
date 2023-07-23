from validation.validation import TrainDataValidation
from preprocessing.preprocessing import Preprocessor

if __name__=="__main__":
    # try:
    #     TrainDataValidation().validate()
    # except:
    #     print("Validation faild")

    try:
        Preprocessor().preprocess()
    except:
        print("Preprocessing Failed")