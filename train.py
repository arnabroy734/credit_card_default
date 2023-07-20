from validation.validation import TrainDataValidation

if __name__=="__main__":
    try:
        TrainDataValidation().validate()
    except:
        print("Validation faild")