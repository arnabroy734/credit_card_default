
# Credit Card Defaulter Prediction

Any bank or financial institution issuing credit cards follows up with card holders to get payments of next month. Suppose a bank issued credit cards to 50000 customers, so following up with all of them every month becomes time consuming. 

Machine learning can be used to make the job easy for any bank issuing credit cards. Potential defaulters can be predicted using historical data and list of customers who need follow up for payment can be narrowed down. **In this project an online system is developed which takes historical data as input and predicts customers who are going to default next month.**


## Dataset Used
This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005. The dataset is available  [**here**](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
## Approach to Solve the Problem
### Data Preprocessing
In any machine leaning problem raw data needs preprocessing e.g., missing value imputation, feature transformation etc. In this particular problem statement two preprocessing steps are used - 

- **Replacement** of additional categories in some of the features. For example in **Education** there are only categories 1, 2, 3 and 4 where 4 is for others as per data description. But the dataset has total 7 categories - so we will replace all categories (apart from 1, 2 and 3) as 4.

- **Encoding** of features (Sex, Education, Marriage etc.) which are categorical in nature. **Probability Ratio Encoding** is used here.

### Model Training and Selection
The dataset is imbalanced in nature i.e., only 11% of total records have class level 1 (defaulter). To handle such imbalanced dataset following steps are followed during model training - 

- The data is split into **train, test and cv**.
- The train data is upsampled using [**minority oversampling technique**](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- Model is built using oversampled data and hyperparameters tuned by measuring performance on this model using cv data. (**cv data is not upsampled**)
 - Best model is selected based on cv score.

### Performance Metrics
Performance metrics depend of businesss use case in any machine learning problem and set by domain expert. Consider the follwoing scenarios - 

- By **reducing false negatives** bank can ensure that no potential defaulters should be left unattended.
 - By **reducing false positives** bank can optimise time and resource required to follow up with customers.
As we don't have the required domain knowledge we set **F1 score** as performance metrics.


## Project Architecture
The project is developed as an online web application where user can **upload** excel file contaning historical data of customers, run the **prediction** after successful upload and then **download** the excel file. **The project also has option to re-train the model but this feature is not exposed as API due to resource limitation on server side.**

### Architecture of Model Training Process

![image](https://github.com/arnabroy734/credit_card_default/assets/86049035/72a71c69-1c11-4ae6-9bdf-e6a4860e52cc)

### Architecture of Web Application for Prediction

![image](https://github.com/arnabroy734/credit_card_default/assets/86049035/2c755b7d-f65c-4367-b12e-4569a776e9f4)



