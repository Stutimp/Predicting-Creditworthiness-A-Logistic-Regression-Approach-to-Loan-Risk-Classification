## credit-risk-classification

### Project on Supervised learning

### Overview 

This analysis aims to train and evaluate a model based on loan risk. I will use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. Here, I am provided a dataset of historical lending activity from a peer-to-peer lending company. This dataset comprises 77,536 rows (data points) and eight columns (features). The columns here are loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, total debts, and loan status.
Based on the provided dataset, using a machine learning model,  I will predict the loan status of the borrowers if those loans are from the healthy or high-risk loan credit category. Here, I am trying to identify the creditworthiness of the borrowers by building two machine learning models and testing both models to predict the loan status of the borrowers and pick the best model for the company.

### Generation of two Machine Learning Models, they are, Machine Learning Model 1 and Machine Learning Model 2, using Original data and resampled dataset separately:

 First, I loaded the provided dataset in a CSV form into jupyter notebook using pd.read_csv(), then after some exploratory analysis, I separated the data into labels and features, with 'variable and 'X' variable. I checked the balance of my target values. I split the dataset into X_train, X_test, y_train, and y_test to create a model, named as Machine Learning Model 1, train the model, and predict with 'X' data to predict the target values to compare the performance and accuracy of my model. Finally, I evaluated my model's performance by computing an accuracy score, generating a confusion matrix, and a classification report. 
In the analysis, I further resampled my training data with the RandomOverSampler module from the imbalance learn library , in order to create another model named as,Machine Learning Model 2 to evaluate the model's ability to detect the performance improvement of minority classes by balancing the dataset. 

## Outcomes

 ## **Machine Learning Model 1:**
Accuracy, Precision, and Recall scores of Machine learning model l predicting  both the `0` (healthy loan) and `1` (high-risk loan) labels are given as follows:

**Precision** 
- For 0 (healthy loan): The precision is 1.00, indicating that when the model predicts a loan as healthy, it is correct 100% of the time.
- For 1 (high-risk loan): The precision is 0.84, meaning that when the model predicts a loan as high-risk, it is correct 84% of the time.

**Recall**:
- For 0 (healthy loan): The recall is 0.99, which means the model successfully identifies 99% of all healthy loans.
- For 1 (high-risk loan): The recall is 0.94, indicating that the model correctly identifies 94% of all high-risk loans.

**F1-Score**:
- For 0 (healthy loan): The F1-score is 1.00, perfectly balancing precision and recall for healthy loan predictions.
- For 1 (high-risk loan): The F1-score is 0.89, which is quite high and indicates a good balance between precision and recall for high-risk loan predictions.

**Accuracy**:
- The overall accuracy of the model is 0.99, suggesting that the model correctly predicts the label (whether healthy or high-risk) for 99% of the loans in the dataset.

**Macro Average**:
- The macro average for precision, recall, and F1-score are 0.92, 0.97, and 0.94, respectively. These numbers suggest the model performs reasonably well across both categories, healthy and high-risk loans, without significant bias towards one class.

**Weighted Average**:
- The weighted averages for precision, recall, and F1-score are very high (0.99 for all), reflecting the model's overall accuracy considering the imbalance between the two classes (there are far more healthy loans than high-risk ones in the dataset).
In summary, the logistic regression model performs excellently in predicting healthy loans with almost perfect precision, recall, and F1-score. It also does a good job identifying high-risk loans, though there is a slight drop in precision compared to healthy loans. However, the high recall for high-risk loans is particularly noteworthy, as it suggests the model can catch most of the high-risk loans, which is crucial for risk management purposes. Overall, Model 1 effectively distinguishes between healthy and high-risk loans.

## **Machine Learning Model 2:**
Accuracy, Precision, and Recall scores of Machine Learning Model 2 predicting  both the `0` (healthy loan) and `1` (high-risk loan) labels are given as follows:

**Precision**:
- For 0 (healthy loan): The precision is 0.99, indicating that 99% of the loans predicted by the model are healthy.
- For 1 (high-risk loan): The precision is also 0.99, meaning that 99% of the loans predicted as high-risk by the model are indeed high-risk.

**Recall**
- For 0 (healthy loan): The recall is 0.99, meaning the model correctly identifies 99% of all healthy loans.
- For 1 (high-risk loan): The recall is also 0.99, indicating that the model correctly identifies 99% of all actual high-risk loans.

**F1-Score**
- For both classes: The F1-score, a harmonic mean of precision and recall, is 0.99. This suggests a balanced performance between precision and recall, indicating that the model is equally good at identifying both healthy and high-risk loans.

**Accuracy**
- The model's overall accuracy is 0.99, meaning it correctly predicts the loan status (healthy or high-risk) for 99% of the loans in the dataset.

**Macro Avg and Weighted Avg**:
The macro and weighted averages for precision, recall, and F1-score are 0.99. The macro average treats both classes equally, indicating strong performance across both classes regardless of their size. The weighted average accounts for class imbalance, which aligns with the macro average due to oversampling, making the classes equal in size.

The logistic regression model 2, after being trained on oversampled data, performs exceptionally well in predicting healthy and high-risk loans. The model demonstrates high precision and recall for both classes, which is particularly noteworthy given the initial class imbalance. Oversampling has effectively mitigated the imbalance issue, allowing the model to learn equally from both classes and thus perform well in both. This indicates a successful application of oversampling to improve model performance on an imbalanced dataset.

### Summary

In summary, the performance of a predictive model should always be evaluated in the context of the specific problem it is solving, the consequences of different types of errors, or the alignment of the model's evaluation metrics with the business or operational goals.
Overall, both models have both pros and cons; if I have to pick one model from both, I would instead pick model 2 considering all the factors mentioned above and also because we are doing credit risk analysis for a lending company to identify the creditworthiness of the borrower better for which Machine learning model 2 is more helpful and practical.


Thank you !


Regards,

Stuti Poudel