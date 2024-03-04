# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

1) **Explain the purpose of the analysis.**

Answer: The purpose of this analysis is to train and evaluate a model based on loan risk. We will use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. 


2) **Explain what financial information the data was on and what you needed to predict.**

Answer: Here, we are provided a dataset of historical lending activity from a peer-to-peer lending company. This dataset comprises 77,536 rows (data points) and eight columns (features). The columns here are loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, total debts, and loan status.
Based on the provided dataset, using a machine learning model, we are asked to predict the loan status of the borrowers if those loans are from the healthy or high-risk loan credit category. Here, we are trying to identify the creditworthiness of the borrowers by building a machine learning model and testing the model to predict the loan status of the borrowers.

3) **Provide basic information about the variables you were trying to predict (e.g., `value_counts`).**

Answer: Some of the important variables used in the analysis are as follows:

- **'y' and 'X' variable:** Here 'y' variable is a single series of the target variable (loan_status), whereas the X variable is a complete data frame after dropping the loan_status series, which we are going to predict with our machine learning model.

- **value_counts:** This is called on series object 'y' to return the counts of unique values in the series. It is also an effective way to assess the balance of classes in a target variable ('y'). For example, the target variable ('y') in our case is loan_status, and we are identifying, out of all values, how many belong to healthy loans(0)  and how many belong to high-risk loans (1).

  - **classifier and testing_predictions:** classifier is a machine learning model that we created, which we will fit in with our training data in order to train the model and finally predict our testing data to predict our final desired outcomes, i.e., loan status, which is saved in the name of testing predictions in our code. 

  -**Score, accuracy_score, and test_matrix:** These are all separate evaluation scores of our trained model. We compute these scores to understand our trained model's accuracy, competencies, and performance.
  
  
4) **Describe the stages of the machine learning process you went through as part of this analysis.**

Answer: First, I loaded the provided dataset in a CSV form into jupyter notebook using pd.read_csv(), then after some exploratory analysis, I separated the data into labels and features, with 'variable and 'X' variable. I checked the balance of my target values. I split the dataset into X_train, X_test, y_train, and y_test to create a model, train the model, and predict with 'X' data to predict the target values to compare the performance and accuracy of our model. Finally, we evaluated our model's performance by computing an accuracy score, generating a confusion matrix, and a classification report. 
In the analysis, we also further resampled our training data with the RandomOverSampler module from the imbalance learn library to evaluate the model's ability to detect the performance improvement of minority classes by balancing the dataset. 

5) **Briefly touch on any methods you used (e.g., `LogisticRegression` or any resampling method).**

Answer: In the analysis, we used the logistic regression on the original data first and then again employed a RandomOverSampler from the imbalanced-learn library to resample data before using logistic regression to improve the model's accuracy potentially and using RandomOverSampler before logistic regression can improve the model's ability to detect the minority class by balancing the dataset, which is beneficial for imbalanced datasets. However, applying proper cross-validation techniques and evaluating the model using appropriate metrics is essential to ensure that the model generalizes well to new data and is balanced.

6) ## Results

**Using bulleted lists, describe all machine learning models' balanced accuracy scores and the precision and recall scores.**

Answer: **Machine Learning Model 1:**
Accuracy, Precision, and Recall scores of Machine learning model l predicting  both the `0` (healthy loan) and `1` (high-risk loan) labels are given as follows:

**Precision** 
- For 0 (healthy loan): The precision is 1.00, indicating that when the model predicts a loan as healthy, it is correct 100% of the time.
- For 1 (high-risk loan): The precision is 0.84, meaning that when the model predicts a loan as high-risk, it is correct 84% of the time.

**Recall**:
- For 0 (healthy loan): The recall is 0.99, which means the model successfully identifies 99% of all healthy loans.
- For 1 (high-risk loan): The recall is 0.94, indicating that the model correctly identifies 94% of all high-risk loans.

**F1-Score**:
- For 0 (healthy loan): The F1-score is 1.00, which perfectly balances precision and recall for healthy loan predictions.
- For 1 (high-risk loan): The F1-score is 0.89, which is quite high and indicates a good balance between precision and recall for high-risk loan predictions.

**Accuracy**:
- The overall accuracy of the model is 0.99, suggesting that the model correctly predicts the label (whether healthy or high-risk) for 99% of the loans in the dataset.

**Macro Average**:
- The macro average for precision, recall, and F1-score are 0.92, 0.97, and 0.94, respectively. These numbers suggest the model performs reasonably well across both categories, healthy and high-risk loans, without significant bias towards one class.

**Weighted Average**:
- The weighted averages for precision, recall, and F1-score are very high (0.99 for all), reflecting the model's overall accuracy considering the imbalance between the two classes (there are far more healthy loans than high-risk ones in the dataset).
In summary, the logistic regression model performs excellently in predicting healthy loans with almost perfect precision, recall, and F1-score. It also does a good job identifying high-risk loans, though there is a slight drop in precision compared to healthy loans. However, the high recall for high-risk loans is particularly noteworthy, as it suggests the model can catch most of the high-risk loans, which is crucial for risk management purposes. Overall, Model 1 effectively distinguishes between healthy and high-risk loans.

**Machine Learning Model 2:**
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

## Summary

7) Summarize the results of the machine learning models and include a recommendation on the model to use, if any. For example:
**Which one seems to perform best? How do you know it performs best?**

Answer: To determine which model performs best, we must consider several aspects of the classification reports: precision, recall, f1-score, and the problem context, particularly the balance of the classes if applicable. A brief explanations are given below:

**In the original dataset classification report**, precision and recall indicate excellent performance for the healthy loan class ('0'). Whereas for the minority class, precision and recall scores are significantly lower in class '1', i.e, high-risk loan class, compared to class '0', i.e, healthy loan class. Also, the macro-average and weighted-average scores are good; however, it indicates the model performs well overall but could be improved in handling the minority class.
**Whereas in the resampled dataset**, for both categories, healthy loans and high-risk loans, precision, recall, and f1-scores, all are identical and very high.
Here, not only accuracy for both categories but macro-average and weighted-average scores are also all extremely high and equal, indicating this model performs excellently across both classes.

Overall, the resampled dataset model (machine learning model 2) achieves a nearly perfect balance in performance between both classes, indicating it has effectively overcome the class imbalance issue in the original dataset model (machine learning model 1). In addition, predictions of  1's (high-risk loans) are significantly high, as 0.99 in machine learning model 2, which is a significant factor for any financial institution, particularly in our case, a lending company considering potential consequences any misclassifications can cause those. Hence, overall, machine learning model 2 performs best in this scenario.


8) **Does performance depend on the problem we are trying to solve? (For example, is predicting the `1s` or the `0s` more important? )**

Answer: In this case, predicting the '1s is more important than the '0s. Because in such scenarios where missing a high-risk loan could lead to significant financial loss, it is crucial to have a high recall for the '1' (healthy loan) class. This means the model needs to correctly identify as many high-risk loans as possible, even if it results in some false positives (healthy loans incorrectly classified as high-risk).
The cost of a false negative (failing to identify a high-risk loan) is typically much higher than that of a false positive (incorrectly identifying a loan as high-risk). A false negative could mean extending credit to a borrower likely to default, leading to direct financial losses.
In summary, the performance of a predictive model should always be evaluated in the context of the specific problem it is solving, the consequences of different types of errors, or the alignment of the model's evaluation metrics with the business or operational goals.

9) **If you do not recommend any of the models, please justify your reasoning.**

Answer: Oversampling, particularly methods like Random OverSampling, can be valuable for addressing class imbalance in machine learning datasets. However, it has several risks and potential drawbacks that can affect model performance and generalization. Some of them are the following:
  1) Overfitting: By replicating the minority class instances, oversampling can lead the model to overfit the training data. This means the model may perform exceptionally well on the training data but poorly on unseen data because it has memorized the minority class instances rather than learning to generalize from them.
  2) Increased Computational Cost: Processing a larger dataset, which results from oversampling, requires more computational resources and time. This can be particularly challenging with large datasets or in situations with limited computational resources.

Overall, both models have both pros and cons; if I have to pick one model from both, I would instead pick model 2 considering all the factors mentioned above and also because we are doing credit risk analysis for a lending company to identify the creditworthiness of the borrower better for which Machine learning model 2 is more helpful and practical.

