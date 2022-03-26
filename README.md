# Sampling and Modeling of Credit Risk Analysis

## Overview

### Purpose
The purpose of this analysis is to determine both the optimal sampling strategy and the optimal predictive model for identifying high risk credit potential.  This analysis includes six separate modeling and sampling strategies: random oversampling with logistic regression (ROLR), SMOTE oversampling with logistic regression (SOLR), cluster centroid undersampling with logistic regression (CURL), SMOTEENN combinatorial sampling with logistic regression (SCRL), random forest classifier (RFC), and adaboost easy ensemble classifier (EEC).  The metrics used for evaluation are a combination of balanced accuracy score paired with precision and recall values for the individual classes (low risk and high risk) as well as an average precision and recall.

## Results

### Logistic Regression with Different Sampling Methods

* ROLR - Balanced Accuracy Score: 0.64728 
|            | Low Risk | High Risk | Average |
|----------- |:--------:|:---------:|:-------:|
| Precision  |1.00      |0.01       |0.99     |
| Recall     |0.66      |0.63       |0.66     |
* SOLR - Balanced Accuracy Score: 0.65442
|            | Low Risk | High Risk | Average |
|----------- |:--------:|:---------:|:-------:|
| Precision  |1.00      |0.01       |0.99     |
| Recall     |0.69      |0.61       |0.69     |
* CURL - Balanced Accuracy Score: 0.54473
|            | Low Risk | High Risk | Average |
|----------- |:--------:|:---------:|:-------:|
| Precision  |1.00      |0.01       |0.99     |
| Recall     |0.40      |0.69       |0.40     |
* SCRL - Balanced Accuracy Score: 0.67646
|            | Low Risk | High Risk | Average |
|----------- |:--------:|:---------:|:-------:|
| Precision  |1.00      |0.01       |0.99     |
| Recall     |0.60      |0.75       |0.60     |

### Ensemble Classifiers with No Sampling Transformations

* RFC - Balanced Accuracy Score: 0.76656 
|            | Low Risk | High Risk | Average |
|----------- |:--------:|:---------:|:-------:|
| Precision  |1.00      |0.03       |0.99     |
| Recall     |0.89      |0.64       |0.89     |
* EEC - Balanced Accuracy Score: 0.93166
|            | Low Risk | High Risk | Average |
|----------- |:--------:|:---------:|:-------:|
| Precision  |1.00      |0.09       |0.99     |
| Recall     |0.92      |0.94       |0.92     |

## Summary

### Discussion of Results

All six of the models had a 1.00 precision on the low risk class, but that is purely a product of data's weighting.  There are 17104 records of low risk and 101 records of high risk, so it is quite easy to build a successful model in predicting low risk records.  A sampling strategy and model that performs best on **high risk** class specifically will be the optimal solution given the business directive.  As a control, a simple logistic regression model was created with no sample transformations to establish a baseline.   From that model 17,102 records were predicted to be low risk and were actually low risk, 2 were low risk but predicted to be high risk, 100 were high risk but labeled as low risk and only 1 record was high risk and labeled as high risk.  Thus, as a predictor of high risk credit the baseline model performed very poorly by only identifying 1 out of 101 high risk records for a resulting recall value of 0.01.  

##### Confusion Matrix for Logistic Regression with no Sampling (Control)
![Confusion Matrix of Control](https://github.com/fillinlater/control_cm.png "Confusion Matrix of Control")

Contextually, it makes sense to use high risk recall as the prime metric for sampling and model success.  This is because we want to identify as many high risk credit records as possible in order to mitigate impact to our bottom line.  The cost of a false positive (i.e. a low risk record being mistakenly flagged as high risk) are likely to be significantly less than the cost of a false negative (i.e. a high risk record being mistakenly flagged as low risk).  With that criteria, the best logistic regression model from the different sampling strategies was seen with SMOTEENN combinatorial sampling (SCRL).  It correctly labeled 76 of the 101 high risk records for a recall of 0.75.  This number does come at a cost however because the combinatorial sampling generated more false positives (6834) than both random oversampling (6308) and SMOTE oversampling(5217), although combinatorial sampling had fewer false positives than Cluster Centroid Undersampling (10340).

The first ensemble method, random forest classifier, had a high risk recall comparable to the logistic regression with oversampling but significantly reduced the number of false positives (1889).  Additionally, the random forest model allowed us to peek behind the curtain at the variables that are most influencing credit risk.  This extra information could benefit business decisions as well as provide clues for further feature selection, trimming, and dimensionality reduction to improve future models. 

![Feature Importance from Random Forest](https://github.com/fillinlater/feature_imp_rf.png "Feature Importance from Random Forest")

The final model, easy ensemble Adaboost, provided the best overall results.  The high risk recall score was 0.94, correctly identifying 93 of the 101 high risk records.  Additionally this model resulted in the fewest false positives (983).  Of the six models created on this dataset, easy ensemble is clearly the optimal solution.

### Final Recommendation

The recommendation would be to use the easy ensemble predictive model to classify high risk records.  This model has the highest recall of high risk records while providing the least amount of false positives, or said another way, low risk records mistakenly identified as high risk.  Further steps can be taken to improve the model.  The SMOTEENN combinatorial sampling has been shown to improve model performance and could be applied to the input data in data pre-processing.  Also, the feature importances derived from the random forest can be used to prune any unnecessary features or low importance features.  It would be a good idea to scale the features using scikit learn's StandardScaler or similar tools.  Finally, it can never hurt to gather more data and feed it to the model as more data makes for better models.