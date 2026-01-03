# Experimentation Log
This document notes experiments conducted to compare logistic regression and a linear support vector machine for use on the phishing email dataset. These numbers are used to justify the final model choice for this detector. 

## Experimental Setup
- Dataset split: training 80%, testing 20%
- Preprocessing: text normalisation TF-IDF (lowercasing, whitespace removal) vectorisation
- Evaluation metrics: precision, recall and F1-score

## Class Imbalance Analysis 
| Label  | Total Emails | Percentage |
| ------ | ------------ | ---------- |
| 1      | 42891        | 51.998%    |       
| 0      | 39595        | 48.002%    |

## Logistic Regression
Testing accuracy: 98.46%

| Label  | Precision | Recall | F1-Score | 
| ------ | ----------| ------ | -------- |
| 0      | 0.99      | 0.98   | 0.98     |
| 1      | 0.98      | 0.99   | 0.99     |

## Support Vector Machine
Hyperparameters:
* C = 10
* kernel = linear

Testing accuracy: 99.11%

| Label  | Precision | Recall | F1-Score | 
| ------ | ----------| ------ | -------- |
| 0      | 0.99      | 0.99   | 0.99     |
| 1      | 0.99      | 0.99   | 0.99     |

## Final Decision
From the experimentation results, a linear SVM was chosen for the system's model, as it achieed the highest test accuracy and consistently strong precision and recall across both classes. 

## Limitations and Future Work
The following factors are main points for model improvement based on these experimentation statistics:
* Training/testing split: the data is currently split 80/20, but could be improved by using cross-validation for more robust evaluation
* Non-linearity: only linear models were used to analyse this dataset. A model, such as a neural network, could be used to try and identify non-linear relatioships within the data.
