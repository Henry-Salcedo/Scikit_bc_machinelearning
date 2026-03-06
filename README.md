# Scikit_bc_machinelearning

## Puropse
  Generating 3 machine learning classification models that use standard data division for the breast cancer dataset in sklearn. Then, determine if one of the 3 is better than the others, and if not, determine the pros/cons of the models.

This project's aim was to focus on “Scikit learn” to create machine learning models.
The project reads the breast cancer dataset that is within “Scikit learn” that contains multiple columns.
The goal is to generate machine learning models and identify which one demonstrates the highest predictive accuracy and effectiveness.
  
Key analyses include:
- Extracted the data (X) and target (y).
- Split the data into 80% training and 20% test_size.
- Created a Logistic regression, K-nearest neighbor, and Random forest models.
- Tested their prediction by reviewing accuracy, precision, recall, and F1. Then printing the findings.
- Wrote a brief paragraph defining which model was the best performing.

## Class Design
The project can be classified into 5 different areas:

- Creating the data split
- Setting up models
- Prediction scores
- Print out results (accuracy, precision, etc.)
- Reviewed results for the best-performing model

## Class Attributes and Methods

### Data Split

**Attributes:**
- b_cancer.data - Feature data from the dataset
- b_cancer.target - Labels from dataset
- test_size=0.2 - 20% for test set
- random_state=32 - Seed for reproducibility
  
**Methods:**
- datasets.load_breast_cancer() - Loads the breast cancer dataset
- model_selection.train_test_split() - Splits data into train/test sets

---
### Setting Up Models

**Attributes:**
- LogisticRegression(random_state=43, solver=‘liblinear’)
- KNeighborsClassifier(n_neighbors=5)
- RandomForestClassifier(n_estimators=200, random_state=28)

**Methods:**
- .fit(x_train, y_train) - Trains each model on training data

---
### Predicting Scores
**Attributes:**
- reg_test_pred - Array of predictions from the Logistic Regression model
- knn_test_pred - Array of predictions from the K-Nearest Neighbors model
- rf_test_pred - Array of predictions from Random Forest model
  
**Methods:**
-.predict(x_test) - Generates predictions on test data

---
### Results

**Attributes:**
- log_reg_accuracy, log_reg_precision, log_reg_recall
- knn_accuracy, knn_precision, knn_recall
- rf_accuracy, rf_precision, rf_recall
- Calculate F1 scores for each model after initial metrics.
  
**Methods:**
- accuracy_score(y_test, predictions) - Calculates accuracy
- precision_score(y_test, predictions) - Calculates precision
- recall_score(y_test, predictions) - Calculates recall
- Printed calculated scores up to here.
- f1_score(y_test, predictions) - Calculates F1 score
- Then printed f1_scores after being calculated.

---
## Limitations
- I limited myself to what I learn in a class related to the project (e.g., not performing any scaled features or cross-validation).
- In addition to the limit of class-related, I also limited myself from generating models since such an area has not been discussed (Once more, such a thing is possible now, however, these limits are to test what the library can do).
- Focused solely on using sklearn to establish a better understanding of its usage. 

## Post Implementation/Thoughts
These are some of the implementation/Thoughts post methods to highlight the areas that I improved and/or discussed upon the initial methods:

- Messed with the n-neigbors = 5 - After reviewing other areas like 3, or 13, they did little to improve more than 5, if I could uses something like StandardScaler , and gridsearch, then those would greatly help determine the best possible k value.
- RandomForestClassifier(n_estimators=200) - Change from 100 to 200 and saw improvements to the accuracy, precision, and F1 scores.
