# Lead Scoring Model for Conversion Prediction
## Overview
This repository focuses on building a robust Lead Scoring Model to prioritize leads for customer conversion. By identifying the most promising leads, companies can optimize their sales strategies and resources for higher conversion rates. The project employs Python and machine learning algorithms for data analysis, feature engineering, and predictive modeling.

## Installation
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
## Data Preprocessing
The first step involves data cleaning and transformation:

- Handling missing values by imputation or removal.
- Dropping columns with high percentages of missing values.
- Encoding categorical variables.
## Exploratory Data Analysis (EDA) and Feature Engineering
EDA is performed to understand the data and identify key features that impact customer conversion. And new features are engineered to better capture the underlying patterns in the data. For example, the average time spent per visit is calculated. Categorical variables like 'Do Not Email' and 'Do Not Call' are transformed into binary format.
## Model Training and Evaluation
Three machine learning models are trained on the preprocessed data:
- Logistic Regression
- Random Forest Classifier
- Support Vector Classifier (SVC)
## Results
The performance metrics for the trained models are as follows:

#### Logistic Regression: ~84.8% accuracy
#### Random Forest Classifier: ~93.2% accuracy
#### Support Vector Classifier (SVC): ~88.0% accuracy
## Usage 
Clone this GitHub repository.
Install the required packages.
Add your dataset in the input directory.
Run the Python script or Jupyter Notebook.
