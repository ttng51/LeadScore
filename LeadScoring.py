import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

#read dataset
lead_scoring_data = pd.read_csv('../input/lead-scoring-dataset/Lead Scoring.csv')

##Data Cleaning
# Checking for missing values in each column
missing_values = lead_scoring_data.isnull().sum().sort_values(ascending=False)
missing_values_percentage = (lead_scoring_data.isnull().sum() / lead_scoring_data.shape[0] * 100).sort_values(ascending=False)

# Combining the missing values and missing values percentage into a DataFrame
missing_info = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_values_percentage})
missing_info[missing_info['Missing Values'] > 0]

# Dropping columns with more than 40% missing values
columns_to_drop = missing_info[missing_info['Percentage'] > 40].index
lead_scoring_data.drop(columns=columns_to_drop, inplace=True)

# Imputing numerical columns with the median
numerical_cols = lead_scoring_data.select_dtypes(include=['float64']).columns
lead_scoring_data[numerical_cols] = lead_scoring_data[numerical_cols].apply(lambda x: x.fillna(x.median()), axis=0)

# Imputing categorical columns with 'Unknown' or mode
categorical_cols = lead_scoring_data.select_dtypes(include=['object']).columns
lead_scoring_data[categorical_cols] = lead_scoring_data[categorical_cols].apply(lambda x: x.fillna('Unknown'), axis=0)

# Checking if there are any more missing values
remaining_missing_values = lead_scoring_data.isnull().sum().max()

##Exploratory Data Analysis (EDA)
###The distribution of the target variable, 'Converted'
###The relationship between features and the target variable

# Checking the distribution of the target variable 'Converted'
sns.set(style="white")
plt.figure(figsize=(6, 4))
sns.countplot(x='Converted', data=lead_scoring_data)
plt.title('Distribution of Target Variable (Converted)')
plt.xlabel('Converted')
plt.ylabel('Count')
plt.show()
##It seems like we have a somewhat balanced dataset in terms of conversion rates, which is good for model training.

# List of some key features to examine
key_features = ['Lead Origin', 'Lead Source', 'Total Time Spent on Website', 'TotalVisits', 'Last Activity']

# Plotting the relationship
for feature in key_features:
    plt.figure(figsize=(12, 4))
    if lead_scoring_data[feature].dtype == 'object':
        sns.countplot(x=feature, hue='Converted', data=lead_scoring_data)
        plt.xticks(rotation=45)
    else:
        sns.boxplot(x='Converted', y=feature, data=lead_scoring_data)
    plt.title(f'Relationship between {feature} and Conversion')
    plt.show()


###Feature Engineering
# Creating a new feature that captures the average time spent per visit
lead_scoring_data['Avg_Time_Per_Visit'] = lead_scoring_data['Total Time Spent on Website'] / (lead_scoring_data['TotalVisits'] + 1)

# Converting 'Do Not Email' and 'Do Not Call' to binary variables (Yes: 1, No: 0)
lead_scoring_data['Do Not Email'] = lead_scoring_data['Do Not Email'].map({'Yes': 1, 'No': 0})
lead_scoring_data['Do Not Call'] = lead_scoring_data['Do Not Call'].map({'Yes': 1, 'No': 0})

# Encoding categorical variables
label_encoders = {}
for col in lead_scoring_data.select_dtypes(include=['object']).columns:
    if col != 'Prospect ID':  # Exclude the 'Prospect ID' as it's an identifier
        le = LabelEncoder()
        lead_scoring_data[col] = le.fit_transform(lead_scoring_data[col])
        label_encoders[col] = le

# Standardizing the numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(lead_scoring_data.drop(['Prospect ID', 'Converted'], axis=1))

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, lead_scoring_data['Converted'], test_size=0.2, random_state=42
)

# Initialize the models again to ensure they are defined
log_reg_model = LogisticRegression(random_state=42)
random_forest_model = RandomForestClassifier(random_state=42)
svc_model = SVC(probability=True, random_state=42)

# Train the Random Forest model
random_forest_model.fit(X_train, y_train)

# Train the Support Vector Classifier model
svc_model.fit(X_train, y_train)

# Train the model
log_reg_model.fit(X_train, y_train)
# Making predictions
y_pred_log_reg = log_reg_model.predict(X_test)
y_pred_rf = random_forest_model.predict(X_test)
y_pred_svc = svc_model.predict(X_test)

# Evaluate the performance of all models
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_svc = accuracy_score(y_test, y_pred_svc)

classification_log_rf = classification_report(y_test, y_pred_log_reg)
classification_rf = classification_report(y_test, y_pred_rf)
classification_rep_svc = classification_report(y_test, y_pred_svc)

#Accuracy

#Logistic Regression: ~84.8%
#Random Forest: ~93.2%
#Support Vector Classifier (SVC): ~88.0%

accuracy_log_reg, accuracy_rf, accuracy_svc, classification_log_rf, classification_rf, classification_rep_svc

# Using the trained Random Forest model to predict the probability of conversion for the test data
y_pred_prob_rf = random_forest_model.predict_proba(X_test)[:, 1]

# Converting the probabilities to lead scores in the range of 0-100
lead_scores_rf = (y_pred_prob_rf * 100).astype(int)

# Creating a DataFrame to display actual conversion status and the corresponding lead score using Random Forest model
lead_score_rf_df = pd.DataFrame({'Actual Conversion': y_test, 'Lead Score (Random Forest)': lead_scores_rf})

# Displaying some sample lead scores along with actual conversion status
lead_score_rf_df.sample(20)
