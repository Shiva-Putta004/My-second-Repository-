import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
# In a Kaggle environment, use:
# df = pd.read_csv("/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
# For local execution:
# df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# For demonstration, I'll use the sample data provided
data = """Age,Attrition,BusinessTravel,DailyRate,Department,DistanceFromHome,Education,EducationField,EmployeeCount,EmployeeNumber,EnvironmentSatisfaction,Gender,HourlyRate,JobInvolvement,JobLevel,JobRole,JobSatisfaction,MaritalStatus,MonthlyIncome,MonthlyRate,NumCompaniesWorked,Over18,OverTime,PercentSalaryHike,PerformanceRating,RelationshipSatisfaction,StandardHours,StockOptionLevel,TotalWorkingYears,TrainingTimesLastYear,WorkLifeBalance,YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion,YearsWithCurrManager
41,Yes,Travel_Rarely,1102,Sales,1,2,Life Sciences,1,1,2,Female,94,3,2,Sales Executive,4,Single,5993,19479,8,Y,Yes,11,3,1,80,0,8,0,1,6,4,0,5
49,No,Travel_Frequently,279,Research & Development,8,1,Life Sciences,1,2,3,Male,61,2,2,Research Scientist,2,Married,5130,24907,1,Y,No,23,4,4,80,1,10,3,3,10,7,1,7
37,Yes,Travel_Rarely,1373,Research & Development,2,2,Other,1,4,4,Male,92,2,1,Laboratory Technician,3,Single,2090,2396,6,Y,Yes,15,3,2,80,0,7,3,3,0,0,0,0
33,No,Travel_Frequently,1392,Research & Development,3,4,Life Sciences,1,5,4,Female,56,3,1,Research Scientist,3,Married,2909,23159,1,Y,Yes,11,3,3,80,0,8,3,3,8,7,3,0
27,No,Travel_Rarely,591,Research & Development,2,1,Medical,1,7,1,Male,40,3,1,Laboratory Technician,2,Married,3468,16632,9,Y,No,12,3,4,80,1,6,3,3,2,2,2,2
32,No,Travel_Frequently,1005,Research & Development,2,2,Life Sciences,1,8,4,Male,79,3,1,Laboratory Technician,4,Single,3068,11864,0,Y,No,13,3,3,80,0,8,2,2,7,7,3,6
59,No,Travel_Rarely,1324,Research & Development,3,3,Medical,1,10,3,Female,81,4,1,Laboratory Technician,1,Married,2670,9964,4,Y,Yes,20,4,1,80,3,12,3,2,1,0,0,0
30,No,Travel_Rarely,1358,Research & Development,24,1,Life Sciences,1,11,4,Male,67,3,1,Laboratory Technician,3,Divorced,2693,13335,1,Y,No,22,4,2,80,1,1,2,3,1,0,0,0
38,No,Travel_Frequently,216,Research & Development,23,3,Life Sciences,1,12,4,Male,44,2,3,Manufacturing Director,3,Single,9526,8787,0,Y,No,21,4,2,80,0,10,2,3,9,7,1,8"""

# Convert string data to a DataFrame
import io
df = pd.read_csv(io.StringIO(data))

# In a real implementation, you'd use the full dataset:
# df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
# print(f"Loaded dataset with {len(df)} employees")

# 1. Data Exploration
print("Data overview:")
print(f"Shape: {df.shape}")
print("\nColumns:", df.columns.tolist())
print("\nSample data:")
print(df.head(3))

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values:")
print(missing_values[missing_values > 0] if sum(missing_values) > 0 else "No missing values")

# Target variable distribution (Attrition)
print("\nAttrition distribution:")
print(df['Attrition'].value_counts())
print(f"Attrition rate: {df['Attrition'].value_counts(normalize=True)['Yes']:.2%}")

# PerformanceRating distribution
print("\nPerformance Rating distribution:")
print(df['PerformanceRating'].value_counts())

# 2. Feature Engineering
# Convert categorical 'Attrition' to binary
df['Attrition_Binary'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Calculate tenure ratio (career progression within company)
df['TenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)  # Adding 1 to avoid division by zero

# Calculate promotion velocity
df['PromotionVelocity'] = df['YearsAtCompany'] / (df['YearsSinceLastPromotion'] + 1)

# Calculate salary per experience level
df['SalaryPerExperience'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)

# Calculate job level to work years ratio
df['LevelToExperienceRatio'] = df['JobLevel'] / (df['TotalWorkingYears'] + 1)

# Calculate satisfaction composite score
df['SatisfactionComposite'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + 
                               df['WorkLifeBalance'] + df['RelationshipSatisfaction']) / 4

# Convert OverTime to binary
df['OverTime_Binary'] = df['OverTime'].map({'Yes': 1, 'No': 0})

# 3. Data Preprocessing
# Identify categorical and numerical columns
categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 
                    'JobRole', 'MaritalStatus', 'Over18']
numerical_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 
                  'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
                  'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
                  'NumCompaniesWorked', 'PercentSalaryHike', 'RelationshipSatisfaction',
                  'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                  'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                  'YearsSinceLastPromotion', 'YearsWithCurrManager',
                  'TenureRatio', 'PromotionVelocity', 'SalaryPerExperience',
                  'LevelToExperienceRatio', 'SatisfactionComposite', 'OverTime_Binary']

# Remove columns that won't be used in modeling
columns_to_exclude = ['Attrition', 'Attrition_Binary', 'EmployeeCount', 'EmployeeNumber', 'StandardHours', 'OverTime']

# 4. Explore Feature Relationships

# Correlation with Attrition
print("\nTop features correlated with Attrition:")
corr_with_attrition = df[numerical_cols + ['Attrition_Binary']].corr()['Attrition_Binary'].sort_values(ascending=False)
print(corr_with_attrition.head(10))

# Plot key relationships
plt.figure(figsize=(10, 6))
sns.heatmap(df[['OverTime_Binary', 'WorkLifeBalance', 'JobSatisfaction', 
                'EnvironmentSatisfaction', 'Attrition_Binary']].corr(), 
            annot=True, cmap='coolwarm')
plt.title('Key Correlations with Attrition')
plt.tight_layout()
plt.show()

# Overtime vs Attrition
plt.figure(figsize=(8, 5))
sns.countplot(x='OverTime', hue='Attrition', data=df)
plt.title('Overtime vs Attrition')
plt.show()

# Attrition by Department
plt.figure(figsize=(10, 5))
sns.countplot(x='Department', hue='Attrition', data=df)
plt.title('Attrition by Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Attrition Prediction Model
print("\n--- Building Attrition Prediction Model ---")

# Create the feature set X and target y
X = df.drop(columns_to_exclude + ['PerformanceRating'], axis=1)
y_attrition = df['Attrition_Binary']

# Split data for attrition model
X_train, X_test, y_train, y_test = train_test_split(X, y_attrition, test_size=0.2, random_state=42, stratify=y_attrition)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create attrition prediction pipeline
attrition_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Define hyperparameters for grid search
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
}

# Grid search for best hyperparameters (commented out to save computation time)
# grid_search = GridSearchCV(attrition_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
# grid_search.fit(X_train, y_train)
# print(f"Best parameters: {grid_search.best_params_}")
# best_model = grid_search.best_estimator_

# For demo purposes, we'll just use the pipeline with default parameters
attrition_pipeline.fit(X_train, y_train)
best_model = attrition_pipeline

# Evaluate attrition model
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\nAttrition Model Performance:")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Performance Rating Prediction Model
print("\n--- Building Performance Rating Prediction Model ---")

# Create target for performance prediction
y_performance = df['PerformanceRating']

# Split data for performance model
X_perf_train, X_perf_test, y_perf_train, y_perf_test = train_test_split(
    X, y_performance, test_size=0.2, random_state=42)

# Create performance rating prediction pipeline
performance_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Train the performance model
performance_pipeline.fit(X_perf_train, y_perf_train)

# Evaluate performance model
y_perf_pred = performance_pipeline.predict(X_perf_test)

print("\nPerformance Rating Model Results:")
print(f"Mean Squared Error: {mean_squared_error(y_perf_test, y_perf_pred):.4f}")
print(f"R² Score: {r2_score(y_perf_test, y_perf_pred):.4f}")

# 7. Feature Importance Analysis
# Get feature names after preprocessing
cat_feature_names = []
for i, col in enumerate(categorical_cols):
    # Get the categories for this column
    categories = best_model.named_steps['preprocessor'].transformers_[1][1].categories_[i]
    # Create feature names combining column name and category
    cat_feature_names.extend([f"{col}_{cat}" for cat in categories])

feature_names = numerical_cols + cat_feature_names

# Extract feature importances for attrition
importances = best_model.named_steps['classifier'].feature_importances_
if len(importances) < len(feature_names):
    # Adjust if feature names length doesn't match (this can happen with one-hot encoding)
    feature_names = feature_names[:len(importances)]

# Create DataFrame for feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot top 15 features for attrition prediction
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Top 15 Features for Attrition Prediction')
plt.tight_layout()
plt.show()

print("\nTop 10 features for attrition prediction:")
print(feature_importance.head(10))

# Extract feature importances for performance
perf_importances = performance_pipeline.named_steps['regressor'].feature_importances_
if len(perf_importances) < len(feature_names):
    feature_names_perf = feature_names[:len(perf_importances)]
else:
    feature_names_perf = feature_names

# Create DataFrame for performance feature importance
perf_feature_importance = pd.DataFrame({
    'Feature': feature_names_perf,
    'Importance': perf_importances
}).sort_values(by='Importance', ascending=False)

# Plot top 15 features for performance prediction
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=perf_feature_importance.head(15))
plt.title('Top 15 Features for Performance Rating Prediction')
plt.tight_layout()
plt.show()

print("\nTop 10 features for performance prediction:")
print(perf_feature_importance.head(10))

# 8. Prediction on Full Dataset
# Apply models to predict attrition risk and performance for all employees
df['Attrition_Risk'] = best_model.predict_proba(X)[:, 1]
df['Predicted_Performance'] = performance_pipeline.predict(X)

# 9. Identify High-Risk, High-Performing Employees
high_risk = df['Attrition_Risk'] > 0.5
high_performance = df['PerformanceRating'] >= 4

high_risk_high_performers = df[high_risk & high_performance]
print(f"\nNumber of high-risk, high-performing employees: {len(high_risk_high_performers)}")

# 10. Insights and Recommendations
print("\n--- Key Insights and Recommendations ---")

# Department-specific attrition rates
dept_attrition = df.groupby('Department')['Attrition_Binary'].mean().sort_values(ascending=False)
print("\nAttrition rate by department:")
print(dept_attrition)

# JobRole-specific attrition rates
role_attrition = df.groupby('JobRole')['Attrition_Binary'].mean().sort_values(ascending=False)
print("\nAttrition rate by job role:")
print(role_attrition.head())

# Analyze key factors for retained vs. departed employees
print("\nKey metrics comparison - Retained vs. Departed:")
comparison_metrics = ['JobSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction', 
                      'RelationshipSatisfaction', 'YearsAtCompany', 'MonthlyIncome']
                      
metrics_comparison = df.groupby('Attrition')[comparison_metrics].mean()
print(metrics_comparison)

# Output summary insights
print("\nKey Findings:")
print(f"1. Overall attrition rate: {df['Attrition_Binary'].mean():.2%}")
print(f"2. Department with highest attrition: {dept_attrition.index[0]} ({dept_attrition.iloc[0]:.2%})")
print(f"3. Job role with highest attrition: {role_attrition.index[0]} ({role_attrition.iloc[0]:.2%})")
print(f"4. Top attrition factor: {feature_importance['Feature'].iloc[0]}")
print(f"5. Top performance factor: {perf_feature_importance['Feature'].iloc[0]}")

print("\nRecommendations for HR:")
print("1. Focus retention efforts on employees with high attrition risk scores")
print("2. Create targeted interventions for high-risk, high-performing employees")
print(f"3. Implement department-specific retention strategies for {dept_attrition.index[0]}")
print(f"4. Address {feature_importance['Feature'].iloc[0]} as the primary factor driving attrition")
print("5. Develop performance enhancement programs focused on the top factors identified by the model")
print("6. Implement regular sentiment analysis to detect early warning signs of attrition")

# 11. Create a function for predicting attrition risk and performance for new employees
def predict_employee_outcomes(employee_data):
    """
    Predict attrition risk and performance for a new employee
    
    Parameters:
    employee_data (dict): Dictionary containing employee features
    
    Returns:
    dict: Predicted attrition risk and performance rating
    """
    # Convert employee data to DataFrame
    employee_df = pd.DataFrame([employee_data])
    
    # Ensure all required columns exist
    for col in X.columns:
        if col not in employee_df.columns:
            employee_df[col] = 0
    
    # Make predictions
    attrition_risk = best_model.predict_proba(employee_df[X.columns])[:, 1][0]
    performance = performance_pipeline.predict(employee_df[X.columns])[0]
    
    return {
        'attrition_risk': attrition_risk,
        'attrition_risk_percent': f"{attrition_risk:.1%}",
        'predicted_performance': performance,
        'retention_recommendation': 'High Risk' if attrition_risk > 0.5 else 'Medium Risk' if attrition_risk > 0.3 else 'Low Risk'
    }

# Example usage of the prediction function
sample_employee = {
    'Age': 35,
    'BusinessTravel': 'Travel_Frequently',
    'DailyRate': 1000,
    'Department': 'Research & Development',
    'DistanceFromHome': 10,
    'Education': 3,
    'EducationField': 'Life Sciences',
    'Gender': 'Male',
    'HourlyRate': 65,
    'JobInvolvement': 3,
    'JobLevel': 2,
    'JobRole': 'Research Scientist',
    'JobSatisfaction': 2,
    'MaritalStatus': 'Single',
    'MonthlyIncome': 5000,
    'MonthlyRate': 20000,
    'NumCompaniesWorked': 3,
    'Over18': 'Y',
    'OverTime_Binary': 1,
    'PercentSalaryHike': 15,
    'RelationshipSatisfaction': 3,
    'StockOptionLevel': 0,
    'TotalWorkingYears': 10,
    'TrainingTimesLastYear': 2,
    'WorkLifeBalance': 2,
    'YearsAtCompany': 5,
    'YearsInCurrentRole': 3,
    'YearsSinceLastPromotion': 2,
    'YearsWithCurrManager': 3,
    'TenureRatio': 0.5,
    'PromotionVelocity': 2.5,
    'SalaryPerExperience': 500,
    'LevelToExperienceRatio': 0.2,
    'SatisfactionComposite': 2.5
}

print("\nSample Prediction for a New Employee:")
prediction_result = predict_employee_outcomes(sample_employee)
print(prediction_result)
