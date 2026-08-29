# diabetes_proj
ENGE707 Data Engineering and Machine Learning Pipeline Project 

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load Raw Data

# Load the original dataset
df = pd.read_csv("../diabetes/diabetic_data.csv")

# Create a protected copy of the raw dataset
df_raw = df.copy()

# Create a separate dataset for cleaning
df_clean = df_raw.copy()

# INITIAL DATA INSPECTION

# Display the first five rows
df_raw.head()

# Display number of rows and columns
df_raw.shape

# Display information about columns and data types
df_raw.info(verbose=True)

# Check for standard missing values
df_raw.isnull().sum()

# IDENTIFY UNKNOWN VALUES

# Count values represented by "?"
question_mark_count = (df_raw == "?").sum()

# Calculate percentage of "?" values
question_mark_percent = (question_mark_count / len(df_raw)) * 100

# Create a summary table
question_mark_summary = pd.DataFrame({
    "Unknown (?)": question_mark_count,
    "Percentage": question_mark_percent
})

# Display columns containing "?" values
question_mark_summary[
    question_mark_summary["Unknown (?)"] > 0
].sort_values(
    "Percentage",
    ascending=False
)

# CHECK CATEGORICAL VARIABLES

# Examine race distribution
df_raw["race"].value_counts()

# Examine gender distribution
df_raw["gender"].value_counts()

# Examine age distribution
df_raw["age"].value_counts()

# Examine target variable distribution
df_raw["readmitted"].value_counts()

# CHECK FOR DUPLICATES

# Count duplicate rows
df_raw.duplicated().sum()

# SUMMARY STATISTICS

# Display summary statistics for numerical variables
df_raw.describe()

# CONVERT UNKNOWN VALUES TO MISSING VALUES

# Replace "?" with NaN in the clean dataset only
df_clean = df_clean.replace("?", np.nan)

# Check missing values after conversion
missing_count = df_clean.isnull().sum()

# Calculate percentage of missing values
missing_percent = (missing_count / len(df_clean)) * 100

# Create missing-value summary
missing_summary = pd.DataFrame({
    "Missing": missing_count,
    "Percentage": missing_percent
})

# Display columns containing missing values
missing_summary[
    missing_summary["Missing"] > 0
].sort_values(
    "Percentage",
    ascending=False
)

# Treat missing values as unknown 
df_clean["medical_specialty"] = df_clean["medical_specialty"].fillna("Unknown")
df_clean["payer_code"] = df_clean["payer_code"].fillna("Unknown")

# FEATURE RELEVANCE

from scipy.stats import chi2_contingency

# Test the relationship between categorical features and readmission
for column in ["medical_specialty", "payer_code"]:

contingency = pd.crosstab(
        df_clean[column],
        df_clean["readmitted"]
    )

chi2, p, dof, expected = chi2_contingency(contingency)

  print(f"{column}")
    print(f"Chi-square: {chi2:.2f}")
    print(f"p-value: {p:.4f}")

# Calculate Cramér's V to measure the strength of the relationship

def cramers_v(column, target):

  contingency = pd.crosstab(
        df_clean[column],
        df_clean[target]
    )

 chi2, p, dof, expected = chi2_contingency(contingency)

 n = contingency.sum().sum()
    min_dimension = min(contingency.shape) - 1

  v = np.sqrt((chi2 / n) / min_dimension)

 return chi2, p, v


for column in ["medical_specialty", "payer_code"]:

   chi2, p, v = cramers_v(
        column,
        "readmitted"
    )

 print(f"{column}")
    print(f"Chi-square: {chi2:.2f}")
    print(f"p-value: {p:.4f}")
    print(f"Cramér's V: {v:.4f}")

# OUTLIER IDENTIFICATION

# Identify numerical columns
numerical_columns = df_clean.select_dtypes(include="number").columns

# Calculate IQR boundaries for each numerical column
for column in numerical_columns:
    Q1 = df_clean[column].quantile(0.25)
    Q3 = df_clean[column].quantile(0.75)
    IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

outliers = df_clean[
        (df_clean[column] < lower_bound) |
        (df_clean[column] > upper_bound)
    ]

print(f"\n{column}")
    print(f"Q1: {Q1}")
    print(f"Q3: {Q3}")
    print(f"IQR: {IQR}")
    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")
    print(f"Number of potential outliers: {len(outliers)}")
