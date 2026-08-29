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

# Calculate chi-square and Cramér's V
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


# Test the relationship between categorical features and readmission
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

# INVESTIGATE POTENTIAL OUTLIERS

# Examine upper-bound outlier values

for column in [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]:

Q1 = df_clean[column].quantile(0.25)
Q3 = df_clean[column].quantile(0.75)
    IQR = Q3 - Q1

upper_bound = Q3 + 1.5 * IQR

 print(f"\n{column}")
    print(
        df_clean[
            df_clean[column] > upper_bound
        ][column]
        .value_counts()
        .sort_index()
    )

# HANDLE INVALID AND MISSING CATEGORICAL VALUES

# Check gender values including invalid values
print("Gender values:")
print(df_clean["gender"].value_counts(dropna=False))

# Replace invalid gender values with Unknown
df_clean["gender"] = df_clean["gender"].replace(
    "Unknown/Invalid",
    "Unknown"
)

# Check race values including missing values
print("\nRace values:")
print(df_clean["race"].value_counts(dropna=False))

# Replace missing race values with Unknown
df_clean["race"] = df_clean["race"].fillna("Unknown")

# OUTLIER DECISION

# Review the maximum values of numerical features
print(df_clean[numerical_columns].max())

# REMOVE FEATURES WITH HIGH VOLUME MISSING OR WEAK RELEVANCE

df_clean = df_clean.drop(columns=[
"weight",
"max_glu_serum",
"A1Cresult",
"medical_specialty",
"payer_code"
])

# CHECK CLEANED DATASET

# Check the number of rows and columns
print("Dataset shape:", df_clean.shape)

# Check remaining missing values
print("\nRemaining missing values:")
print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

# Display remaining columns
print("\nRemaining columns:")
print(df_clean.columns.tolist())

