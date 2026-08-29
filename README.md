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


