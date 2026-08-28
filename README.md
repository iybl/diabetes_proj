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
