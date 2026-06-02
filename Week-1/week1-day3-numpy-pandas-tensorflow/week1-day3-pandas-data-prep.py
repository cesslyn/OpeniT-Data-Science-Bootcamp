# Week 1 Day 3 - Pandas Data Preparation 

import numpy as np 
import pandas as pd 

# Sample Dataset
df = pd.DataFrame({ 
"customer_id": [101, 102, 103, 104, 105, 106, 107, 108], 
"region": ["North", "South", "North", "East", "West", "South", "East", 
"West"], 
"plan_type": ["Basic", "Premium", "Basic", "Premium", "Basic", "Premium", 
"Basic", "Premium"], 
"monthly_fee": [29.99, 59.99, 29.99, 59.99, 29.99, None, 29.99, 59.99], 
"usage_hours": [12, 45, 18, 50, 9, 38, None, 41], 
"support_tickets": [1, 3, 0, 4, 2, 1, 0, 2], 
"churned": [0, 1, 0, 1, 0, 0, 0, 1] 
}) 

# Inspecting DataFrame
print(df.head()) 
print(df.tail()) 
df.info() 
print(df.describe(include="all")) 
print(df.isna().sum()) 

print("\n(1) How many rows and columns are in the dataset? ")
print("Answer: There are 8 rows and 7 columns in the dataset.")
print("\n(2) Which columns are numeric? ")
print("Answer: The numeric columns are 'monthly_fee', 'usage_hours', 'support_tickets', and 'churned'.")
print("\n(3) Which columns are categorical? ")
print("Answer: The categorical columns are 'customer_id', 'region', and 'plan_type'.")
print("\n(4) Which columns contain missing values? ")
print("Answer: The columns 'monthly_fee' and 'usage_hours' contain missing values.")
print("\n(5) Which column appears to be the target variable?")
print("Answer: The 'churned' column appears to be the target variable, as it indicates whether a customer has churned (1) or not (0).")

# Cleaning Missing Values
clean_df = df.copy() # working copy

# Filling missing numeric values using the median
numeric_cols = clean_df.select_dtypes(include=["number"]).columns 
clean_df[numeric_cols] = clean_df[numeric_cols].fillna(clean_df[numeric_cols].median()) 
clean_df[numeric_cols].fillna(clean_df[numeric_cols].median()) 

# Filling missing text values using unknown
text_cols = clean_df.select_dtypes(include=["object"]).columns 
clean_df[text_cols] = clean_df[text_cols].fillna("Unknown") 

print(clean_df.isna().sum()) 

# Validate missing data types and categories
print(clean_df.dtypes)

# Review category values
print(clean_df["region"].value_counts()) 
print(clean_df["plan_type"].value_counts())

# Standardizing category spelling issues if it exist
clean_df["region"] = clean_df["region"].str.strip().str.title() 
clean_df["plan_type"] = clean_df["plan_type"].str.strip().str.title() 

# Summary Tables
# Plan type
plan_summary = clean_df.groupby("plan_type").agg( 
customers=("customer_id", "count"), 
avg_monthly_fee=("monthly_fee", "mean"), 
avg_usage_hours=("usage_hours", "mean"), 
churn_rate=("churned", "mean") 
) 
print(plan_summary)

# Region
region_summary = clean_df.groupby("region").agg( 
customers=("customer_id", "count"), 
total_support_tickets=("support_tickets", "sum"), 
churn_rate=("churned", "mean") 
) 
print(region_summary) 

# Feartures for TensorFlow
feature_df = clean_df[["region", "plan_type", "monthly_fee", "usage_hours", 
"support_tickets"]] 
target = clean_df["churned"]

# One-hot encode for categorical columns
feature_df = pd.get_dummies(feature_df, columns=["region", "plan_type"], 
drop_first=False)

print(feature_df.dtypes) 

training_df = feature_df.copy() 
training_df["churned"] = target 
print(training_df.head()) 

training_df.to_csv("week1-day3-cleaned-training-data.csv", index=False) 

check_df = pd.read_csv("week1-day3-cleaned-training-data.csv") 
print(check_df.head()) 