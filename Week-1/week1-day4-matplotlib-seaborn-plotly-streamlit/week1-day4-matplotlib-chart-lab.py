# Week 1 Day 4 - Matplotlib Chart Lab 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Load and Create Dataset
df = pd.DataFrame({ 
"month": pd.date_range("2025-01-01", periods=8, freq="MS"), 
"region": ["North", "South", "North", "East", "West", "South", "East", "West"], 
"product_category": ["Software", "Hardware", "Software", "Services", "Software", 
"Hardware", "Services", "Software"], 
"revenue": [12000, 9500, 14200, 11000, 13500, 10200, 15100, 16400], 
"units_sold": [120, 85, 140, 95, 130, 90, 150, 160], 
"customer_satisfaction": [4.1, 3.8, 4.3, 4.0, 4.2, 3.9, 4.4, 4.5] 
}) 


# Inspect the dataset
print(df.head()) 
print(df.info()) 
print(df.describe(include="all")) 

# Clean and Prepare Chart Data
clean_df = df.copy() 
clean_df["month"] = pd.to_datetime(clean_df["month"]) 
clean_df = clean_df.sort_values("month") 

# Check for missing values
print(clean_df.isna().sum())

# Fill missing numeric values using the median
numeric_cols = clean_df.select_dtypes(include=["number"]).columns 
clean_df[numeric_cols] = clean_df[numeric_cols].fillna(clean_df[numeric_cols].median())
clean_df[numeric_cols].fillna(clean_df[numeric_cols].median())

# Standardize text categories
text_cols = clean_df.select_dtypes(include=["object"]).columns 
for col in text_cols: 
    clean_df[col] = clean_df[col].str.strip().str.title() 

clean_df.to_csv("week1-day4-cleaned-visualization-data.csv", index=False)

# Line chart
monthly_revenue = clean_df.groupby("month", as_index=False)["revenue"].sum() 
 
plt.figure(figsize=(8, 4)) 
plt.plot(monthly_revenue["month"], monthly_revenue["revenue"], marker="o") 
plt.title("Monthly Revenue Trend") 
plt.xlabel("Month") 
plt.ylabel("Revenue") 
plt.xticks(rotation=45) 
plt.tight_layout() 
plt.savefig("week1-day4-matplotlib-sales-trend.png") 
plt.show()

print("\n(1) What does the line chart show about revenue over time? ")
print("Answer: The line chart shows that revenue has generally increased over the months, with some fluctuations. There is a noticeable upward trend, indicating growth in revenue over time.")
print("\n(2) Which month has the highest revenue? ")
print("Answer: The month with the highest revenue is August, as indicated by the peak in the line chart.")
print("\nWhy is a line chart appropriate for monthly data? ")
print("Answer: A line chart is appropriate for monthly data because it effectively shows trends and patterns over time. It allows us to easily visualize how revenue changes from month to month, making it easier to identify any increases, decreases, or seasonal patterns in the data.")

# Bar chart that compares revenue by region
region_revenue = clean_df.groupby("region", as_index=False)["revenue"].sum() 
 
plt.figure(figsize=(7, 4)) 

plt.bar(region_revenue["region"], region_revenue["revenue"]) 
plt.title("Revenue by Region") 
plt.xlabel("Region") 
plt.ylabel("Revenue") 
plt.tight_layout() 
plt.show()

# Scatter plot that compares units sold and revenue
plt.figure(figsize=(7, 4)) 
plt.scatter(clean_df["units_sold"], clean_df["revenue"]) 
plt.title("Units Sold vs Revenue") 
plt.xlabel("Units Sold") 
plt.ylabel("Revenue") 
plt.tight_layout() 
plt.show() 

