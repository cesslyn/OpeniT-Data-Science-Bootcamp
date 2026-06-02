# Week 1 Day 4 - Seaborn Statistical Visualization 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Load the Cleaned Dataset
clean_df = pd.read_csv("week1-day4-cleaned-visualization-data.csv") 
clean_df["month"] = pd.to_datetime(clean_df["month"]) 

# Category Comparison Chart
plt.figure(figsize=(8, 4)) 
sns.barplot(data=clean_df, x="product_category", y="revenue", estimator="mean") 
plt.title("Average Revenue by Product Category") 
plt.xlabel("Product Category") 
plt.ylabel("Average Revenue") 
plt.tight_layout() 
plt.savefig("week1-day4-seaborn-customer-insights.png") 
plt.show() 

print("\n(1) Which product category has the highest average revenue? ")
print("Answer: The Software category has the highest average revenue, as indicated by the tallest bar in the chart.")
print("\n(2)What does an average hide that a detailed chart might show?")
print("Answer: An average can hide the variability and distribution of the data. For example, it may not show if there are outliers or if the revenue is consistently high across all products in the category. A detailed chart might reveal that while the average revenue is high, there could be a few products driving that average, while others have much lower revenue.")
print("\n(3) Why is Seaborn useful for grouped summaries? ")
print("Answer: Seaborn is useful for grouped summaries because it provides built-in functions that make it easy to create informative visualizations that summarize data across different categories. It can automatically calculate and display summary statistics, such as means and confidence intervals, which helps in understanding the central tendency and variability of the data within groups.")

# Distribution Chart
plt.figure(figsize=(8, 4)) 
sns.histplot(data=clean_df, x="revenue", kde=True) 
plt.title("Revenue Distribution") 
plt.xlabel("Revenue") 
plt.ylabel("Count") 
plt.tight_layout() 
plt.show()

# Relationship plot
plt.figure(figsize=(8, 4)) 
sns.scatterplot( 
data=clean_df, 
x="units_sold", 
y="revenue", 
hue="product_category" 
) 
plt.title("Units Sold vs Revenue by Product Category") 
plt.xlabel("Units Sold") 
plt.ylabel("Revenue") 
plt.tight_layout() 
plt.show() 

# Correlation heatmap
numeric_df = clean_df.select_dtypes(include=["number"]) 
correlation = numeric_df.corr() 
plt.figure(figsize=(6, 4)) 
sns.heatmap(correlation, annot=True, fmt=".2f") 
plt.title("Correlation Heatmap") 
plt.tight_layout() 
plt.show() 

print("\n(1) Does correlation prove causation? Why or why not?")
print("Answer: No, correlation does not prove causation. Correlation simply indicates that there is a relationship between two variables, but it does not establish that one variable causes the other. There could be other factors at play, such as confounding variables, or the relationship could be coincidental.")