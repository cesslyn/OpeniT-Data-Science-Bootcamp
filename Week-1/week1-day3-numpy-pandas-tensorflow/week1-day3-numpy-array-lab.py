# Week 1 Day 3 - NumPy Array Lab 

import numpy as np

# Creating and Inspecting Arrays
scores = np.array([72, 88, 91, 65, 79, 95]) 
monthly_sales = np.array([ 
[1200, 1350, 1280], 
[1500, 1420, 1600], 
[980, 1100, 1050], 
[1750, 1680, 1800] 
])

print(scores.shape)
print(scores.dtype)
print(scores.ndim)
print(scores.size)

# Indexing and Slicing
print("\n")

# Inspect monthly_sales array
print("Monthly Sales Array Properties:")
print("Shape:", monthly_sales.shape)
print("Data Type:", monthly_sales.dtype)
print("Dimensions:", monthly_sales.ndim)
print("Size:", monthly_sales.size)

print("\nScores Array Operations:")

# 1. First score
print("First score:", scores[0])

# 2. Last score
print("Last score:", scores[-1])

# 3. First three scores
print("First three scores:", scores[:3])

# 4. Scores >= 80
print("Scores >= 80:", scores[scores >= 80])

print("\nMonthly Sales Array Operations:")

# 1. First row
print("First row:")
print(monthly_sales[0])

# 2. Second column
print("Second column:")
print(monthly_sales[:, 1])

# 3. Value in row 2, column 3
print("Value in row 2, column 3:", monthly_sales[1, 2])

# 4. First two rows and all columns
print("First two rows and all columns:")
print(monthly_sales[:2, :])

# Vectorized Operations
import numpy as np

scores = np.array([72, 88, 91, 65, 79, 95])

# Add 5 points and cap at 100
curved_scores = np.minimum(scores + 5, 100)
print("Curved Scores:", curved_scores)

# Passing scores (75 or higher)
passed = curved_scores >= 75
print("Passed:", passed)

# Calculations
average_original = np.mean(scores)
average_curved = np.mean(curved_scores)
highest_curved = np.max(curved_scores)
num_passing = np.sum(passed)

print("\nStatistics:")
print("Average Original Score:", average_original)
print("Average Curved Score:", average_curved)
print("Highest Curved Score:", highest_curved)
print("Number of Passing Scores:", num_passing)

# Aggregate Across Axes
sales_by_person = monthly_sales.sum(axis=1) 
sales_by_month = monthly_sales.sum(axis=0) 
overall_sales = monthly_sales.sum()

print("Sales by person:", sales_by_person) 
print("Sales by month:", sales_by_month) 
print("Overall sales:", overall_sales) 

# Broadcasting
monthly_bonus = np.array([100, 150, 200]) 
adjusted_sales = monthly_sales + monthly_bonus 
print(adjusted_sales)

# Questions
print("\n(1) Why did NumPy allow a 1D array with 3 values to be added to a 4 by 3 array? :")
print("Answer: NumPy allows this through broadcasting, where the 1D array is broadcast along the rows of the 4 by 3 array.")
print("\n(2) Which dimension did the bonus values align with? ")
print("Answer: The bonus values aligned with the columns (the second dimension) of the monthly_sales array.")