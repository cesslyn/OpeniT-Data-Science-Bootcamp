# Week 1 Day 4 - Plotly Interactive Visualization 

import pandas as pd 
import plotly.express as px 

# Load the Cleaned Visualization Data
clean_df = pd.read_csv("week1-day4-cleaned-visualization-data.csv") 
clean_df["month"] = pd.to_datetime(clean_df["month"])

print(clean_df.columns) 

# Interactive Line Chart
monthly_revenue = clean_df.groupby(["month", "region"], 
as_index=False)["revenue"].sum()

fig_line = px.line( 
    monthly_revenue, 
    x="month", 
    y="revenue", 
    color="region", 
    markers=True, 
    title="Interactive Monthly Revenue by Region" 
) 
fig_line.show() 

print("\n(1) What information appears when you hover over a point? ")
print("Answer: When you hover over a point on the line chart, it displays a tooltip that shows the month, region, and revenue value for that specific point. This allows you to see the exact revenue for each region in each month.")
print("\(2) How does interactivity change the review experience?")
print("Answer: Interactivity enhances the review experience by allowing users to explore the data in more depth. Instead of just seeing a static image, users can hover over points to get more detailed information, filter data by region, and zoom in on specific time periods. This makes it easier to identify trends, outliers, and patterns that may not be immediately apparent in a static chart.")
print("\(3) What might be distracting if a chart has too many categories?")
print("Answer: If a chart has too many categories, it can become cluttered and overwhelming, making it difficult to interpret the data. The colors may blend together, and the lines may overlap, which can obscure important trends and insights. Additionally, too many categories can lead to information overload, where users may struggle to focus on the key takeaways from the chart.")

# Interactive bar chart by product category
category_revenue = clean_df.groupby("product_category", 
as_index=False)["revenue"].sum() 
 
fig_bar = px.bar( 
    category_revenue, 
    x="product_category", 
    y="revenue", 
    title="Interactive Revenue by Product Category" 
) 
fig_bar.show() 

# Interactive scatter chart
fig_scatter = px.scatter( 
    clean_df, 
    x="units_sold", 
    y="revenue", 
    color="product_category", 
    size="customer_satisfaction", 
    hover_data=["region", "month"], 
    title="Interactive Units Sold vs Revenue" 
) 
fig_scatter.show() 

fig_scatter.write_html("week1-day4-plotly-revenue-dashboard.html") 
print("Saved chart: week1-day4-plotly-revenue-dashboard.html") 