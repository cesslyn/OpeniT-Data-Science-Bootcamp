# Week 1 Day 4 - Streamlit Dashboard 

import pandas as pd 
import streamlit as st 
import plotly.express as px 

# Load Data and Configure the Page
st.set_page_config(page_title="Revenue Dashboard", layout="wide") 
st.title("Revenue and Customer Insights Dashboard") 
 
@st.cache_data 
def load_data(): 
    df = pd.read_csv("week1-day4-cleaned-visualization-data.csv") 
    df["month"] = pd.to_datetime(df["month"]) 
    return df 
 
df = load_data() 
st.write("Preview of the data") 
st.dataframe(df.head()) 

# Sidebar filters
regions = sorted(df["region"].unique()) 
categories = sorted(df["product_category"].unique()) 
 
selected_regions = st.sidebar.multiselect( 
    "Select region", 
    options=regions, 
    default=regions 
) 
 
selected_categories = st.sidebar.multiselect( 
    "Select product category", 
    options=categories, 
    default=categories 
) 
 
filtered_df = df[ 
    df["region"].isin(selected_regions) 
    & df["product_category"].isin(selected_categories) 
] 

print("\n(1) What happens when a user removes one region from the filter?")
print("Answer: When a user removes one region from the filter, the dashboard will update to exclude data from that region. This means that all visualizations and metrics will only reflect the remaining selected regions, allowing the user to focus on specific areas of interest.")
print("\n(2) Why should dashboard filters have clear default values? ")
print("Answer: Dashboard filters should have clear default values to provide a meaningful initial view of the data. This ensures that users can immediately see the most relevant information without having to manually select filters. Clear default values also help to guide users toward the most important insights and reduce the cognitive load of making filter selections.")
print("\n(3) What message should appear if no rows match the filters? ")
print("Answer: If no rows match the filters, a message should appear indicating that no data is available for the selected criteria. This message should be clear and informative, such as 'No data available for the selected filters. Please adjust your filter selections to see results.' This helps users understand why they are not seeing any visualizations and encourages them to modify their filter choices.")

# Summary Metrics
total_revenue = filtered_df["revenue"].sum() 
total_units = filtered_df["units_sold"].sum() 
avg_satisfaction = filtered_df["customer_satisfaction"].mean() 
 
col1, col2, col3 = st.columns(3) 

col1.metric("Total Revenue", f"${total_revenue:,.0f}") 
col2.metric("Units Sold", f"{total_units:,.0f}") 
col3.metric("Avg Satisfaction", f"{avg_satisfaction:.2f}")

# Interactive Dashboard Charts
monthly_revenue = filtered_df.groupby("month", as_index=False)["revenue"].sum() 
fig_trend = px.line( 
monthly_revenue, 
x="month", 
y="revenue", 
markers=True, 
title="Revenue Trend" 
) 
st.plotly_chart(fig_trend, use_container_width=True) 
fig_scatter = px.scatter( 
filtered_df, 
x="units_sold", 
y="revenue", 
color="product_category", 
hover_data=["region", "month"], 
title="Units Sold vs Revenue" 
) 
st.plotly_chart(fig_scatter, use_container_width=True)

if filtered_df.empty: 
    st.warning("No records match the selected filters. Please adjust the sidebar selections.") 
    st.stop() 
