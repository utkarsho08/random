import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from config.settings import APP_NAME, TRAIN_DATA_PATH
from utils.ui import render_header, init_session_state

st.set_page_config(page_title=f"EDA Dashboard | {APP_NAME}", layout="wide")
init_session_state()

render_header("Exploratory Data Analysis (EDA)", "Comprehensive statistical and visual analysis of the training dataset.")

@st.cache_data
def load_data():
    return pd.read_csv(TRAIN_DATA_PATH)

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load dataset from {TRAIN_DATA_PATH}. Error: {e}")
    st.stop()

# --- 1. Dataset Overview KPIs ---
st.subheader("1. Dataset Overview")
total_rows = df.shape[0]
total_cols = df.shape[1]
missing_values = df.isnull().sum().sum()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Rows", f"{total_rows:,}")
col2.metric("Total Columns", f"{total_cols:,}")
col3.metric("Total Missing Values", f"{missing_values:,}")
col4.metric("Numerical Features", len(numeric_cols))
col5.metric("Categorical Features", len(categorical_cols))

st.divider()

# --- 2. Target Variable & Missing Values ---
st.subheader("2. Target & Data Quality")
col_targ, col_miss = st.columns(2)

with col_targ:
    st.markdown("**Loan Status Distribution**")
    if 'Loan_Status' in df.columns:
        status_counts = df['Loan_Status'].value_counts().reset_index()
        status_counts.columns = ['Loan_Status', 'Count']
        fig_status = px.pie(status_counts, names='Loan_Status', values='Count', hole=0.4, 
                            color_discrete_sequence=['#2ecc71', '#e74c3c'])
        fig_status.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_status, use_container_width=True)
    else:
        st.warning("'Loan_Status' column not found.")

with col_miss:
    st.markdown("**Missing Values by Feature**")
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ['Feature', 'Missing Count']
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    
    if not missing_df.empty:
        fig_missing = px.bar(missing_df, x='Feature', y='Missing Count', text='Missing Count',
                             color_discrete_sequence=['#f39c12'])
        fig_missing.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("No missing values found in the dataset!")

st.divider()

# --- 3. Categorical Feature Analysis ---
st.subheader("3. Categorical Feature Analysis")
if categorical_cols:
    cat_sel = st.selectbox("Select a Categorical Feature to Analyze:", categorical_cols)
    if 'Loan_Status' in df.columns and cat_sel != 'Loan_Status':
        fig_cat = px.histogram(df, x=cat_sel, color='Loan_Status', barmode='group',
                               color_discrete_sequence=['#2ecc71', '#e74c3c'],
                               title=f"{cat_sel} vs Loan Status")
    else:
        fig_cat = px.histogram(df, x=cat_sel, title=f"Distribution of {cat_sel}")
    st.plotly_chart(fig_cat, use_container_width=True)

st.divider()

# --- 4. Univariate Analysis & Outliers ---
st.subheader("4. Univariate Analysis & Outlier Detection")

if numeric_cols:
    num_sel = st.selectbox("Select a Numerical Feature to Analyze:", numeric_cols)
    
    col_hist, col_box = st.columns(2)
    with col_hist:
        fig_hist = px.histogram(df, x=num_sel, marginal='rug', title=f"Histogram of {num_sel}",
                                color_discrete_sequence=['#3498db'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col_box:
        fig_box = px.box(df, y=num_sel, title=f"Boxplot of {num_sel} (Outlier Detection)",
                         color_discrete_sequence=['#9b59b6'])
        st.plotly_chart(fig_box, use_container_width=True)

    # Outlier Summary
    st.markdown("**Outlier Summary (IQR Rule)**")
    q1 = df[num_sel].quantile(0.25)
    q3 = df[num_sel].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[num_sel] < lower_bound) | (df[num_sel] > upper_bound)]
    st.info(f"Feature: **{num_sel}** | Lower Bound: **{lower_bound:.2f}** | Upper Bound: **{upper_bound:.2f}**")
    st.warning(f"Detected **{len(outliers)}** outliers out of {total_rows} records ({len(outliers)/total_rows*100:.1f}%).")

st.divider()

# --- 5. Bivariate Analysis ---
st.subheader("5. Bivariate Analysis")

if len(numeric_cols) >= 2:
    col_x, col_y = st.columns(2)
    with col_x:
        x_axis = st.selectbox("X-Axis Feature:", numeric_cols, index=0)
    with col_y:
        y_axis = st.selectbox("Y-Axis Feature:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        
    color_col = 'Loan_Status' if 'Loan_Status' in df.columns else None
    
    fig_scatter = px.scatter(df, x=x_axis, y=y_axis, color=color_col, opacity=0.7,
                             title=f"Scatter Plot: {x_axis} vs {y_axis}",
                             color_discrete_sequence=['#2ecc71', '#e74c3c'])
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# --- 6. Correlation Heatmap ---
st.subheader("6. Correlation Heatmap")
st.markdown("Displays linear correlations between all numerical features. Values closer to 1 or -1 indicate strong relationships.")

if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
                   z=corr_matrix.values,
                   x=corr_matrix.columns,
                   y=corr_matrix.index,
                   colorscale='RdBu',
                   zmin=-1, zmax=1,
                   text=np.round(corr_matrix.values, 2),
                   texttemplate="%{text}",
                   hoverinfo="z"))
                   
    fig_corr.update_layout(height=600, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.warning("Not enough numerical features to generate a correlation heatmap.")
