import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from config.settings import APP_NAME, MODELS_DIR
from utils.ui import render_header, init_session_state

st.set_page_config(page_title=f"Model Evaluation | {APP_NAME}", layout="wide")
init_session_state()

render_header("Model Evaluation & Comparison", "Analysis of Logistic Regression, Random Forest, and XGBoost with SMOTE balancing.")

METRICS_PATH = os.path.join(MODELS_DIR, 'evaluation_metrics.json')

if not os.path.exists(METRICS_PATH):
    st.warning("Evaluation metrics not found. Please run the training script.")
    st.stop()

with open(METRICS_PATH, 'r') as f:
    eval_data = json.load(f)

best_model = eval_data.get('best_model', 'N/A')
metrics = eval_data.get('metrics', {})

st.subheader(f"🏆 Best Model Selected: **{best_model}**")
st.markdown("The best model is selected based on the highest F1 Score to ensure a balance between Precision and Recall on the imbalanced dataset.")
st.divider()

# --- 1. Class Imbalance & SMOTE ---
st.subheader("1. Class Imbalance Mitigation (SMOTE)")
col_sm1, col_sm2 = st.columns(2)

pre_smote = eval_data.get('pre_smote_distribution', {})
post_smote = eval_data.get('post_smote_distribution', {})

# Map string keys if they exist
def map_keys(d):
    mapping = {'1': 'Approved', '0': 'Rejected', '1.0': 'Approved', '0.0': 'Rejected'}
    return {mapping.get(str(k), str(k)): v for k, v in d.items()}

pre_smote_mapped = map_keys(pre_smote)
post_smote_mapped = map_keys(post_smote)

with col_sm1:
    fig_pre = px.pie(names=list(pre_smote_mapped.keys()), values=list(pre_smote_mapped.values()), 
                     title="Target Distribution (Before SMOTE)", hole=0.4,
                     color_discrete_sequence=['#e74c3c', '#2ecc71'])
    st.plotly_chart(fig_pre, use_container_width=True)

with col_sm2:
    fig_post = px.pie(names=list(post_smote_mapped.keys()), values=list(post_smote_mapped.values()), 
                      title="Target Distribution (After SMOTE)", hole=0.4,
                      color_discrete_sequence=['#3498db', '#f1c40f'])
    st.plotly_chart(fig_post, use_container_width=True)

st.divider()

# --- 2. Metrics Leaderboard ---
st.subheader("2. Metrics Leaderboard")

leaderboard_data = []
for model_name, data in metrics.items():
    leaderboard_data.append({
        'Model': model_name,
        'Accuracy': round(data['accuracy'], 4),
        'Precision': round(data['precision'], 4),
        'Recall': round(data['recall'], 4),
        'F1 Score': round(data['f1_score'], 4),
        'ROC-AUC': round(data['roc_auc'], 4)
    })

df_leaderboard = pd.DataFrame(leaderboard_data)
st.dataframe(df_leaderboard.style.highlight_max(subset=['F1 Score', 'ROC-AUC', 'Accuracy'], color='lightgreen', axis=0), use_container_width=True)

st.divider()

# --- 3. ROC Curves ---
st.subheader("3. ROC Curve Comparison")

fig_roc = go.Figure()
fig_roc.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)

colors = ['#3498db', '#e74c3c', '#2ecc71']
for idx, (model_name, data) in enumerate(metrics.items()):
    if 'roc_curve' in data:
        fpr = data['roc_curve']['fpr']
        tpr = data['roc_curve']['tpr']
        auc = data['roc_auc']
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{model_name} (AUC={auc:.2f})", mode='lines', line=dict(color=colors[idx % len(colors)])))

fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', margin=dict(t=30, b=0, l=0, r=0))
st.plotly_chart(fig_roc, use_container_width=True)

st.divider()

# --- 4. Confusion Matrices ---
st.subheader("4. Confusion Matrices")

cols = st.columns(len(metrics))

for idx, (model_name, data) in enumerate(metrics.items()):
    cm = np.array(data['confusion_matrix'])
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', 
                       labels=dict(x="Predicted Label", y="True Label", color="Count"),
                       x=['Rejected', 'Approved'], y=['Rejected', 'Approved'],
                       title=f"{model_name}")
    fig_cm.update_layout(margin=dict(t=30, b=0, l=0, r=0))
    cols[idx].plotly_chart(fig_cm, use_container_width=True)

st.divider()

# --- 5. Feature Importance ---
st.subheader("5. Feature Importance")

model_sel = st.selectbox("Select Model for Feature Importance:", list(metrics.keys()))

if 'feature_importances' in metrics[model_sel]:
    fi = metrics[model_sel]['feature_importances']
    df_fi = pd.DataFrame({'Feature': list(fi.keys()), 'Importance': list(fi.values())})
    df_fi = df_fi.sort_values(by='Importance', ascending=True).tail(10)
    
    fig_fi = px.bar(df_fi, x='Importance', y='Feature', orientation='h', 
                    title=f"Top 10 Important Features ({model_sel})",
                    color_discrete_sequence=['#9b59b6'])
    st.plotly_chart(fig_fi, use_container_width=True)
else:
    st.info(f"Feature importance not available or extractable for {model_sel}.")
