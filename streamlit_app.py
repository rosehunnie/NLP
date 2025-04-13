import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.title("📊 RFM Analysis & Customer Segmentation")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your transactional data CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data", df.head())

    # Let user map columns
    st.subheader("🔧 Column Mapping")

    customer_col = st.selectbox("Select Customer ID column", df.columns)
    date_col = st.selectbox("Select Transaction Date column", df.columns)
    amount_col = st.selectbox("Select Invoice Amount column", df.columns)

    try:
        # Convert selected date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

        # RFM aggregation
        rfm = df.groupby(df[customer_col]).agg({
            date_col: lambda x: (snapshot_date - x.max()).days,
            customer_col: 'count',
            amount_col: 'sum'
        })

        rfm.columns = ['Recency', 'Frequency', 'Monetary']

        # Segment customers
        st.subheader("🏷️ RFM Segmentation")

        # Scoring by quartile
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])

        # RFM segment code
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

        # Segment labels
        def label_segment(rfm_score):
            if rfm_score.startswith('4'):
                return 'Champions'
            elif rfm_score.startswith('3'):
                return 'Loyal'
            elif rfm_score.startswith('2'):
                return 'Needs Attention'
            else:
                return 'At Risk'

        rfm['Segment'] = rfm['RFM_Score'].apply(label_segment)

        st.dataframe(rfm[['Recency', 'Frequency', 'Monetary', 'RFM_Score', 'Segment']].head(10))

        # Download results
        st.download_button(
            label="📥 Download RFM Segments as CSV",
            data=rfm.to_csv().encode('utf-8'),
            file_name='rfm_segments.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")
