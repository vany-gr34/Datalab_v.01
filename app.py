import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


from helpers.file_parser import parse_contents
from helpers.model_registry import CLASSIFIERS, REGRESSORS
from preprocessing.api import PreprocessingAPI
from visalustsation import VisualizerManager, NumericVisualizer, CategoricalVisualizer, CorrelationVisualizer
from ingestion import ingest
from ingestion.base_ingestor import IngestionError
from deployment.exporter import ExportManager
import joblib

# Streamlit page config
st.set_page_config(
    page_title="DataLab Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dashboard CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: #0a0e27;
        color: #e4e4e7;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0a0e27 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    
    /* Sidebar Title */
    [data-testid="stSidebar"] h1 {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.25rem;
        letter-spacing: -0.02em;
        margin-bottom: 2rem;
        padding-left: 0.5rem;
    }
    
    /* Radio Buttons - Navigation */
    [data-testid="stSidebar"] .stRadio > label {
        color: #9ca3af;
        font-weight: 500;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stRadio > div {
        background: transparent;
        padding: 0;
        gap: 0.25rem;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid transparent;
        padding: 0.75rem 1rem;
        border-radius: 0;
        margin: 0;
        transition: all 0.2s ease;
        color: #d1d5db;
        font-weight: 400;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(59, 130, 246, 0.1);
        border-left-color: #3b82f6;
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"] > div:first-child {
        display: none;
    }
    
    /* Headers */
    h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2.5rem;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #f3f4f6;
        font-weight: 600;
        font-size: 1.75rem;
        letter-spacing: -0.02em;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    h3 {
        color: #e5e7eb;
        font-weight: 600;
        font-size: 1.25rem;
        letter-spacing: -0.01em;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: #3b82f6;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 0.625rem 1.5rem;
        font-weight: 500;
        font-size: 0.9375rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        letter-spacing: 0.01em;
    }
    
    .stButton > button:hover {
        background: #2563eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    .stButton > button:disabled {
        background: #374151;
        color: #6b7280;
        cursor: not-allowed;
        box-shadow: none;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* Alert Boxes */
    .stAlert {
        background: rgba(31, 41, 55, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
        backdrop-filter: blur(8px);
    }
    
    [data-testid="stNotificationContentInfo"] {
        background: rgba(59, 130, 246, 0.1);
        border-left: 3px solid #3b82f6;
        color: #93c5fd;
    }
    
    [data-testid="stNotificationContentSuccess"] {
        background: rgba(34, 197, 94, 0.1);
        border-left: 3px solid #22c55e;
        color: #86efac;
    }
    
    [data-testid="stNotificationContentWarning"] {
        background: rgba(251, 146, 60, 0.1);
        border-left: 3px solid #fb923c;
        color: #fdba74;
    }
    
    [data-testid="stNotificationContentError"] {
        background: rgba(239, 68, 68, 0.1);
        border-left: 3px solid #ef4444;
        color: #fca5a5;
    }
    
    /* Data Frames */
    [data-testid="stDataFrame"] {
        background: rgba(17, 24, 39, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0;
        overflow: hidden;
    }
    
    [data-testid="stDataFrame"] table {
        font-size: 0.875rem;
    }
    
    [data-testid="stDataFrame"] thead tr {
        background: rgba(31, 41, 55, 0.8);
        color: #9ca3af;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stDataFrame"] tbody tr {
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stDataFrame"] tbody tr:hover {
        background: rgba(59, 130, 246, 0.05);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(17, 24, 39, 0.5);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
        background: rgba(59, 130, 246, 0.05);
    }
    
    [data-testid="stFileUploader"] section {
        border: none;
        padding: 0;
    }
    
    [data-testid="stFileUploader"] button {
        background: transparent;
        color: #3b82f6;
        border: 1px solid #3b82f6;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background: rgba(59, 130, 246, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 0;
        padding: 0.75rem 1.5rem;
        color: #9ca3af;
        font-weight: 500;
        border-bottom: 2px solid transparent;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #e5e7eb;
        background: rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background: transparent;
        color: #3b82f6;
        border-bottom-color: #3b82f6;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 6px;
        color: #e5e7eb;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #3b82f6;
    }
    
    .stMultiSelect > div > div {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 6px;
    }
    
    .stMultiSelect > div > div:hover {
        border-color: #3b82f6;
    }
    
    /* Multiselect tags */
    .stMultiSelect span[data-baseweb="tag"] {
        background: #3b82f6;
        color: #ffffff;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    /* Text Input */
    .stTextInput > div > div {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 6px;
        color: #e5e7eb;
    }
    
    .stTextInput > div > div:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #3b82f6;
        font-size: 2rem;
        font-weight: 700;
        font-feature-settings: "tnum";
    }
    
    [data-testid="stMetricLabel"] {
        color: #9ca3af;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Metric Container */
    [data-testid="metric-container"] {
        background: rgba(17, 24, 39, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1.25rem;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #3b82f6;
    }
    
    /* Code blocks */
    .stCode {
        background: rgba(15, 15, 30, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.875rem;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(31, 41, 55, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(107, 114, 128, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(156, 163, 175, 0.7);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(17, 24, 39, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        color: #e5e7eb;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(31, 41, 55, 0.5);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: #3b82f6;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: #3b82f6;
    }
    
    /* Card containers */
    div.element-container {
        margin-bottom: 1rem;
    }
    
    /* Tables */
    .stTable {
        background: rgba(17, 24, 39, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    
    .stTable table {
        font-size: 0.875rem;
    }
    
    .stTable thead tr {
        background: rgba(31, 41, 55, 0.8);
        color: #9ca3af;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }
    
    .stTable tbody tr {
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTable tbody tr:hover {
        background: rgba(59, 130, 246, 0.05);
    }
    
    /* JSON display */
    .stJson {
        background: rgba(15, 15, 30, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.875rem;
    }
    
    /* Markdown */
    .stMarkdown {
        color: #d1d5db;
    }
    
    .stMarkdown a {
        color: #3b82f6;
        text-decoration: none;
    }
    
    .stMarkdown a:hover {
        color: #60a5fa;
        text-decoration: underline;
    }
    
    /* Custom Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .status-success {
        background: #22c55e;
    }
    
    .status-warning {
        background: #fb923c;
    }
    
    .status-error {
        background: #ef4444;
    }
    
    /* Footer */
    footer {
        color: #6b7280;
        text-align: center;
        padding: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 3rem;
    }
    
    /* Image styling */
    .stImage {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

if 'stored_data' not in st.session_state:
    st.session_state.stored_data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = "regression"
if 'deployed_models' not in st.session_state:
    st.session_state.deployed_models = {}
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

st.sidebar.title("Navigation")
phase = st.sidebar.radio("Select Phase", ["Upload Data", "Visualization", "Preprocessing", "Model Training", "Deployment"])

st.title("DataLab Analytics Platform")
st.caption("Enterprise-grade machine learning workspace")




if phase == "Upload Data":
    st.header("Data Upload & Management")

    # Create tabs for different upload methods
    tab1, tab2, tab3, tab4 = st.tabs(["File Upload", "Database", "API", "Ingestion History"])

    with tab1:
        st.subheader("File Upload")
        uploaded_file = st.file_uploader(
            "Upload data file",
            type=["csv", "xlsx", "xls", "json", "parquet"],
            help="Supported formats: CSV, Excel (.xlsx/.xls), JSON, Parquet"
        )

        if uploaded_file:
            try:
                with st.spinner("Processing file with ingestion layer..."):
                    # Convert uploaded file to base64 for ingestion layer
                    file_content = base64.b64encode(uploaded_file.read()).decode()

                    # Determine file type
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    file_type_map = {
                        'csv': 'csv',
                        'xlsx': 'excel',
                        'xls': 'excel',
                        'json': 'json',
                        'parquet': 'parquet'
                    }
                    file_type = file_type_map.get(file_extension, 'csv')

                    # Use ingestion layer
                    dataset = ingest('upload', {
                        'file_content': file_content,
                        'file_name': uploaded_file.name,
                        'file_type': file_type,
                        'content_type': uploaded_file.type or 'application/octet-stream'
                    })

                    st.session_state.stored_data = dataset.data
                    st.session_state.current_dataset = dataset

                    st.success(f"Successfully ingested {uploaded_file.name}")
                    st.info(f"Dataset ID: {dataset.id}")
                    st.info(f"Source: {dataset.metadata.get('source_type', 'unknown')}")

            except IngestionError as e:
                st.error(f"Ingestion failed: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    with tab2:
        st.subheader("Database Connection")
        st.info("Database ingestion coming soon")

        # Placeholder for database connection form
        db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "SQLite"])
        host = st.text_input("Host")
        database = st.text_input("Database Name")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        table = st.text_input("Table Name")

        if st.button("Connect & Ingest", disabled=True):
            st.info("This feature is under development.")

    with tab3:
        st.subheader("API Ingestion")
        st.info("API ingestion coming soon")

        # Placeholder for API configuration
        api_url = st.text_input("API URL")
        auth_type = st.selectbox("Authentication", ["None", "API Key", "Bearer Token", "Basic Auth"])

        if auth_type == "API Key":
            api_key = st.text_input("API Key", type="password")
        elif auth_type == "Bearer Token":
            token = st.text_input("Bearer Token", type="password")

        if st.button("Fetch Data", disabled=True):
            st.info("This feature is under development.")

    with tab4:
        st.subheader("Ingestion History")
        from ingestion import get_ingestion_registry
        registry = get_ingestion_registry()
        history = registry.get_ingestion_history()

        if history:
            history_df = pd.DataFrame(history)
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No ingestion history available.")
    
    if st.session_state.stored_data is not None:
        df = st.session_state.stored_data
        
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        st.subheader("Data Summary")
        st.write(df.describe(include='all'))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.subheader("Column Information")
        col_info = []
        for col in df.columns:
            col_type = df[col].dtype
            missing = df[col].isna().sum()
            if col_type == 'object' or col_type.name == 'category':
                unique_vals = df[col].nunique()
            else:
                unique_vals = None
            col_info.append({
                "Column": col,
                "Type": col_type,
                "Missing Values": missing,
                "Unique Values": unique_vals
            })
        st.table(pd.DataFrame(col_info))
    else:
        st.info("Please upload a data file to get started.")

elif phase == "Visualization":
    st.header("Interactive Data Visualization")

    if st.session_state.stored_data is None:
        st.warning("Upload data first")
        st.stop()

    # Dataset Scope
    dataset_options = ["raw"]
    if st.session_state.preprocessed_data is not None:
        dataset_options.append("preprocessed")
    
    # Control Panel at the top
    with st.expander("Visualization Controls", expanded=True):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            dataset_stage = st.selectbox("Dataset Stage", dataset_options, help="Choose raw or preprocessed dataset")
        
        df = st.session_state.preprocessed_data if dataset_stage == "preprocessed" else st.session_state.stored_data

        # Auto-detect column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

        st.markdown("### Feature Selection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # X Feature
            x_feature = st.selectbox("X Feature", df.columns.tolist(), help="Primary feature for visualization")
        
        with col2:
            # Y Feature (optional)
            y_options = ["None"] + df.columns.tolist()
            y_feature = st.selectbox("Y Feature (optional)", y_options, help="Secondary feature, optional for some plots")
        
        with col3:
            # Color / Group By
            color_options = ["None"] + df.columns.tolist()
            color_feature = st.selectbox("Color / Group By (optional)", color_options, help="Feature for coloring or grouping")

        # Determine available plot types based on selected features
        x_type = "numeric" if x_feature in numeric_cols else "categorical" if x_feature in categorical_cols else "datetime"
        y_type = "numeric" if y_feature in numeric_cols else "categorical" if y_feature in categorical_cols else "datetime" if y_feature in datetime_cols else None

        available_plots = []
        if x_type == "numeric":
            available_plots.extend(["Histogram", "Box Plot"])
            if y_type == "numeric":
                available_plots.extend(["Scatter Plot", "Line Plot"])
            elif y_type == "categorical":
                available_plots.append("Box Plot")
        elif x_type == "categorical":
            available_plots.extend(["Bar Chart", "Count Plot"])
            if y_type == "numeric":
                available_plots.append("Bar Chart")
        elif x_type == "datetime":
            available_plots.extend(["Line Plot", "Area Plot"])

        # Special plots
        if len(numeric_cols) >= 2:
            available_plots.append("Correlation Matrix")
            if len(numeric_cols) <= 5:
                available_plots.append("Pair Plot")

        if not available_plots:
            available_plots = ["Histogram", "Bar Chart"]

        st.markdown("### Visualization Type")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            plot_type = st.selectbox("Plot Type", available_plots)
        
        with col2:
            # Sampling size
            sample_size = st.slider("Sample Size (%)", 10, 100, 100, help="Percentage of data to use for plotting")
        
        # Aggregation function (for bar charts with Y)
        if plot_type in ["Bar Chart"] and y_feature != "None" and y_type == "numeric":
            agg_func = st.selectbox("Aggregation Function", ["mean", "sum", "count", "median"], help="How to aggregate Y values")
        else:
            agg_func = None

    # Filters Section
    with st.expander("Data Filters", expanded=False):
        st.markdown("### Apply Filters")
        
        # Apply sampling first
        if sample_size < 100:
            df_plot = df.sample(frac=sample_size/100, random_state=42)
        else:
            df_plot = df.copy()
        
        # Numerical filters
        if numeric_cols:
            st.markdown("**Numeric Filters**")
            num_filter_cols = st.columns(min(3, len(numeric_cols)))
            for idx, col in enumerate(numeric_cols):
                if col in df_plot.columns:
                    col_non_na = df_plot[col].dropna()
                    if col_non_na.empty:
                        continue
                    try:
                        min_val = float(col_non_na.min())
                        max_val = float(col_non_na.max())
                    except Exception:
                        continue
                    if min_val != max_val:
                        with num_filter_cols[idx % 3]:
                            filter_range = st.slider(f"{col}", min_val, max_val, (min_val, max_val), key=f"num_{col}")
                            if filter_range != (min_val, max_val):
                                df_plot = df_plot[(df_plot[col] >= filter_range[0]) & (df_plot[col] <= filter_range[1])]
        
        # Categorical filters
        if categorical_cols:
            st.markdown("**Categorical Filters**")
            cat_filter_cols = st.columns(min(3, len(categorical_cols)))
            for idx, col in enumerate(categorical_cols):
                if col in df_plot.columns:
                    unique_vals = df_plot[col].dropna().unique().tolist()
                    if len(unique_vals) > 1 and len(unique_vals) <= 50:  # Only show if reasonable number of unique values
                        with cat_filter_cols[idx % 3]:
                            selected_vals = st.multiselect(f"{col}", unique_vals, default=unique_vals, key=f"cat_{col}")
                            if selected_vals != unique_vals:
                                df_plot = df_plot[df_plot[col].isin(selected_vals)]
    
    # If no filters applied, set df_plot
    if 'df_plot' not in locals():
        if sample_size < 100:
            df_plot = df.sample(frac=sample_size/100, random_state=42)
        else:
            df_plot = df.copy()

    # Ensure numeric columns used for plotting are actual numeric dtypes (coerce when possible)
    for col in numeric_cols:
        if col in df_plot.columns:
            try:
                df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
            except Exception:
                pass

    # Generate plot
    st.subheader(f"{plot_type} of {x_feature}" + (f" vs {y_feature}" if y_feature != "None" else ""))

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_style("darkgrid")
    plt.rcParams['figure.facecolor'] = '#0a0e27'
    plt.rcParams['axes.facecolor'] = '#111827'
    plt.rcParams['axes.edgecolor'] = '#374151'
    plt.rcParams['text.color'] = '#e5e7eb'
    plt.rcParams['axes.labelcolor'] = '#e5e7eb'
    plt.rcParams['xtick.color'] = '#9ca3af'
    plt.rcParams['ytick.color'] = '#9ca3af'
    plt.rcParams['grid.color'] = '#374151'

    try:
        if plot_type == "Histogram":
            if x_type == "numeric":
                sns.histplot(data=df_plot, x=x_feature, hue=color_feature if color_feature != "None" else None, ax=ax)
            else:
                st.error("Histogram requires numeric X feature")

        elif plot_type == "Box Plot":
            if x_type == "numeric":
                if y_feature == "None":
                    sns.boxplot(data=df_plot, x=x_feature, ax=ax)
                else:
                    sns.boxplot(data=df_plot, x=x_feature, y=y_feature, ax=ax)
            elif x_type == "categorical" and y_type == "numeric":
                sns.boxplot(data=df_plot, x=x_feature, y=y_feature, ax=ax)
            else:
                st.error("Box plot configuration not supported")

        elif plot_type == "Scatter Plot":
            if x_type == "numeric" and y_type == "numeric":
                sns.scatterplot(data=df_plot, x=x_feature, y=y_feature, hue=color_feature if color_feature != "None" else None, ax=ax)
            else:
                st.error("Scatter plot requires numeric X and Y features")

        elif plot_type == "Line Plot":
            if x_type in ["numeric", "datetime"] and y_type == "numeric":
                sns.lineplot(data=df_plot, x=x_feature, y=y_feature, hue=color_feature if color_feature != "None" else None, ax=ax)
            else:
                st.error("Line plot requires numeric X and Y features")

        elif plot_type == "Bar Chart":
            if x_type == "categorical":
                if y_feature == "None":
                    sns.countplot(data=df_plot, x=x_feature, ax=ax)
                elif y_type == "numeric":
                    if agg_func:
                        df_agg = df_plot.groupby(x_feature)[y_feature].agg(agg_func).reset_index()
                        sns.barplot(data=df_agg, x=x_feature, y=y_feature, ax=ax)
                    else:
                        sns.barplot(data=df_plot, x=x_feature, y=y_feature, ax=ax)
                else:
                    st.error("Bar chart Y must be numeric or None")
            else:
                st.error("Bar chart requires categorical X feature")

        elif plot_type == "Count Plot":
            if x_type == "categorical":
                sns.countplot(data=df_plot, x=x_feature, hue=color_feature if color_feature != "None" else None, ax=ax)
            else:
                st.error("Count plot requires categorical X feature")

        elif plot_type == "Area Plot":
            if x_type == "datetime" and y_type == "numeric":
                df_plot = df_plot.sort_values(x_feature)
                ax.fill_between(df_plot[x_feature], df_plot[y_feature], alpha=0.5)
                ax.plot(df_plot[x_feature], df_plot[y_feature])
            else:
                st.error("Area plot requires datetime X and numeric Y features")

        elif plot_type == "Correlation Matrix":
            corr_cols = numeric_cols[:10]  # Limit for readability
            corr_matrix = df_plot[corr_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title("Correlation Matrix")

        elif plot_type == "Pair Plot":
            pair_cols = numeric_cols[:4]  # Limit for performance
            if len(pair_cols) >= 2:
                pair_df = df_plot[pair_cols].copy()
                # Ensure pairplot columns are numeric; coerce and drop fully-NaN columns
                for c in pair_df.columns:
                    pair_df[c] = pd.to_numeric(pair_df[c], errors='coerce')
                pair_df = pair_df.dropna(axis=1, how='all')
                if pair_df.shape[1] < 2:
                    st.error("Need at least 2 numeric columns for pair plot")
                else:
                    pairplot_fig = sns.pairplot(pair_df, diag_kind='kde', plot_kws={'alpha': 0.6})
                    st.pyplot(pairplot_fig)
                    plt.close(pairplot_fig)
                    fig = None  # Don't show the main fig
            else:
                st.error("Need at least 2 numeric columns for pair plot")

        if fig:
            st.pyplot(fig)
            plt.close(fig)

        # Show data info
        st.subheader("Visualization Info")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points", f"{len(df_plot):,}")
        with col2:
            st.metric("X Feature Type", x_type.capitalize())
        with col3:
            st.metric("Y Feature Type", y_type.capitalize() if y_type else "None")

    except Exception as e:
        st.error(f"Error generating plot: {str(e)}")
        st.code(str(e))

elif phase == "Preprocessing":
    st.header("Data Preprocessing")

    if st.session_state.stored_data is None:
        st.warning("Upload data first")
    else:
        df = st.session_state.stored_data.copy()
        
        # Show original data info
        st.subheader("Original Data")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Preprocessing Options", "Data Info", "Manual Column Selection"])
        
        with tab1:
            # Drop columns
            drop_cols = st.multiselect(
                "Select columns to drop (optional):",
                options=df.columns.tolist()
            )
            if drop_cols:
                df = df.drop(columns=drop_cols)
                st.info(f"Dropped columns: {', '.join(drop_cols)}")
            
            # Auto-detect column types
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            st.write(f"**Detected:** {len(numeric_cols)} numeric, {len(categorical_cols)} categorical columns")
            
            # Preprocessing steps
            preprocessing_options = st.multiselect(
                "Select preprocessing steps:",
                ["missing", "scale", "encode"],
                format_func=lambda x: {
                    "missing": "Handle Missing Values",
                    "scale": "Scale Numeric Features",
                    "encode": "Encode Categorical Features"
                }.get(x)
            )
            
            user_options = {}
            if preprocessing_options:
                if "missing" in preprocessing_options:
                    user_options["missing_strategy"] = st.selectbox(
                        "Missing Value Strategy:",
                        ["impute", "delete", "impute_indicator"]
                    )
                    if user_options["missing_strategy"] != "delete":
                        fill_strategy = st.selectbox(
                            "Fill Strategy (for numeric columns):",
                            ["mean", "median", "most_frequent", "constant"]
                        )
                        user_options["fill_strategy"] = fill_strategy
                        
                        # Warn about categorical columns with mean
                        if fill_strategy == "mean" and categorical_cols:
                            st.warning("""
                            **Note:** 'mean' strategy only works for numeric columns.
                            Categorical columns will use 'most_frequent' strategy automatically.
                            """)
                
                if "scale" in preprocessing_options:
                    user_options["scaler_type"] = st.selectbox(
                        "Scaler Type:",
                        ["standard", "minmax", "robust", "maxabs"]
                    )
                
                if "encode" in preprocessing_options:
                    user_options["encoder_type"] = st.selectbox(
                        "Encoder Type:",
                        ["onehot", "ordinal", "label"]
                    )
        
        with tab2:
            # Show column info
            col_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                missing = df[col].isna().sum()
                unique = df[col].nunique()
                col_type = "Numeric" if col in numeric_cols else "Categorical"
                
                col_info.append({
                    "Column": col,
                    "Type": col_type,
                    "Data Type": dtype,
                    "Missing": missing,
                    "Unique": unique
                })
            
            col_df = pd.DataFrame(col_info)
            st.dataframe(col_df, use_container_width=True)
        
        with tab3:
           
            st.write("Manually adjust column types if auto-detection is wrong:")
            
            all_cols = df.columns.tolist()
            manual_numeric = st.multiselect(
                "Force as numeric columns:",
                options=all_cols,
                default=[col for col in all_cols if col in numeric_cols]
            )
            
            manual_categorical = st.multiselect(
                "Force as categorical columns:",
                options=all_cols,
                default=[col for col in all_cols if col in categorical_cols]
            )
            
            if st.button("Apply Manual Types"):
                numeric_cols = manual_numeric
                categorical_cols = manual_categorical
                st.success("Column types updated")
        
        # Apply preprocessing
        if st.button("Apply Preprocessing", type="primary"):
            if not preprocessing_options:
                st.warning("Please select at least one preprocessing option")
            else:
                try:
                    user_options.update({step: True for step in preprocessing_options})

                    # Add column information to options
                    user_options["numeric_columns"] = numeric_cols
                    user_options["categorical_columns"] = categorical_cols

                    # Build config for PreprocessingAPI
                    api = PreprocessingAPI()
                    # Map missing strategy
                    missing = user_options.get('missing_strategy') if user_options.get('missing_strategy') else 'mean'
                    if missing == 'impute_indicator':
                        # keep imputation; indicator handled by missing handler
                        missing = user_options.get('fill_strategy', 'mean')
                    if missing == 'impute':
                        missing = user_options.get('fill_strategy', 'mean')

                    config = api.create_preprocessing_config(
                        handle_missing=missing,
                        handle_outliers=None,
                        scale_numeric=user_options.get('scaler_type') if user_options.get('scaler_type') else None,
                        encode_categorical=user_options.get('encoder_type') if user_options.get('encoder_type') else None
                    )

                    with st.spinner("Applying preprocessing..."):
                        dataset_id = f"ui_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                        transformed_df, metadata = api.preprocess(df, config, dataset_id=dataset_id, is_training=True)

                    # Store results
                    st.session_state.preprocessed_data = transformed_df
                    st.session_state.preprocess_metadata = metadata.to_dict()
                    st.success("Preprocessing successful")

                    # Show results and metadata
                    st.subheader("Processed Data")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rows", f"{transformed_df.shape[0]:,}")
                    with col2:
                        st.metric("Columns", transformed_df.shape[1])
                    
                    st.dataframe(transformed_df.head(), use_container_width=True)
                    
                    st.subheader("Preprocessing Summary")
                    st.json(metadata.to_dict())

                    # Generate quick visualizations of processed data
                    viz_manager = VisualizerManager([NumericVisualizer(), CategoricalVisualizer(), CorrelationVisualizer()])
                    figs = viz_manager.run(transformed_df, target_column=st.session_state.get('target_col'))
                    from helpers.viz_utils import figs_to_html
                    images = figs_to_html(figs)
                    for item in images:
                        st.markdown(f"**{item['name']}**")
                        st.image(item['img'])

                except Exception as e:
                    st.error(f"Error: {str(e)}")

elif phase == "Model Training":
    st.header("Model Training")
    
    if st.session_state.stored_data is None:
        st.warning("Upload data first")
    else:
        df = st.session_state.preprocessed_data if st.session_state.preprocessed_data is not None else st.session_state.stored_data
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.problem_type = st.selectbox(
                "Select Problem Type:", 
                ["regression", "classification"], 
                index=0 if st.session_state.problem_type == "regression" else 1,
                format_func=lambda x: f"{x.capitalize()}"
            )
        
        with col2:
            st.session_state.target_col = st.selectbox(
                "Select Target Column:", 
                df.columns, 
                index=df.columns.get_loc(st.session_state.target_col) if st.session_state.target_col in df.columns else 0
            )
        
        model_options = REGRESSORS if st.session_state.problem_type == "regression" else CLASSIFIERS
        selected_models = st.multiselect(
            "Select Models:", 
            list(model_options.keys()), 
            default=list(model_options.keys())[:2]
        )
        
        if st.button("Run Models", type="primary"):
            target_col = st.session_state.target_col
            if target_col not in df.columns:
                st.error("Invalid target column selected")
            else:
                X = df.drop(columns=[target_col])
                y = df[target_col]
                X = pd.get_dummies(X)  # Quick categorical handling if needed
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
                
                st.subheader("Model Results")
                results = []
                for model_name in selected_models:
                    ModelClass = model_options[model_name]
                    model = ModelClass()
                    with st.spinner(f"Training {model_name}..."):
                        try:
                            # Assuming models have a .run method as in the provided code; adjust if standard sklearn
                            y_pred, metrics = model.run(X_train, X_valid, y_train, y_valid)
                            results.append({
                                "Model": model_name,
                                "R2" if st.session_state.problem_type == "regression" else "Accuracy": round(metrics.get("R2", metrics.get("Accuracy", 0)), 4),
                                "RMSE" if st.session_state.problem_type == "regression" else "F1": round(metrics.get("RMSE", metrics.get("F1", 0)), 4),
                                "MAE" if st.session_state.problem_type == "regression" else "Precision": round(metrics.get("MAE", metrics.get("Precision", 0)), 4)
                            })
                            # Store trained model
                            st.session_state.trained_models[model_name] = {
                                'model': model,
                                'metrics': metrics,
                                'problem_type': st.session_state.problem_type,
                                'target_col': st.session_state.target_col,
                                'features': list(X.columns)
                            }
                        except AttributeError:
                            # Fallback to standard fit/score if .run not available
                            model.fit(X_train, y_train)
                            score = model.score(X_valid, y_valid)
                            results.append({
                                "Model": model_name,
                                "Score": round(score, 4)
                            })
                            # Store trained model
                            st.session_state.trained_models[model_name] = {
                                'model': model,
                                'score': score,
                                'problem_type': st.session_state.problem_type,
                                'target_col': st.session_state.target_col,
                                'features': list(X.columns)
                            }
                        except Exception as e:
                            st.error(f"{model_name}: Error - {str(e)}")
                
                if results:
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    if st.session_state.problem_type == "regression":
                        best_row = results_df.sort_values("R2", ascending=False).iloc[0]
                        st.success(f"Best Model: {best_row['Model']} (R2 = {best_row['R2']})")
                    else:
                        best_row = results_df.sort_values("Accuracy", ascending=False).iloc[0]
                        st.success(f"Best Model: {best_row['Model']} (Accuracy = {best_row['Accuracy']})")
                else:
                    st.info("No models ran successfully.")

elif phase == "Deployment":
    st.header("Model Deployment & Export")

    if not st.session_state.trained_models:
        st.warning("No trained models available. Please train models first.")
    else:
        # Create tabs for deployment actions
        tab1, tab2, tab3 = st.tabs(["Deploy Models", "Deployed Models", "Export Models"])

        with tab1:
            st.subheader("Select Models to Deploy")

            # Show available trained models
            available_models = list(st.session_state.trained_models.keys())
            selected_for_deployment = st.multiselect(
                "Select models to deploy:",
                available_models,
                help="Choose which trained models to deploy"
            )

            if selected_for_deployment:
                st.subheader("Deployment Configuration")

                col1, col2 = st.columns(2)
                with col1:
                    deployment_name = st.text_input(
                        "Deployment Name:",
                        value=f"deployment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                        help="Name for this deployment"
                    )

                with col2:
                    export_formats = st.multiselect(
                        "Export Formats:",
                        ["joblib", "json", "png"],
                        default=["joblib"],
                        help="Formats to export the model in"
                    )

                # Show model details
                st.subheader("Model Details")
                for model_name in selected_for_deployment:
                    model_info = st.session_state.trained_models[model_name]
                    with st.expander(f"📊 {model_name} Details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Problem Type", model_info['problem_type'].capitalize())
                        with col2:
                            st.metric("Target Column", model_info['target_col'])
                        with col3:
                            if 'metrics' in model_info:
                                if model_info['problem_type'] == 'regression':
                                    st.metric("R² Score", f"{model_info['metrics'].get('R2', 'N/A'):.4f}")
                                else:
                                    st.metric("Accuracy", f"{model_info['metrics'].get('Accuracy', 'N/A'):.4f}")
                            else:
                                st.metric("Score", f"{model_info.get('score', 'N/A'):.4f}")

                        st.write(f"**Features:** {', '.join(model_info['features'])}")

                if st.button("🚀 Deploy Selected Models", type="primary"):
                    try:
                        exporter = ExportManager()
                        deployed_count = 0

                        for model_name in selected_for_deployment:
                            model_info = st.session_state.trained_models[model_name]

                            # Prepare metadata
                            metadata = {
                                'deployment_name': deployment_name,
                                'model_name': model_name,
                                'problem_type': model_info['problem_type'],
                                'target_column': model_info['target_col'],
                                'features': model_info['features'],
                                'metrics': model_info.get('metrics', {}),
                                'deployed_at': pd.Timestamp.now().isoformat()
                            }

                            # Export model
                            export_result = exporter.export_model(
                                model=model_info['model'],
                                model_name=model_name,
                                preprocessing_pipeline=None,  # Could be added later
                                metadata=metadata
                            )

                            # Store in deployed models dict
                            deployment_key = f"{deployment_name}_{model_name}"
                            st.session_state.deployed_models[deployment_key] = {
                                'model_info': model_info,
                                'export_paths': export_result,
                                'metadata': metadata,
                                'deployment_name': deployment_name
                            }

                            deployed_count += 1

                        st.success(f"✅ Successfully deployed {deployed_count} model(s)!")

                        # Show export locations
                        st.subheader("Export Locations")
                        for model_name in selected_for_deployment:
                            deployment_key = f"{deployment_name}_{model_name}"
                            deployed_data = st.session_state.deployed_models.get(deployment_key)
                            if deployed_data:
                                with st.expander(f"📁 {model_name} Export Paths"):
                                    for export_type, path in deployed_data['export_paths'].items():
                                        st.code(f"{export_type.upper()}: {path}", language="text")

                    except Exception as e:
                        st.error(f"Deployment failed: {str(e)}")
                        st.code(str(e))

        with tab2:
            st.subheader("Currently Deployed Models")

            # Get all deployed models
            deployed_keys = list(st.session_state.deployed_models.keys())

            if not deployed_keys:
                st.info("No models currently deployed.")
            else:
                # Group by deployment name
                deployments = {}
                for key in deployed_keys:
                    try:
                        data = st.session_state.deployed_models.get(key)
                        dep_name = data.get('deployment_name', 'Unknown')
                        if dep_name not in deployments:
                            deployments[dep_name] = []
                        deployments[dep_name].append((key, data))
                    except:
                        continue

                for dep_name, models in deployments.items():
                    with st.expander(f"🚀 Deployment: {dep_name} ({len(models)} models)"):
                        for key, data in models:
                            model_name = data['metadata']['model_name']
                            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                            with col1:
                                st.write(f"**{model_name}**")
                                st.caption(f"Target: {data['metadata']['target_column']}")

                            with col2:
                                problem_type = data['metadata']['problem_type']
                                st.metric("Type", problem_type.capitalize())

                            with col3:
                                if 'metrics' in data['model_info'] and data['model_info']['metrics']:
                                    if problem_type == 'regression':
                                        score = data['model_info']['metrics'].get('R2', 'N/A')
                                    else:
                                        score = data['model_info']['metrics'].get('Accuracy', 'N/A')
                                    st.metric("Score", f"{score:.4f}" if isinstance(score, (int, float)) else score)
                                else:
                                    st.metric("Score", "N/A")

                            with col4:
                                if st.button(f"🗑️ Delete", key=f"delete_{key}"):
                                    try:
                                        del st.session_state.deployed_models[key]
                                        st.success(f"Deleted {model_name}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Delete failed: {str(e)}")

        with tab3:
            st.subheader("Export Deployed Models")

            # Get deployed models for export
            exportable_models = list(st.session_state.deployed_models.keys())

            if not exportable_models:
                st.info("No deployed models available for export.")
            else:
                selected_for_export = st.multiselect(
                    "Select deployed models to export:",
                    exportable_models,
                    help="Choose which deployed models to export"
                )

                if selected_for_export:
                    export_formats = st.multiselect(
                        "Export Formats:",
                        ["joblib", "json", "png"],
                        default=["joblib"],
                        help="Formats to export the model in"
                    )

                    if st.button("📤 Export Selected Models", type="primary"):
                        try:
                            exporter = ExportManager()
                            export_results = {}

                            for key in selected_for_export:
                                data = st.session_state.deployed_models.get(key)
                                model_info = data['model_info']
                                metadata = data['metadata']

                                # Export with additional formats if requested
                                result = exporter.export_model(
                                    model=model_info['model'],
                                    model_name=metadata['model_name'],
                                    preprocessing_pipeline=None,
                                    metadata=metadata
                                )

                                export_results[key] = result

                            st.success("✅ Export completed!")

                            # Show export results
                            for key, paths in export_results.items():
                                data = st.session_state.deployed_models.get(key)
                                model_name = data['metadata']['model_name']
                                with st.expander(f"📁 {model_name} Export Results"):
                                    for fmt, path in paths.items():
                                        st.code(f"{fmt.upper()}: {path}", language="text")

                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")
                            st.code(str(e))

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6b7280; padding: 2rem 0;'>
        <p><strong>DataLab Analytics Platform</strong> | Enterprise ML Workspace</p>
        <p style='font-size: 0.875rem; margin-top: 0.5rem;'>Built with Streamlit</p>
    </div>
    """, 
    unsafe_allow_html=True
)
