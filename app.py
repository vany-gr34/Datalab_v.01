import streamlit as st

from modules import upload, visualization, preprocessing, training, deployment

# Streamlit page config
st.set_page_config(
    page_title="DataLab v0.1",
    layout="wide",
    initial_sidebar_state="collapsed"
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
        background: linear-gradient(180deg, #0f172a 0%, #0a0e27 100%);
        border-right: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
    
    /* Sidebar Header Section */
    [data-testid="stSidebar"] h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 1.5rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        padding: 0 1rem;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Add logo/icon area above title */
    [data-testid="stSidebar"]::before {
        content: ";
        display: block;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 1.2rem;
        filter: drop-shadow(0 0 10px rgba(59, 130, 246, 0.5));
    }
    
    /* Sidebar Subtitle */
    [data-testid="stSidebar"] .sidebar-subtitle {
        color: #9ca3af;
        font-size: 0.75rem;
        text-align: center;
        margin-bottom: 2rem;
        padding: 0 1rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Radio Buttons Container - Navigation */
    [data-testid="stSidebar"] .stRadio {
        padding: 0 0.75rem;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #6b7280;
        font-weight: 600;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
        padding-left: 0.5rem;
        display: block;
    }
    
    [data-testid="stSidebar"] .stRadio > div {
        background: transparent;
        padding: 0;
        gap: 0.5rem;
    }
    
    /* Individual Radio Button Items */
    [data-testid="stSidebar"] .stRadio > div > label {
        background: rgba(255, 255, 255, 0.02);
        border-left: 3px solid transparent;
        padding: 0.875rem 1rem;
        border-radius: 8px;
        margin: 0 0 0.5rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        color: #d1d5db;
        font-weight: 500;
        font-size: 0.9375rem;
        position: relative;
        overflow: hidden;
    }
    
    /* Hover effect */
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(59, 130, 246, 0.15);
        border-left-color: #3b82f6;
        color: #ffffff;
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    /* Selected/Active state */
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        border-left-color: #3b82f6;
        color: #ffffff;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
    }
    
    /* Hide radio button circles */
    [data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"] > div:first-child {
        display: none;
    }
    
    /* Add icons to navigation items using pseudo-elements */
    [data-testid="stSidebar"] .stRadio > div > label:nth-child(1)::before {
        content: " ";
        margin-right: 0.5rem;
        font-size: 1.1rem;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:nth-child(2)::before {
        content: "";
        margin-right: 0.5rem;
        font-size: 1.1rem;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:nth-child(3)::before {
        content: " ";
        margin-right: 0.5rem;
        font-size: 1.1rem;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:nth-child(4)::before {
        content: "";
        margin-right: 0.5rem;
        font-size: 1.1rem;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:nth-child(5)::before {
        content: " ";
        margin-right: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Sidebar Footer */
    [data-testid="stSidebar"]::after {
        content: "";
        display: block;
        margin-top: auto;
        padding: 1.5rem 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: #6b7280;
        font-size: 0.75rem;
        text-align: center;
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
    
    /* Hide accidental second sidebar if rendered */
    [data-testid="stSidebar"]:nth-of-type(2) { display: none !important; }

    /* Hide Streamlit header/top bar */
    [data-testid="stHeader"] { display: none; }
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

st.sidebar.title("DataLab v 0.1")
st.sidebar.markdown('<p class="sidebar-subtitle">ML Workspace v0.1</p>', unsafe_allow_html=True)

phase = st.sidebar.radio(
    "WORKFLOW PHASES",
    ["Upload Data", "Preprocessing", "Visualization", "Model Training", "Deployment"],
    label_visibility="visible"
)


st.caption("Enterprise-grade machine learning workspace")

if phase == "Upload Data":
    upload.show()
elif phase == "Visualization":
    visualization.show()
elif phase == "Preprocessing":
    preprocessing.show()
elif phase == "Model Training":
    training.show()
elif phase == "Deployment":
    deployment.show()

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6b7280; padding: 2rem 0;'>
        <p><strong>DataLab Analytics Platform</strong> | Enterprise ML Workspace</p>
        <p style='font-size: 0.875rem; margin-top: 0.5rem;'>Built with Streamlit • Powered by AI</p>
    </div>
    """, 
    unsafe_allow_html=True
)