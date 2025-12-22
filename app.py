import streamlit as st
import pandas as pd
import numpy as np


# Make sure you have numpy imported
from sklearn.model_selection import train_test_split


from helpers.file_parser import parse_contents
from helpers.model_registry import CLASSIFIERS, REGRESSORS
from helpers.preprocessing_registry import build_preprocessor
from helpers.visualization_registry import build_visualizer

# Streamlit page config
st.set_page_config(
    page_title="🧪 Interactive Data Lab",
    page_icon="🧪",
    layout="wide"
)

# Custom CSS for vibrant theme (dark mode inspired)
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stSidebar .stRadio > div {
        background-color: #2A2A2A;
        color: #00FF00;
    }
    .stButton > button {
        background-color: #00FF00;
        color: #1E1E1E;
    }
    .stAlert {
        background-color: #2A2A2A;
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

st.sidebar.title("Navigation")
phase = st.sidebar.radio("Select Phase", ["Upload Data", " Visualization", "Preprocessing", " Model Training", "Deployment"])

st.title("DataLabV.01")




if phase == "Upload Data":
    st.header("Upload & View Data")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file:
        try:
            # Read CSV with explicit dtype specification for problem columns
            df = pd.read_csv(uploaded_file, low_memory=False)
            
            # ---- FIX: Ensure Arrow compatibility ----
            def make_arrow_compatible(df):
                for col in df.columns:
                    # Check if column is object/string type
                    if df[col].dtype == 'object':
                        # Check if it contains mixed types (numbers and strings)
                        try:
                            # Try to convert to numeric first
                            numeric_series = pd.to_numeric(df[col], errors='coerce')
                            # If all values can be converted to numeric
                            if numeric_series.notna().all():
                                df[col] = numeric_series.astype('float64')
                            else:
                                # Keep as string but ensure it's proper string
                                df[col] = df[col].astype(str)
                        except:
                            df[col] = df[col].astype(str)
                    
                    # Convert Int64 (nullable integer) to regular int or float
                    elif str(df[col].dtype) == 'Int64':
                        df[col] = df[col].astype('float64')
                    
                    # Convert any other nullable types
                    elif 'int' in str(df[col].dtype).lower():
                        df[col] = df[col].fillna(0).astype('int64')
                    
                    elif 'float' in str(df[col].dtype).lower():
                        df[col] = df[col].fillna(0.0).astype('float64')
                
                return df
            
            df = make_arrow_compatible(df)
            st.session_state.stored_data = df
            st.success(f"Uploaded {uploaded_file.name} successfully!")
            
        except Exception as e:
            st.error(f"Failed to read file! {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    if st.session_state.stored_data is not None:
        df = st.session_state.stored_data
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        st.subheader("Data Summary")
        st.write(df.describe(include='all'))
        
        st.write(f"Data Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
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
        st.info("Please upload a CSV file.")
elif phase == "Preprocessing":
    st.header("Preprocessing")

    if st.session_state.stored_data is None:
        st.warning("Upload data first!")
    else:
        df = st.session_state.stored_data.copy()
        
        # Show original data info
        st.subheader("Original Data")
        st.write(f"Shape: {df.shape}")
        
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
                st.success("Column types updated!")
        
        # Apply preprocessing
        if st.button("Apply Preprocessing ", type="primary"):
            if not preprocessing_options:
                st.warning("Please select at least one preprocessing option!")
            else:
                try:
                    user_options.update({step: True for step in preprocessing_options})
                    
                    # Add column information to options
                    user_options["numeric_columns"] = numeric_cols
                    user_options["categorical_columns"] = categorical_cols
                    
                    # Build preprocessor
                    preprocessor = build_preprocessor(user_options)
                    
                    with st.spinner("Applying preprocessing..."):
                        transformed_df = preprocessor.fit_transform(df)
                        
                        try:
                             feature_names = preprocessor.get_feature_names_out()
                             transformed_df = pd.DataFrame(transformed_df, columns=feature_names)
                        except Exception:
                             transformed_df = pd.DataFrame(transformed_df)
                    # Store results
                    st.session_state.preprocessed_data = transformed_df
                    st.success("✅ Preprocessing successful!")
                    
                    # Show results
                    st.subheader("Processed Data")
                    st.write(f"Shape: {transformed_df.shape}")
                    st.dataframe(transformed_df.head())
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

elif phase == " Model Training":
    st.header(" Model Training")
    if st.session_state.stored_data is None:
        st.warning("Upload data first!")
    else:
        df = st.session_state.preprocessed_data if st.session_state.preprocessed_data is not None else st.session_state.stored_data
        st.session_state.problem_type = st.selectbox(
            "Select Problem Type:", 
            ["regression", "classification"], 
            index=0 if st.session_state.problem_type == "regression" else 1,
            format_func=lambda x: f"{x.capitalize()} {'📈' if x == 'regression' else '🏷️'}"
        )
        model_options = REGRESSORS if st.session_state.problem_type == "regression" else CLASSIFIERS
        selected_models = st.multiselect("Select Models:", list(model_options.keys()), default=list(model_options.keys())[:2])
        st.session_state.target_col = st.selectbox("Select Target Column:", df.columns, index=df.columns.get_loc(st.session_state.target_col) if st.session_state.target_col in df.columns else 0)
        
        if st.button("Run Models 🚀"):
            target_col = st.session_state.target_col
            if target_col not in df.columns:
                st.error("Invalid target column selected. ❌")
            else:
                X = df.drop(columns=[target_col])
                y = df[target_col]
                X = pd.get_dummies(X)  # Quick categorical handling if needed
                X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
                
                st.subheader("📈 Model Results")
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
                        except AttributeError:
                            # Fallback to standard fit/score if .run not available
                            model.fit(X_train, y_train)
                            score = model.score(X_valid, y_valid)
                            results.append({
                                "Model": model_name,
                                "Score": round(score, 4)
                            })
                        except Exception as e:
                            st.error(f"{model_name}: Error - {str(e)}")
                
                if results:
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    if st.session_state.problem_type == "regression":
                        best_row = results_df.sort_values("R2", ascending=False).iloc[0]
                        st.success(f"🏆 Best Model: {best_row['Model']} (R2 = {best_row['R2']})")
                    else:
                        best_row = results_df.sort_values("Accuracy", ascending=False).iloc[0]
                        st.success(f"🏆 Best Model: {best_row['Model']} (Accuracy = {best_row['Accuracy']})")
                else:
                    st.info("No models ran successfully.")

elif phase == " Deployment":
    st.header(" Deployment")
    st.info("Deployment phase coming soon! 🔮 (Future feature)")

st.markdown("Modular ML Data Lab")