import streamlit as st
import pandas as pd
import base64
from io import BytesIO

from helpers.file_parser import parse_contents
from ingestion import ingest
from ingestion.base_ingestor import IngestionError

def show():
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

       api_url = st.text_input("API URL", placeholder="https://jsonplaceholder.typicode.com/posts")
       method = st.selectbox("HTTP Method", ["GET", "POST"])

       auth_ui = st.selectbox("Authentication", ["None", "API Key", "Bearer Token", "Basic Auth"])

       auth_type = None
       auth_config = {}
       if auth_ui == "API Key":
          auth_type = "api_key"
          auth_config = {
            "key_name": st.text_input("API Key Header", value="X-API-Key"),
            "key_value": st.text_input("API Key", type="password")
        }

       elif auth_ui == "Bearer Token":
          auth_type = "bearer"
          auth_config = {
            "token": st.text_input("Bearer Token", type="password")
        }

       elif auth_ui == "Basic Auth":
          auth_type = "basic"
          auth_config = {
            "username": st.text_input("Username"),
            "password": st.text_input("Password", type="password")
        }

       json_path = st.text_input("JSON Path (optional)", placeholder="data.items")

       if st.button("Fetch Data"):
           try:
               with st.spinner("Fetching data from API..."):
                   dataset = ingest("api", {
                       "url": api_url,
                       "method": method,
                       "auth_type": auth_type,
                       "auth_config": auth_config,
                       "json_path": json_path or None,
                   })

                   st.session_state.stored_data = dataset.data
                   st.session_state.current_dataset = dataset

                   st.success("API data ingested successfully")
                   st.info(f"Dataset ID: {dataset.id}")

           except IngestionError as e:
               st.error(f"Ingestion failed: {e}")
           except Exception as e:
               st.error(f"Unexpected error: {e}")


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
