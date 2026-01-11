import streamlit as st
import pandas as pd

from preprocessing.api import PreprocessingAPI
from visalustsation import VisualizerManager, NumericVisualizer, CategoricalVisualizer, CorrelationVisualizer
from helpers.viz_utils import figs_to_html

def show():
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

            # Initialize exclusion lists with defaults
            exclude_from_scale = [col for col in numeric_cols if col == st.session_state.get('target_col')]
            exclude_from_encode = []
            exclude_from_missing = []

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

        # Column exclusion section
        with st.expander("Advanced: Exclude columns from preprocessing"):
            st.write("""
            Select columns that should NOT be preprocessed (e.g., target column, ID columns).
            These columns will be included in the output but won't be scaled, encoded, or modified.
            """)
            
            exclude_from_scale = st.multiselect(
                "Exclude from scaling:",
                options=numeric_cols,
                default=exclude_from_scale,
                help="These numeric columns won't be scaled"
            )
            
            exclude_from_encode = st.multiselect(
                "Exclude from encoding:",
                options=categorical_cols,
                default=exclude_from_encode,
                help="These categorical columns won't be encoded"
            )
            
            exclude_from_missing = st.multiselect(
                "Exclude from missing value handling:",
                options=df.columns.tolist(),
                default=exclude_from_missing,
                help="These columns won't have missing values imputed"
            )

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
                    user_options["exclude_from_scale"] = exclude_from_scale
                    user_options["exclude_from_encode"] = exclude_from_encode
                    user_options["exclude_from_missing"] = exclude_from_missing

                    # Calculate actual columns to process (excluding the specified ones)
                    numeric_to_scale = [col for col in numeric_cols if col not in exclude_from_scale]
                    categorical_to_encode = [col for col in categorical_cols if col not in exclude_from_encode]
                    columns_for_missing = [col for col in df.columns if col not in exclude_from_missing]

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
                        encode_categorical=user_options.get('encoder_type') if user_options.get('encoder_type') else None,
                        numeric_columns_to_scale=numeric_to_scale,
                        categorical_columns_to_encode=categorical_to_encode,
                        columns_to_handle_missing=columns_for_missing
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
                    images = figs_to_html(figs)
                    for item in images:
                        st.markdown(f"**{item['name']}**")
                        st.image(item['img'])

                except Exception as e:
                    st.error(f"Error: {str(e)}")
