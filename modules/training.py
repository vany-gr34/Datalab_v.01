import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib

from helpers.model_registry import get_available_models
from helpers.model_recommender import recommend_models, recommend_best_model

def show():
    st.header("Model Training & Selection")

    if st.session_state.stored_data is None:
        st.warning("Please upload data first.")
        return

    if st.session_state.preprocessed_data is None:
        st.info("Using raw data for training. Consider preprocessing for better results.")
        df = st.session_state.stored_data
    else:
        df = st.session_state.preprocessed_data

    # Target column selection
    if st.session_state.target_col is None and st.session_state.problem_type != "clustering":
        st.subheader("Target Column Selection")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        target_options = numeric_cols + categorical_cols
        if target_options:
            selected_target = st.selectbox("Select target column for prediction:", target_options)

            # Problem type selection
            st.subheader("Problem Type Selection")
            problem_type_options = ["Auto-detect", "Classification", "Regression", "Clustering"]
            selected_problem_type = st.radio(
                "Choose the type of machine learning problem:",
                problem_type_options,
                help="Auto-detect will analyze your target column to determine the problem type. You can also manually select classification, regression, or clustering."
            )

            if st.button("Set Target Column & Problem Type"):
                # Handle clustering case - no target column needed
                if selected_problem_type == "Clustering":
                    st.session_state.target_col = None  # No target for clustering
                    st.session_state.problem_type = "clustering"
                    st.success("Problem type set to: Clustering (no target column needed)")
                    st.rerun()
                    return

                st.session_state.target_col = selected_target

                # Determine problem type
                if selected_problem_type == "Auto-detect":
                    # Smart auto-detection based on data characteristics
                    target_series = df[selected_target].dropna()
                    unique_vals = target_series.nunique()
                    total_samples = len(target_series)
                    cardinality_ratio = unique_vals / total_samples if total_samples > 0 else 0

                    # Classification criteria:
                    # 1. Low unique values (typically 2-20 classes)
                    # 2. Categorical/object type
                    # 3. Low cardinality ratio (much fewer unique values than samples)

                    if selected_target in numeric_cols:
                        # For numeric columns, check unique value count
                        # Low unique values -> classification, High unique values -> regression
                        if unique_vals <= 20 and cardinality_ratio < 0.1:
                            st.session_state.problem_type = "classification"
                            auto_msg = " (Auto-detected as classification - low unique numeric values)"
                        else:
                            st.session_state.problem_type = "regression"
                            auto_msg = " (Auto-detected as regression - continuous numeric values)"
                    else:
                        # For categorical/object columns
                        if unique_vals <= 20:
                            st.session_state.problem_type = "classification"
                            auto_msg = " (Auto-detected as classification - categorical target)"
                        else:
                            st.session_state.problem_type = "regression"
                            auto_msg = " (Auto-detected as regression - too many unique categories)"

                elif selected_problem_type == "Classification":
                    st.session_state.problem_type = "classification"
                    auto_msg = " (Manually set as classification)"
                else:  # Regression
                    st.session_state.problem_type = "regression"
                    auto_msg = " (Manually set as regression)"

                st.success(f"Target column set to: {selected_target}{auto_msg}")
                st.rerun()
        else:
            st.error("No suitable columns found for target selection.")
        return

    # Pre-training Model Recommendations
    st.subheader("Model Recommendations")
    recommended_models = recommend_models(df, st.session_state.problem_type)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("**Recommended models based on your dataset characteristics:**")
        for i, model in enumerate(recommended_models, 1):
            st.write(f"{i}. {model}")
    with col2:
        st.info(f"Dataset: {len(df)} samples, {len(df.columns)-1} features\nProblem: {st.session_state.problem_type.capitalize()}")

    # Training interface
    st.subheader("Model Training")

    # Model selection
    available_models = get_available_models(st.session_state.problem_type)
    selected_models = st.multiselect(
        "Select models to train:",
        list(available_models.keys()),
        default=list(available_models.keys())[:3],  # Default to first 3 models
        help="Choose which models to train and compare"
    )

    if not selected_models:
        st.warning("Please select at least one model to train.")
        return

    # Training configuration - different for clustering
    if st.session_state.problem_type == "clustering":
        col1, col2 = st.columns(2)
        with col1:
            random_state = st.number_input("Random State", value=42, help="Random seed for reproducibility")
        with col2:
            st.info("Clustering doesn't require train-test split")

        # Feature selection for clustering
        all_features = df.columns.tolist()  # All columns can be features for clustering
        selected_features = st.multiselect(
            "Select features for clustering:",
            all_features,
            default=all_features,
            help="Choose which columns to use as features for clustering"
        )

        if not selected_features:
            st.warning("Please select at least one feature.")
            return

        # Prepare data for clustering
        X = df[selected_features]
        X = X.fillna(X.mean(numeric_only=True))  # Handle missing values
    else:
        # Regular supervised learning configuration
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Size (%)", 10, 50, 20, help="Percentage of data to use for testing")
        with col2:
            random_state = st.number_input("Random State", value=42, help="Random seed for reproducibility")

        # Feature selection
        all_features = [col for col in df.columns if col != st.session_state.target_col]
        selected_features = st.multiselect(
            "Select features for training:",
            all_features,
            default=all_features,
            help="Choose which columns to use as features"
        )

        if not selected_features:
            st.warning("Please select at least one feature.")
            return

        # Prepare data
        X = df[selected_features]
        y = df[st.session_state.target_col]

        # Handle missing values
        X = X.fillna(X.mean(numeric_only=True))
        if st.session_state.problem_type == "classification":
            y = y.fillna(y.mode().iloc[0] if not y.mode().empty else "Unknown")
        else:
            y = y.fillna(y.mean())

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state
        )

    if st.button("🚀 Train Models", type="primary"):
        st.session_state.trained_models = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name}...")
            progress_bar.progress((i + 1) / len(selected_models))

            try:
                model_class = available_models[model_name]
                model = model_class()

                # Run model using its `run` method which returns (y_pred, metrics)
                if st.session_state.problem_type == "clustering":
                    y_pred, model_metrics = model.run(X)
                else:
                    y_pred, model_metrics = model.run(X_train, X_test, y_train, y_test)

                # Normalize metrics for the UI so downstream code can rely on consistent keys
                if st.session_state.problem_type == "classification":
                    metrics = {
                        "Accuracy": model_metrics.get("accuracy") or model_metrics.get("Accuracy"),
                        "Precision": model_metrics.get("precision") or model_metrics.get("Precision"),
                        "Recall": model_metrics.get("recall") or model_metrics.get("Recall"),
                        "F1-Score": model_metrics.get("f1") or model_metrics.get("F1-Score") or model_metrics.get("f1-score")
                    }
                elif st.session_state.problem_type == "clustering":
                    metrics = {
                        "Silhouette": model_metrics.get("silhouette") or model_metrics.get("Silhouette"),
                        "DaviesBouldin": model_metrics.get("davies_bouldin") or model_metrics.get("DaviesBouldin")
                    }
                else:
                    mse = model_metrics.get("MSE") or model_metrics.get("mse")
                    rmse = model_metrics.get("RMSE") or model_metrics.get("rmse") or (np.sqrt(mse) if mse is not None else None)
                    r2 = model_metrics.get("R2") or model_metrics.get("r2")
                    metrics = {
                        "MSE": mse,
                        "RMSE": rmse,
                        "R2": r2
                    }

                # Store trained model - different for clustering
                if st.session_state.problem_type == "clustering":
                    st.session_state.trained_models[model_name] = {
                        "model": model,
                        "features": selected_features,
                        "target_col": None,  # No target for clustering
                        "problem_type": st.session_state.problem_type,
                        "metrics": metrics,
                        "training_info": {
                            "random_state": random_state,
                            "n_features": len(selected_features),
                            "n_samples": len(X)
                        }
                    }
                else:
                    st.session_state.trained_models[model_name] = {
                        "model": model,
                        "features": selected_features,
                        "target_col": st.session_state.target_col,
                        "problem_type": st.session_state.problem_type,
                        "metrics": metrics,
                        "training_info": {
                            "test_size": test_size/100,
                            "random_state": random_state,
                            "n_features": len(selected_features),
                            "n_samples": len(X_train)
                        }
                    }

            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")

        progress_bar.empty()
        status_text.empty()
        st.success("Model training completed!")

    # Display trained models
    if st.session_state.trained_models:
        if st.session_state.problem_type == "classification":
            display_classification_results()
        elif st.session_state.problem_type == "clustering":
            display_clustering_results()
        else:
            display_regression_results()


def display_classification_results():
    """Display classification model results with detailed metrics and analysis."""
    st.subheader("📊 Classification Model Results")

    # Dataset analysis
    target_col = st.session_state.target_col
    df = st.session_state.stored_data if st.session_state.preprocessed_data is None else st.session_state.preprocessed_data
    target_values = df[target_col].dropna()
    n_classes = target_values.nunique()
    class_counts = target_values.value_counts()

    # Class distribution analysis
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Classes", n_classes)
    with col2:
        st.metric("Total Samples", len(target_values))
    with col3:
        majority_class_pct = (class_counts.max() / len(target_values)) * 100
        st.metric("Majority Class %", f"{majority_class_pct:.1f}%")

    # Class imbalance warning
    if majority_class_pct > 70:
        st.warning("⚠️ **Class Imbalance Detected**: The majority class represents more than 70% of the data. Consider using techniques like SMOTE, class weighting, or collecting more balanced data.")

    # Binary vs Multi-class
    if n_classes == 2:
        st.info("🔢 **Binary Classification**: This is a binary classification problem with 2 classes.")
    else:
        st.info(f"🔢 **Multi-class Classification**: This is a multi-class classification problem with {n_classes} classes.")

    # Model comparison table
    st.subheader("Model Performance Comparison")
    comparison_data = []
    for model_name, model_data in st.session_state.trained_models.items():
        row = {"Model": model_name}
        row.update(model_data["metrics"])
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.style.highlight_max(axis=0), use_container_width=True)

    # Best model recommendation with explanation
    best_recommendation = recommend_best_model(st.session_state.trained_models, "classification")
    if best_recommendation:
        st.success(f"🏆 **Recommended Model: {best_recommendation['model_name']}**")
        st.info(best_recommendation['justification'])

    # Detailed model analysis
    st.subheader("Model Details & Analysis")
    for model_name, model_data in st.session_state.trained_models.items():
        with st.expander(f"🔍 {model_name} - Detailed Analysis"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{model_data['metrics']['Accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{model_data['metrics']['Precision']:.4f}")
            with col3:
                st.metric("Recall", f"{model_data['metrics']['Recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{model_data['metrics']['F1-Score']:.4f}")

            # Performance interpretation
            accuracy = model_data['metrics']['Accuracy']
            if accuracy >= 0.9:
                st.success("Excellent performance! This model shows strong predictive capabilities.")
            elif accuracy >= 0.8:
                st.info("Good performance. The model demonstrates reliable predictions.")
            elif accuracy >= 0.7:
                st.warning("Moderate performance. Consider hyperparameter tuning or feature engineering.")
            else:
                st.error("Poor performance. This model may not be suitable for this dataset.")

            # Training info
            st.write(f"**Training Details:** {model_data['training_info']['n_samples']} samples, {model_data['training_info']['n_features']} features")
            st.write(f"**Features Used:** {', '.join(model_data['features'])}")


def display_clustering_results():
    """Display clustering model results with detailed metrics and analysis."""
    st.subheader("🎯 Clustering Model Results")

    # Dataset analysis
    df = st.session_state.stored_data if st.session_state.preprocessed_data is None else st.session_state.preprocessed_data

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Number of Features", len(df.columns))
    with col3:
        st.metric("Data Shape", f"{len(df)} × {len(df.columns)}")

    # Model comparison table
    st.subheader("Model Performance Comparison")
    comparison_data = []
    for model_name, model_data in st.session_state.trained_models.items():
        row = {"Model": model_name}
        row.update(model_data["metrics"])
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    # For clustering, higher Silhouette and Davies-Bouldin scores are better
    st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Silhouette']).highlight_min(axis=0, subset=['DaviesBouldin']), use_container_width=True)

    # Best model recommendation with explanation
    best_recommendation = recommend_best_model(st.session_state.trained_models, "clustering")
    if best_recommendation:
        st.success(f"🏆 **Recommended Model: {best_recommendation['model_name']}**")
        st.info(best_recommendation['justification'])

    # Detailed model analysis
    st.subheader("Model Details & Analysis")
    for model_name, model_data in st.session_state.trained_models.items():
        with st.expander(f"🔍 {model_name} - Detailed Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                silhouette = model_data['metrics'].get('Silhouette')
                if silhouette is not None:
                    st.metric("Silhouette Score", f"{silhouette:.4f}")
                else:
                    st.metric("Silhouette Score", "N/A")
            with col2:
                db = model_data['metrics'].get('DaviesBouldin')
                if db is not None:
                    st.metric("Davies-Bouldin Score", f"{db:.4f}")
                else:
                    st.metric("Davies-Bouldin Score", "N/A")

            # Performance interpretation
            silhouette = model_data['metrics'].get('Silhouette')
            if silhouette is not None:
                if silhouette >= 0.7:
                    st.success("Excellent clustering! Well-separated and cohesive clusters.")
                elif silhouette >= 0.5:
                    st.info("Good clustering. Reasonably separated clusters.")
                elif silhouette >= 0.25:
                    st.warning("Moderate clustering. Consider different preprocessing or algorithms.")
                else:
                    st.error("Poor clustering. Data may not have clear cluster structure.")
            else:
                st.warning("Silhouette score unavailable (likely single cluster or other issue).")

            # Training info
            st.write(f"**Training Details:** {model_data['training_info']['n_samples']} samples, {model_data['training_info']['n_features']} features")
            st.write(f"**Features Used:** {', '.join(model_data['features'])}")


def display_regression_results():
    """Display regression model results with detailed metrics and analysis."""
    st.subheader("📈 Regression Model Results")

    # Dataset analysis
    target_col = st.session_state.target_col
    df = st.session_state.stored_data if st.session_state.preprocessed_data is None else st.session_state.preprocessed_data
    target_values = df[target_col].dropna()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target Mean", f"{target_values.mean():.2f}")
    with col2:
        st.metric("Target Std", f"{target_values.std():.2f}")
    with col3:
        st.metric("Target Range", f"{target_values.min():.2f} - {target_values.max():.2f}")

    # Model comparison table
    st.subheader("Model Performance Comparison")
    comparison_data = []
    for model_name, model_data in st.session_state.trained_models.items():
        row = {"Model": model_name}
        row.update(model_data["metrics"])
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    # For regression, higher R2 is better (green), lower MSE/RMSE is better (green)
    st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['R2']).highlight_min(axis=0, subset=['MSE', 'RMSE']), use_container_width=True)

    # Best model recommendation with explanation
    best_recommendation = recommend_best_model(st.session_state.trained_models, "regression")
    if best_recommendation:
        st.success(f"🏆 **Recommended Model: {best_recommendation['model_name']}**")
        st.info(best_recommendation['justification'])

    # Detailed model analysis
    st.subheader("Model Details & Analysis")
    for model_name, model_data in st.session_state.trained_models.items():
        with st.expander(f"🔍 {model_name} - Detailed Analysis"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MSE", f"{model_data['metrics']['MSE']:.4f}")
            with col2:
                st.metric("RMSE", f"{model_data['metrics']['RMSE']:.4f}")
            with col3:
                st.metric("R² Score", f"{model_data['metrics']['R2']:.4f}")

            # Performance interpretation
            r2 = model_data['metrics']['R2']
            if r2 >= 0.8:
                st.success("Excellent performance! This model explains most of the variance in the target variable.")
            elif r2 >= 0.6:
                st.info("Good performance. The model captures a reasonable amount of variance.")
            elif r2 >= 0.3:
                st.warning("Moderate performance. Consider feature engineering or more complex models.")
            else:
                st.error("Poor performance. This model may not be suitable for this dataset.")

            # Training info
            st.write(f"**Training Details:** {model_data['training_info']['n_samples']} samples, {model_data['training_info']['n_features']} features")
            st.write(f"**Features Used:** {', '.join(model_data['features'])}")
