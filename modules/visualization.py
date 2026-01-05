import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from visalustsation import VisualizerManager, NumericVisualizer, CategoricalVisualizer, CorrelationVisualizer
from helpers.viz_utils import figs_to_html

def show():
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
