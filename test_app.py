import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    # Test imports
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split

    from helpers.file_parser import parse_contents
    from helpers.model_registry import CLASSIFIERS, REGRESSORS
    from preprocessing.api import PreprocessingAPI
    from visalustsation import VisualizerManager, NumericVisualizer, CategoricalVisualizer, CorrelationVisualizer
    from ingestion import ingest
    from ingestion.base_ingestor import IngestionError

    print("All imports successful")

    # Test basic functionality
    # Create sample data
    df = pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5],
        'categorical_col': ['A', 'B', 'A', 'B', 'A'],
        'target': [0, 1, 0, 1, 0]
    })

    # Test visualization logic
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    # Test plot generation
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='numeric_col', ax=ax)
    plt.close(fig)
    print("Plot generation successful")

    print("Basic functionality test passed")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
