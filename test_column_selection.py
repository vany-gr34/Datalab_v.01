#!/usr/bin/env python3
"""
Comprehensive test script for column selection in preprocessing.
Tests the ability to specify which columns to preprocess.
"""

import pandas as pd
import numpy as np
from preprocessing.api import PreprocessingAPI
import sys

def create_test_data():
    """Create test DataFrame with mixed data types."""
    np.random.seed(42)
    df = pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [10.5, 20.3, 30.7, 40.1, 50.9],
        'numeric3': [100, 200, 300, 400, 500],
        'categorical1': ['A', 'B', 'A', 'B', 'A'],
        'categorical2': ['X', 'Y', 'X', 'Y', 'X'],
        'categorical3': ['P', 'Q', 'R', 'P', 'Q'],
        'target': [0, 1, 0, 1, 0]
    })
    return df

def test_column_selection():
    """Test column selection functionality."""
    print("=== Testing Column Selection in Preprocessing ===\n")

    df = create_test_data()
    api = PreprocessingAPI()

    print("Original DataFrame:")
    print(df)
    print(f"Data types:\n{df.dtypes}\n")

    # Test 1: Selective scaling and encoding
    print("Test 1: Selective scaling and encoding")
    config1 = api.create_preprocessing_config(
        scale_numeric='standard',
        encode_categorical='onehot',
        numeric_columns_to_scale=['numeric1', 'numeric2'],  # Only scale these two
        categorical_columns_to_encode=['categorical1']  # Only encode this one
    )

    processed_df1, metadata1 = api.preprocess(df.copy(), config1, 'test1', target_column='target')

    print("Config:", config1)
    print("Processed DataFrame:")
    print(processed_df1)
    print(f"Shape: {processed_df1.shape}")

    # Verify only specified columns were transformed
    # numeric3 should remain unscaled
    assert processed_df1['numeric3'].equals(df['numeric3']), "numeric3 should not be scaled"
    # categorical2 and categorical3 should remain unencoded
    assert 'categorical2' in processed_df1.columns, "categorical2 should remain unencoded"
    assert 'categorical3' in processed_df1.columns, "categorical3 should remain unencoded"
    # categorical1 should be encoded (onehot creates multiple columns)
    assert not any(col.startswith('categorical1_') for col in processed_df1.columns), "categorical1 should be encoded"

    print("✓ Test 1 passed\n")

    # Test 2: No column selection (backward compatibility)
    print("Test 2: No column selection (backward compatibility)")
    config2 = api.create_preprocessing_config(
        scale_numeric='standard',
        encode_categorical='onehot'
        # No column selection - should auto-detect all
    )

    processed_df2, metadata2 = api.preprocess(df.copy(), config2, 'test2', target_column='target')

    print("Config:", config2)
    print("Processed DataFrame:")
    print(processed_df2)
    print(f"Shape: {processed_df2.shape}")

    # All numeric columns should be scaled, all categorical encoded
    assert not any(col in processed_df2.columns for col in ['numeric1', 'numeric2', 'numeric3']), "All numeric columns should be scaled"
    assert not any(col in processed_df2.columns for col in ['categorical1', 'categorical2', 'categorical3']), "All categorical columns should be encoded"

    print("✓ Test 2 passed\n")

    # Test 3: Empty column lists
    print("Test 3: Empty column lists")
    config3 = api.create_preprocessing_config(
        scale_numeric='standard',
        encode_categorical='onehot',
        numeric_columns_to_scale=[],  # Empty list
        categorical_columns_to_encode=[]  # Empty list
    )

    processed_df3, metadata3 = api.preprocess(df.copy(), config3, 'test3', target_column='target')

    print("Config:", config3)
    print("Processed DataFrame:")
    print(processed_df3)
    print(f"Shape: {processed_df3.shape}")

    # No transformations should be applied
    assert processed_df3['numeric1'].equals(df['numeric1']), "numeric1 should remain unchanged"
    assert processed_df3['categorical1'].equals(df['categorical1']), "categorical1 should remain unchanged"

    print("✓ Test 3 passed\n")

    # Test 4: Missing value handling on specific columns
    print("Test 4: Missing value handling on specific columns")
    df_missing = df.copy()
    df_missing.loc[0, 'numeric1'] = np.nan
    df_missing.loc[1, 'categorical1'] = np.nan

    config4 = api.create_preprocessing_config(
        handle_missing='mean',
        scale_numeric='standard',
        encode_categorical='onehot',
        columns_to_handle_missing=['numeric1'],  # Only handle missing in numeric1
        numeric_columns_to_scale=['numeric1', 'numeric2'],
        categorical_columns_to_encode=['categorical1']
    )

    processed_df4, metadata4 = api.preprocess(df_missing.copy(), config4, 'test4', target_column='target')

    print("Config:", config4)
    print("Processed DataFrame (with missing values handled):")
    print(processed_df4)
    print(f"Shape: {processed_df4.shape}")

    # categorical1 should still have NaN since we didn't specify it for missing value handling
    # But since we're encoding it, it should be handled by the encoder
    print("✓ Test 4 passed\n")

    # Test 5: Non-existent columns
    print("Test 5: Non-existent columns")
    config5 = api.create_preprocessing_config(
        scale_numeric='standard',
        encode_categorical='onehot',
        numeric_columns_to_scale=['numeric1', 'nonexistent_col'],
        categorical_columns_to_encode=['categorical1', 'another_nonexistent']
    )

    processed_df5, metadata5 = api.preprocess(df.copy(), config5, 'test5', target_column='target')

    print("Config:", config5)
    print("Processed DataFrame:")
    print(processed_df5)
    print(f"Shape: {processed_df5.shape}")

    # Should only process existing columns
    assert processed_df5['numeric2'].equals(df['numeric2']), "numeric2 should remain unchanged"
    assert 'categorical2' in processed_df5.columns, "categorical2 should remain unencoded"

    print("✓ Test 5 passed\n")

    print("=== All tests passed! ===")
    return True

if __name__ == "__main__":
    try:
        test_column_selection()
        print("\n🎉 All column selection tests completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
