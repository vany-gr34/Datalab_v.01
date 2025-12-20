import os
import joblib
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder
)



def encode_categorical_features(df, method="onehot"):
    

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    if method == "onehot":
        df = pd.get_dummies(df, columns=categorical_cols)

    elif method == "label":
        encoder = LabelEncoder()
        for col in categorical_cols:
            df[col] = encoder.fit_transform(df[col])

    else:
        raise ValueError("Encoding method must be 'onehot' or 'label'")

    return df



def normalize_features(
    df,
    method="standard",
    save_scaler=True,
    scaler_dir="scalers",
    scaler_name="scaler.pkl"
):
    

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_cols) == 0:
        return df

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Normalization method must be 'standard' or 'minmax'")

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    if save_scaler:
        os.makedirs(scaler_dir, exist_ok=True)
        scaler_path = os.path.join(scaler_dir, scaler_name)
        joblib.dump(scaler, scaler_path)

    return df



def preprocessing_pipeline(
    df,
    encoding_method="onehot",
    normalization_method="standard"
):

    df = encode_categorical_features(df, method=encoding_method)
    df = normalize_features(df, method=normalization_method)

    return df
