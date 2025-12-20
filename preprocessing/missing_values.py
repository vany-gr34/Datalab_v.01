import pandas as pd


def missing_values_summary(df):
   

    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent
    })

    return summary



def handle_missing_values(
    df,
    numeric_strategy="median",
    categorical_strategy="mode",
    constant_value="Unknown",
    drop_threshold=None
):


    df = df.copy()

    # Drop columns with too many missing values
    if drop_threshold is not None:
        missing_percent = (df.isnull().sum() / len(df)) * 100
        cols_to_drop = missing_percent[missing_percent > drop_threshold].index
        df.drop(columns=cols_to_drop, inplace=True)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    # Numeric columns
    for col in numeric_cols:
        if df[col].isnull().sum() == 0:
            continue

        if numeric_strategy == "mean":
            df[col].fillna(df[col].mean(), inplace=True)
        elif numeric_strategy == "median":
            df[col].fillna(df[col].median(), inplace=True)
        elif numeric_strategy == "drop":
            df.drop(columns=[col], inplace=True)
        else:
            raise ValueError("numeric_strategy must be 'mean', 'median', or 'drop'")

    # Categorical columns
    for col in categorical_cols:
        if df[col].isnull().sum() == 0:
            continue

        if categorical_strategy == "mode":
            df[col].fillna(df[col].mode()[0], inplace=True)
        elif categorical_strategy == "constant":
            df[col].fillna(constant_value, inplace=True)
        elif categorical_strategy == "drop":
            df.drop(columns=[col], inplace=True)
        else:
            raise ValueError("categorical_strategy must be 'mode', 'constant', or 'drop'")

    return df
