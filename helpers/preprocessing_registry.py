from preprocessing.preprocessor import Preprocessor
from preprocessing.scaler import NumericScaler
from preprocessing.encoders import CategoricalEncoder
from preprocessing.missing_values import MissingValueHandler

def build_preprocessor(user_options):
    numeric_transformers = []
    categorical_transformers = []
    missing_handler = None

    # Handle missing values
    if "missing" in user_options:
        missing_strategy = user_options.get("missing_strategy", "impute")
        fill_strategy = user_options.get("fill_strategy", "mean")
        
        # FIX: For categorical columns, automatically use 'most_frequent' if fill_strategy is 'mean'
        # Don't pass categorical_fill_strategy to the constructor
        missing_handler = MissingValueHandler(
            strategy=missing_strategy, 
            fill_strategy=fill_strategy
        )

    # Numeric scaler
    if "scale" in user_options:
        scaler_type = user_options.get("scaler_type", "standard")
        numeric_transformers.append(NumericScaler(method=scaler_type))

    # Categorical encoder
    if "encode" in user_options:
        encoder_type = user_options.get("encoder_type", "onehot")
        categorical_transformers.append(CategoricalEncoder(method=encoder_type))

    return Preprocessor(
        numeric_transformers=numeric_transformers,
        categorical_transformers=categorical_transformers,
        missing_value_handler=missing_handler
    )