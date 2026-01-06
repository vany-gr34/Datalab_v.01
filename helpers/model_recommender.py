import numpy as np
import pandas as pd

def recommend_models(df, problem_type):
    """
    Returns a list of recommended models based on problem type and dataset characteristics.
    Uses intelligent scoring based on dataset properties.
    """
    n_samples = len(df)
    n_features = len(df.columns) - 1  # Excluding target column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    n_numeric = len(numeric_cols)
    n_categorical = len(categorical_cols)

    # Determine dataset size
    if n_samples < 1000:
        dataset_size = 'small'
    elif n_samples < 10000:
        dataset_size = 'medium'
    else:
        dataset_size = 'large'

    # Determine feature complexity
    if n_features < 10:
        feature_complexity = 'low'
    else:
        feature_complexity = 'high'

    # Determine feature type dominance
    if n_categorical > n_numeric:
        feature_type = 'categorical'
    elif n_numeric > n_categorical:
        feature_type = 'numeric'
    else:
        feature_type = 'mixed'

    # Additional dataset characteristics
    has_missing = df.isnull().any().any()
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
    
    target_col = None
    for col in df.columns:
        if col not in numeric_cols and col not in categorical_cols:
            continue
        target_col = col
        break

    # Class imbalance check for classification
    class_imbalance = False
    imbalance_ratio = 1.0
    if problem_type == 'classification' and target_col:
        target_values = df[target_col].dropna()
        if len(target_values) > 0:
            class_counts = target_values.value_counts()
            majority_pct = class_counts.max() / len(target_values)
            class_imbalance = majority_pct > 0.7
            imbalance_ratio = majority_pct

    # Outlier check for regression
    has_outliers = False
    outlier_pct = 0
    if problem_type == 'regression' and target_col and target_col in numeric_cols:
        target_values = df[target_col].dropna()
        if len(target_values) > 0:
            Q1 = target_values.quantile(0.25)
            Q3 = target_values.quantile(0.75)
            IQR = Q3 - Q1
            outlier_threshold = 1.5 * IQR
            outliers = ((target_values < (Q1 - outlier_threshold)) | (target_values > (Q3 + outlier_threshold)))
            outlier_pct = outliers.sum() / len(target_values)
            has_outliers = outlier_pct > 0.05  # More than 5% outliers

    # Base recommendations by problem type
    if problem_type == 'classification':
        all_models = {
            'Logistic Regression': {'simple': True, 'large_scale': False, 'categorical': False, 'imbalanced': False},
            'KNN': {'simple': True, 'large_scale': False, 'categorical': False, 'imbalanced': False},
            'Decision Tree': {'simple': True, 'large_scale': True, 'categorical': True, 'imbalanced': False},
            'Naive Bayes': {'simple': True, 'large_scale': True, 'categorical': True, 'imbalanced': False},
            'Random Forest': {'simple': False, 'large_scale': True, 'categorical': True, 'imbalanced': True},
            'SVM': {'simple': False, 'large_scale': False, 'categorical': False, 'imbalanced': True},
            'XGBoost': {'simple': False, 'large_scale': True, 'categorical': True, 'imbalanced': True},
            'Neural Network': {'simple': False, 'large_scale': True, 'categorical': False, 'imbalanced': True}
        }
    elif problem_type == 'regression':
        all_models = {
            'Linear Regression': {'simple': True, 'large_scale': False, 'categorical': False, 'robust': False},
            'Ridge Regression': {'simple': True, 'large_scale': False, 'categorical': False, 'robust': True},
            'KNN Regressor': {'simple': True, 'large_scale': False, 'categorical': False, 'robust': False},
            'Decision Tree Regressor': {'simple': True, 'large_scale': True, 'categorical': True, 'robust': False},
            'Random Forest Regressor': {'simple': False, 'large_scale': True, 'categorical': True, 'robust': True},
            'XGBoost Regressor': {'simple': False, 'large_scale': True, 'categorical': True, 'robust': True},
            'SVR': {'simple': False, 'large_scale': False, 'categorical': False, 'robust': True},
            'Lasso Regressor': {'simple': True, 'large_scale': False, 'categorical': False, 'robust': True}
        }
    else:
        return []

    # Score each model based on dataset characteristics
    model_scores = {}
    
    for model_name, properties in all_models.items():
        score = 0
        reasons = []
        
        # Dataset size scoring
        if dataset_size == 'small':
            if properties.get('simple', False):
                score += 3
                reasons.append('good for small datasets')
            else:
                score -= 1
        elif dataset_size == 'large':
            if properties.get('large_scale', False):
                score += 3
                reasons.append('handles large datasets well')
        else:  # medium
            score += 1  # All models work reasonably on medium datasets
        
        # Feature type scoring
        if feature_type == 'categorical' and properties.get('categorical', False):
            score += 2
            reasons.append('handles categorical features')
        elif feature_type == 'numeric' and not properties.get('categorical', False):
            score += 1
        
        # Feature complexity scoring
        if feature_complexity == 'high':
            if properties.get('large_scale', False):
                score += 2
                reasons.append('handles high dimensionality')
        else:
            if properties.get('simple', False):
                score += 1
        
        # Class imbalance scoring (classification only)
        if problem_type == 'classification' and class_imbalance:
            if properties.get('imbalanced', False):
                score += 2
                reasons.append('handles class imbalance')
            else:
                score -= 1
        
        # Outlier robustness scoring (regression only)
        if problem_type == 'regression' and has_outliers:
            if properties.get('robust', False):
                score += 2
                reasons.append('robust to outliers')
        
        # Missing values penalty
        if has_missing and missing_pct > 0.1:
            if model_name in ['Random Forest', 'XGBoost', 'Random Forest Regressor', 'XGBoost Regressor', 'Decision Tree', 'Decision Tree Regressor']:
                score += 1
                reasons.append('handles missing values')
            else:
                score -= 1
        
        model_scores[model_name] = {'score': score, 'reasons': reasons}
    
    # Sort models by score in descending order
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    recommendations = [model_name for model_name, _ in sorted_models[:5]]
    
    # Ensure we have at least 3 recommendations
    if len(recommendations) < 3:
        recommendations = list(all_models.keys())[:3]
    
    return recommendations[:5]  # Limit to top 5 recommendations


def recommend_best_model(trained_models, problem_type):
    """
    Recommends the best model from trained models with explanation.

    Args:
        trained_models: dict of model_name -> {'metrics': dict, 'problem_type': str, ...}
        problem_type: 'classification' or 'regression'

    Returns:
        dict: {'model_name': str, 'score': float, 'justification': str}
    """
    if not trained_models:
        return None

    best_model = None
    best_score = -float('inf') if problem_type == 'classification' else float('inf')
    metric_key = 'Accuracy' if problem_type == 'classification' else 'R2'

    for model_name, model_data in trained_models.items():
        score = model_data['metrics'].get(metric_key, 0)

        if problem_type == 'classification':
            if score > best_score:
                best_score = score
                best_model = model_name
        else:  # regression
            if score > best_score:  # Higher R2 is better
                best_score = score
                best_model = model_name

    if best_model is None:
        return None

    # Generate justification
    justification = f"Selected {best_model} as the best performing model with {metric_key} of {best_score:.4f}."

    if problem_type == 'classification':
        if best_score > 0.9:
            justification += " This model shows excellent performance on the dataset."
        elif best_score > 0.8:
            justification += " This model demonstrates strong predictive capabilities."
        else:
            justification += " Consider further tuning or feature engineering to improve performance."
    else:
        if best_score > 0.8:
            justification += " This model explains a high proportion of variance in the target variable."
        elif best_score > 0.6:
            justification += " This model provides reasonable predictive power."
        else:
            justification += " Consider exploring additional features or more complex models."

    return {
        'model_name': best_model,
        'score': best_score,
        'justification': justification
    }
