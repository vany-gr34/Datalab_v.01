import numpy as np

def recommend_models(df, problem_type):
    """
    Returns a list of recommended models based on problem type and dataset characteristics.
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

    # Base recommendations by problem type
    if problem_type == 'classification':
        base_models = ['Logistic Regression', 'KNN', 'Random Forest', 'Decision Tree', 'SVM', 'Naive Bayes']
    elif problem_type == 'regression':
        base_models = ['Linear Regression', 'Ridge Regression', 'KNN Regressor', 'Random Forest Regressor', 'Decision Tree Regressor', 'XGBoost Regressor']
    else:
        return []

    # Intelligent filtering based on dataset characteristics
    recommendations = []

    # Small dataset: prefer simple, interpretable models
    if dataset_size == 'small':
        if problem_type == 'classification':
            recommendations = ['Logistic Regression', 'KNN', 'Decision Tree', 'Naive Bayes']
        else:
            recommendations = ['Linear Regression', 'Ridge Regression', 'KNN Regressor', 'Decision Tree Regressor']

    # Large dataset: can handle more complex models
    elif dataset_size == 'large':
        if problem_type == 'classification':
            recommendations = ['Random Forest', 'SVM', 'XGBoost', 'Logistic Regression']
        else:
            recommendations = ['Random Forest Regressor', 'XGBoost Regressor', 'Linear Regression', 'Ridge Regression']

    # Medium dataset: balanced approach
    else:
        recommendations = base_models[:4]  # Top 4 models

    # Adjust for feature complexity
    if feature_complexity == 'high':
        # Prefer models that handle high dimensionality well
        if problem_type == 'classification':
            if 'Random Forest' not in recommendations:
                recommendations.append('Random Forest')
            if 'SVM' not in recommendations:
                recommendations.append('SVM')
        else:
            if 'Random Forest Regressor' not in recommendations:
                recommendations.append('Random Forest Regressor')
            if 'XGBoost Regressor' not in recommendations:
                recommendations.append('XGBoost Regressor')

    # Adjust for categorical features
    if feature_type == 'categorical' and n_categorical > 0:
        # Note: categorical features may need preprocessing
        if problem_type == 'classification':
            if 'Decision Tree' not in recommendations:
                recommendations.append('Decision Tree')
            if 'Random Forest' not in recommendations:
                recommendations.append('Random Forest')

    # Ensure we have at least 3 recommendations
    if len(recommendations) < 3:
        for model in base_models:
            if model not in recommendations:
                recommendations.append(model)
            if len(recommendations) >= 3:
                break

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
