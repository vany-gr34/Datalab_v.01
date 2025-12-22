def recommend_models(df, problem_type):
    """
    Returns a list of recommended models based on problem type.
    """
    if problem_type == 'classification':
        return ['LogisticRegression', 'KNN', 'RandomForest']
    elif problem_type == 'regression':
        return ['LinearRegression', 'KNNRegressor', 'RandomForestRegressor']
    return []
