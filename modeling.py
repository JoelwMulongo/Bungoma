
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def train_model(X_train_scaled, y_train):
    """Train a Random Forest Regressor with hyperparameter tuning."""
    # Initialize model
    rf_model = RandomForestRegressor(random_state=42)
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best R² Score (CV): {grid_search.best_score_:.2f}")
    
    return best_model

if __name__ == "__main__":
    # Example usage (requires preprocessed data)
    from preprocessing import preprocess_data
    import pandas as pd
    
    data = pd.read_csv("../data/bungoma.csv")
    X_train_scaled, X_test_scaled, y_train, y_test, _, _ = preprocess_data(data)
    model = train_model(X_train_scaled, y_train)
