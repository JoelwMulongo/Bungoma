
from sklearn.metrics import mean_squared_error -import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
    """Evaluate the model and generate visualizations."""
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Actual vs Predicted Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual TeenPregnancyRate')
    plt.ylabel('Predicted TeenPregnancyRate')
    plt.title('Actual vs Predicted Teenage Pregnancy Rates')
    plt.savefig('../plots/actual_vs_predicted.png')
    plt.show()
    
    # Cross-Validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"Cross-Validation R² Scores: {cv_scores}")
    print(f"Average CV R² Score: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.savefig('../plots/feature_importance.png')
    plt.show()

if __name__ == "__main__":
    # Example usage
    from preprocessing import preprocess_data
    from modeling import train_model
    import pandas as pd
    
    data = pd.read_csv("../data/bungoma.csv")
    X_train_scaled, X_test_scaled, y_train, y_test, _, feature_names = preprocess_data(data)
    model = train_model(X_train_scaled, y_train)
    evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
