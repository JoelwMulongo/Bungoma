
from data_loading import load_data
from eda import perform_eda
from preprocessing import preprocess_data
from modeling import train_model
from evaluation import evaluate_model

def main():
    # File path
    data_path = "../data/bungoma.csv"
    
    # Load data
    print("Loading data...")
    data = load_data(data_path)
    
    # Perform EDA
    print("Performing EDA...")
    perform_eda(data)
    
    # Preprocess data
    print("Preprocessing data...")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_data(data)
    
    # Train model
    print("Training model...")
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, feature_names)

if __name__ == "__main__":
    main()
