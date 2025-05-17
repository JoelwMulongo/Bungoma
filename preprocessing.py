
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def preprocess_data(data):
    """Preprocess the dataset for modeling."""
    # Encode categorical variables
    le = LabelEncoder()
    data['ContraceptiveAccess'] = le.fit_transform(data['ContraceptiveAccess'])
    data['EducationPrograms'] = le.fit_transform(data['EducationPrograms'])
    
    # Drop geographical columns
    data_processed = data.drop(['SubCounty', 'Ward', 'Village'], axis=1)
    
    # Separate features and target
    X = data_processed.drop('TeenPregnancyRate', axis=1)
    y = data_processed['TeenPregnancyRate']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv("../data/bungoma.csv")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_data(data)
    print("Preprocessing complete. Shapes:")
    print(f"X_train_scaled: {X_train_scaled.shape}, X_test_scaled: {X_test_scaled.shape}")
