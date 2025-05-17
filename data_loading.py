
import pandas as pd

def load_data(file_path):
    """Load the dataset and perform initial inspection."""
    data = pd.read_csv(file_path)
    
    print("First 5 rows of the dataset:")
    print(data.head())
    
    print("\nDataset Info:")
    print(data.info())
    
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    print("\nBasic Statistics:")
    print(data.describe())
    
    return data

if __name__ == "__main__":
    # Example usage
    data = load_data("../data/bungoma.csv")
