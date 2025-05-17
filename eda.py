# src/eda.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def perform_eda(data):
    """Perform exploratory data analysis with visualizations."""
    plt.style.use('seaborn')
    
    # Distribution of Teenage Pregnancy Rate
    plt.figure(figsize=(10, 6))
    sns.histplot(data['TeenPregnancyRate'], bins=30, kde=True)
    plt.title('Distribution of Teenage Pregnancy Rates')
    plt.xlabel('Teenage Pregnancy Rate (%)')
    plt.ylabel('Frequency')
    plt.savefig('../plots/teen_pregnancy_distribution.png')
    plt.show()
    
    # Correlation Analysis
    numerical_cols = ['TeenPregnancyRate', 'SchoolDropouts', 'HealthCenters']
    corr_matrix = data[numerical_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig('../plots/correlation_heatmap.png')
    plt.show()
    
    # Contraceptive Access vs TeenPregnancyRate
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='ContraceptiveAccess', y='TeenPregnancyRate', data=data)
    plt.title('Teenage Pregnancy Rate by Contraceptive Access')
    plt.savefig('../plots/contraceptive_access_boxplot.png')
    plt.show()
    
    # Education Programs vs TeenPregnancyRate
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='EducationPrograms', y='TeenPregnancyRate', data=data)
    plt.title('Teenage Pregnancy Rate by Education Programs')
    plt.savefig('../plots/education_programs_boxplot.png')
    plt.show()
    
    # Average TeenPregnancyRate by SubCounty
    subcounty_rates = data.groupby('SubCounty')['TeenPregnancyRate'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=subcounty_rates.values, y=subcounty_rates.index)
    plt.title('Average Teenage Pregnancy Rate by SubCounty')
    plt.xlabel('Teenage Pregnancy Rate (%)')
    plt.ylabel('SubCounty')
    plt.savefig('../plots/subcounty_pregnancy_rates.png')
    plt.show()

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv("../data/bungoma.csv")
    perform_eda(data)
