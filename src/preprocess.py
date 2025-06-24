import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class HousePricePreprocessor:
    """
    A comprehensive preprocessor for the house price dataset.
    Handles data cleaning, transformation, and feature selection.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.selected_features = None
        
    def load_and_explore_data(self, filepath):
        """Load the dataset and perform initial exploration"""
        print("="*50)
        print("LOADING AND EXPLORING DATA")
        print("="*50)
        
        # Load data
        self.df = pd.read_csv(filepath)
        print(f"Dataset shape: {self.df.shape}")
        
        # Basic info
        print("\nDataset Info:")
        print(self.df.info())
        
        # Missing values
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_percent
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Basic statistics
        print("\nNumerical Features Statistics:")
        print(self.df.describe())
        
        # Categorical features
        print("\nCategorical Features:")
        for col in self.df.select_dtypes(include=['object']).columns:
            print(f"\n{col}:")
            print(f"  Unique values: {self.df[col].nunique()}")
            print(f"  Values: {self.df[col].value_counts().head()}")
        
        return self.df
    
    def clean_data(self):
        """Clean the messy dataset"""
        print("\n" + "="*50)
        print("CLEANING DATA")
        print("="*50)
        
        # Make a copy to preserve original
        df_clean = self.df.copy()
        
        # 1. Clean neighborhood names (standardize formatting)
        print("1. Cleaning neighborhood names...")
        df_clean['neighborhood'] = df_clean['neighborhood'].str.replace(' Area', '')
        df_clean['neighborhood'] = df_clean['neighborhood'].str.title()
        print(f"   Unique neighborhoods after cleaning: {df_clean['neighborhood'].unique()}")
        
        # 2. Clean property types
        print("2. Cleaning property types...")
        # Standardize property type formatting
        property_type_mapping = {
            'single family': 'Single Family',
            'single-family': 'Single Family', 
            'singlefamily': 'Single Family',
            'single_family': 'Single Family',
            'condo': 'Condo',
            'townhouse': 'Townhouse',
            'duplex': 'Duplex'
        }
        
        df_clean['property_type'] = df_clean['property_type'].str.lower()
        df_clean['property_type'] = df_clean['property_type'].str.replace('_', ' ')
        df_clean['property_type'] = df_clean['property_type'].str.replace('-', ' ')
        df_clean['property_type'] = df_clean['property_type'].replace(property_type_mapping)
        df_clean['property_type'] = df_clean['property_type'].str.title()
        print(f"   Unique property types: {df_clean['property_type'].unique()}")
        
        # 3. Clean garage column (convert to boolean)
        print("3. Cleaning garage column...")
        garage_true_values = ['true', 'yes', '1', 'TRUE', 'YES', 'True']
        garage_false_values = ['false', 'no', '0', 'FALSE', 'NO', 'False']
        
        df_clean['garage'] = df_clean['garage'].astype(str)
        df_clean['garage_bool'] = df_clean['garage'].apply(lambda x: 
            True if x in garage_true_values 
            else False if x in garage_false_values 
            else np.nan)
        
        print(f"   Garage values before: {self.df['garage'].unique()}")
        print(f"   Garage boolean distribution: {df_clean['garage_bool'].value_counts(dropna=False)}")
        
        # 4. Handle missing values
        print("4. Handling missing values...")
        
        # Fill missing year_built with median
        median_year = df_clean['year_built'].median()
        df_clean['year_built'].fillna(median_year, inplace=True)
        print(f"   Filled missing year_built with median: {median_year}")
        
        # Fill missing school_rating with neighborhood median
        df_clean['school_rating'] = df_clean.groupby('neighborhood')['school_rating'].transform(
            lambda x: x.fillna(x.median())
        )
        # If still missing, fill with overall median
        df_clean['school_rating'].fillna(df_clean['school_rating'].median(), inplace=True)
        print(f"   Filled missing school_rating with neighborhood median")
        
        # Fill missing crime_rate with neighborhood median
        df_clean['crime_rate'] = df_clean.groupby('neighborhood')['crime_rate'].transform(
            lambda x: x.fillna(x.median())
        )
        df_clean['crime_rate'].fillna(df_clean['crime_rate'].median(), inplace=True)
        print(f"   Filled missing crime_rate with neighborhood median")
        
        # Fill missing garage with mode (most common value)
        garage_mode = df_clean['garage_bool'].mode()[0]
        df_clean['garage_bool'].fillna(garage_mode, inplace=True)
        print(f"   Filled missing garage with mode: {garage_mode}")
        
        # 5. Handle outliers and impossible values
        print("5. Handling outliers and impossible values...")
        
        # Fix impossible bedroom/bathroom values
        df_clean.loc[df_clean['bedrooms'] == 0, 'bedrooms'] = 1
        df_clean.loc[df_clean['bathrooms'] == 0, 'bathrooms'] = 1
        print(f"   Fixed impossible bedroom/bathroom values")
        
        # Handle square footage outliers (cap at 99th percentile)
        sqft_99th = df_clean['square_feet'].quantile(0.99)
        outliers_sqft = (df_clean['square_feet'] > sqft_99th).sum()
        df_clean.loc[df_clean['square_feet'] > sqft_99th, 'square_feet'] = sqft_99th
        print(f"   Capped {outliers_sqft} square footage outliers at {sqft_99th}")
        
        # Handle price outliers
        price_1st = df_clean['price'].quantile(0.01)
        price_99th = df_clean['price'].quantile(0.99)
        price_outliers = ((df_clean['price'] < price_1st) | 
                         (df_clean['price'] > price_99th)).sum()
        df_clean = df_clean[
            (df_clean['price'] >= price_1st) & 
            (df_clean['price'] <= price_99th)
        ]
        print(f"   Removed {price_outliers} price outliers")
        
        # 6. Remove duplicates
        print("6. Removing duplicates...")
        duplicates = df_clean.duplicated().sum()
        df_clean = df_clean.drop_duplicates()
        print(f"   Removed {duplicates} duplicate rows")
        
        self.df_clean = df_clean
        print(f"\nCleaning complete! Dataset shape: {df_clean.shape}")
        
        return df_clean
    
    def feature_engineering(self):
        """Create new features and transform existing ones"""
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        df_features = self.df_clean.copy()
        
        # 1. Create age feature
        current_year = 2025
        df_features['house_age'] = current_year - df_features['year_built']
        print("1. Created house_age feature")
        
        # 2. Create price per square foot
        df_features['price_per_sqft'] = df_features['price'] / df_features['square_feet']
        print("2. Created price_per_sqft feature")
        
        # 3. Create total rooms
        df_features['total_rooms'] = df_features['bedrooms'] + df_features['bathrooms']
        print("3. Created total_rooms feature")
        
        # 4. Create bedroom to bathroom ratio
        df_features['bed_bath_ratio'] = df_features['bedrooms'] / df_features['bathrooms']
        print("4. Created bed_bath_ratio feature")
        
        # 5. Create categorical bins
        # Age categories
        df_features['age_category'] = pd.cut(df_features['house_age'], 
                                           bins=[0, 10, 25, 50, 100], 
                                           labels=['New', 'Modern', 'Mature', 'Old'])
        
        # Size categories
        df_features['size_category'] = pd.cut(df_features['square_feet'],
                                            bins=[0, 1200, 2000, 3000, float('inf')],
                                            labels=['Small', 'Medium', 'Large', 'XLarge'])
        print("5. Created categorical features for age and size")
        
        # 6. Encode categorical variables
        print("6. Encoding categorical variables...")
        categorical_cols = ['neighborhood', 'property_type', 'age_category', 'size_category']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_features[f'{col}_encoded'] = le.fit_transform(df_features[col])
            self.label_encoders[col] = le
            print(f"   Encoded {col}")
        
        # 7. Convert garage to numeric
        df_features['garage_numeric'] = df_features['garage_bool'].astype(int)
        
        self.df_features = df_features
        print(f"\nFeature engineering complete! Dataset shape: {df_features.shape}")
        
        return df_features
    
    def select_features(self, target_col='price', k=10):
        """Select the best features for prediction"""
        print("\n" + "="*50)
        print("FEATURE SELECTION")
        print("="*50)
        
        # Define potential features (excluding target and original categorical columns)
        # Also excluding size_category_encoded as it has 0% importance in final model
        exclude_cols = [target_col, 'garage', 'garage_bool', 'neighborhood', 
                       'property_type', 'age_category', 'size_category', 'year_built',
                       'size_category_encoded']
        
        feature_cols = [col for col in self.df_features.columns if col not in exclude_cols]
        
        X = self.df_features[feature_cols]
        y = self.df_features[target_col]
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"Available features: {feature_cols}")
        print(f"Selecting top {k} features...")
        
        # Use SelectKBest with f_regression
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = [feature_cols[i] for i, selected in enumerate(selected_mask) if selected]
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'feature': feature_cols,
            'score': selector.scores_,
            'selected': selected_mask
        }).sort_values('score', ascending=False)
        
        print("\nFeature Selection Results:")
        print(feature_scores)
        
        print(f"\nSelected features: {selected_features}")
        
        self.selected_features = selected_features
        self.feature_selector = selector
        
        return selected_features, X_selected, y[mask]
    
    def normalize_features(self, X, fit=True):
        """Normalize features using StandardScaler"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            print("Features normalized (fitted scaler)")
        else:
            X_scaled = self.scaler.transform(X)
            print("Features normalized (using fitted scaler)")
        
        return X_scaled
    
    def save_processed_data(self, filepath='../data/house_prices_processed.csv'):
        """Save the processed dataset"""
        # Create final dataset with selected features + target
        final_features = self.selected_features + ['price']
        final_df = self.df_features[final_features].copy()
        
        # Remove any rows with NaN values
        final_df = final_df.dropna()
        
        final_df.to_csv(filepath, index=False)
        print(f"\nProcessed dataset saved to: {filepath}")
        print(f"Final dataset shape: {final_df.shape}")
        
        return final_df
    
    def create_visualizations(self):
        """Create visualizations to understand the data better"""
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('House Price Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price distribution
        axes[0, 0].hist(self.df_features['price'], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Price Distribution')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Price vs Square Feet
        axes[0, 1].scatter(self.df_features['square_feet'], self.df_features['price'], 
                          alpha=0.6, color='coral')
        axes[0, 1].set_title('Price vs Square Feet')
        axes[0, 1].set_xlabel('Square Feet')
        axes[0, 1].set_ylabel('Price ($)')
        
        # 3. Price by Neighborhood
        neighborhood_price = self.df_features.groupby('neighborhood')['price'].median().sort_values()
        axes[0, 2].bar(range(len(neighborhood_price)), neighborhood_price.values, color='lightgreen')
        axes[0, 2].set_title('Median Price by Neighborhood')
        axes[0, 2].set_xlabel('Neighborhood')
        axes[0, 2].set_ylabel('Median Price ($)')
        axes[0, 2].set_xticks(range(len(neighborhood_price)))
        axes[0, 2].set_xticklabels(neighborhood_price.index, rotation=45, ha='right')
        
        # 4. Bedrooms vs Price
        bedroom_price = self.df_features.groupby('bedrooms')['price'].mean()
        axes[1, 0].bar(bedroom_price.index, bedroom_price.values, color='gold')
        axes[1, 0].set_title('Average Price by Bedrooms')
        axes[1, 0].set_xlabel('Number of Bedrooms')
        axes[1, 0].set_ylabel('Average Price ($)')
        
        # 5. House Age vs Price
        axes[1, 1].scatter(self.df_features['house_age'], self.df_features['price'], 
                          alpha=0.6, color='purple')
        axes[1, 1].set_title('Price vs House Age')
        axes[1, 1].set_xlabel('House Age (years)')
        axes[1, 1].set_ylabel('Price ($)')
        
        # 6. Feature Correlation Heatmap (top features)
        if self.selected_features:
            corr_features = self.selected_features[:6] + ['price']  # Top 6 features + target
            corr_matrix = self.df_features[corr_features].corr()
            im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            axes[1, 2].set_title('Feature Correlation Heatmap')
            axes[1, 2].set_xticks(range(len(corr_features)))
            axes[1, 2].set_yticks(range(len(corr_features)))
            axes[1, 2].set_xticklabels(corr_features, rotation=45, ha='right')
            axes[1, 2].set_yticklabels(corr_features)
            
            # Add correlation values to heatmap
            for i in range(len(corr_features)):
                for j in range(len(corr_features)):
                    text = axes[1, 2].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                         ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig('../visualizations/data_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Visualizations saved as '../visualizations/data_analysis_plots.png'")

def main():
    """Main preprocessing pipeline"""
    print("🏠 HOUSE PRICE PREDICTION - DATA PREPROCESSING")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = HousePricePreprocessor()
    
    # Step 1: Load and explore data
    df_raw = preprocessor.load_and_explore_data('../data/house_prices_raw.csv')
    
    # Step 2: Clean data
    df_clean = preprocessor.clean_data()
    
    # Step 3: Feature engineering
    df_features = preprocessor.feature_engineering()
    
    # Step 4: Feature selection
    selected_features, X_selected, y = preprocessor.select_features(k=7)
    
    # Step 5: Save processed data
    final_df = preprocessor.save_processed_data()
    
    # Step 6: Create visualizations
    preprocessor.create_visualizations()
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*60)
    print("Files created:")
    print("- ../data/house_prices_processed.csv (clean dataset)")
    print("- ../visualizations/data_analysis_plots.png (visualizations)")
    print(f"\nDataset ready for ML training with {len(selected_features)} features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"{i:2d}. {feature}")
    
    return preprocessor, final_df

if __name__ == "__main__":
    preprocessor, processed_data = main() 