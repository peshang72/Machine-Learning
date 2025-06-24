import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def create_messy_house_dataset():
    """Create a realistic house price dataset with various data quality issues"""
    
    n_samples = 1000
    
    # Base features
    np.random.seed(42)
    
    # Location data (with inconsistent formatting)
    neighborhoods = ['Downtown', 'Suburbs', 'Waterfront', 'Industrial', 'Historic', 
                    'University District', 'Shopping District']
    locations = np.random.choice(neighborhoods, n_samples)
    
    # Add some inconsistent formatting to locations
    messy_locations = []
    for loc in locations:
        if random.random() < 0.1:  # 10% chance of formatting issues
            if random.random() < 0.5:
                messy_locations.append(loc.lower())  # lowercase
            else:
                messy_locations.append(loc.upper())  # uppercase
        elif random.random() < 0.05:  # 5% chance of typos
            messy_locations.append(loc + ' Area')  # inconsistent naming
        else:
            messy_locations.append(loc)
    
    # House features
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.uniform(1, 4, n_samples).round(1)
    
    # Square footage with some outliers
    sqft = np.random.normal(2000, 800, n_samples)
    sqft = np.maximum(sqft, 500)  # minimum 500 sqft
    
    # Add some extreme outliers
    outlier_indices = np.random.choice(n_samples, 20, replace=False)
    sqft[outlier_indices] = np.random.uniform(8000, 15000, 20)  # mansions
    
    # Age of house
    current_year = 2024
    year_built = np.random.randint(1950, 2024, n_samples)
    age = current_year - year_built
    
    # Property type with inconsistent formatting
    property_types = ['Single Family', 'Condo', 'Townhouse', 'Duplex']
    prop_types = np.random.choice(property_types, n_samples)
    messy_prop_types = []
    for prop in prop_types:
        if random.random() < 0.08:  # 8% formatting issues
            if random.random() < 0.33:
                messy_prop_types.append(prop.replace(' ', '_'))  # underscore instead of space
            elif random.random() < 0.5:
                messy_prop_types.append(prop.replace(' ', ''))   # no space
            else:
                messy_prop_types.append(prop.lower().replace(' ', '-'))  # kebab-case
        else:
            messy_prop_types.append(prop)
    
    # Garage (with inconsistent boolean representation)
    has_garage = np.random.choice([True, False], n_samples, p=[0.7, 0.3])
    garage_strings = []
    for garage in has_garage:
        if random.random() < 0.1:  # 10% inconsistent representation
            if garage:
                garage_strings.append(random.choice(['yes', 'YES', '1', 'true', 'True']))
            else:
                garage_strings.append(random.choice(['no', 'NO', '0', 'false', 'False']))
        else:
            garage_strings.append(str(garage))
    
    # School rating (with some missing values)
    school_rating = np.random.uniform(3, 10, n_samples).round(1)
    
    # Crime rate (higher in some neighborhoods)
    base_crime_rate = np.random.uniform(0.5, 8.0, n_samples)
    neighborhood_crime_multiplier = {
        'Downtown': 1.5, 'Industrial': 1.8, 'Suburbs': 0.6, 
        'Waterfront': 0.7, 'Historic': 0.8, 'University District': 1.2,
        'Shopping District': 1.1
    }
    
    crime_rates = []
    for i, loc in enumerate(messy_locations):
        # Clean the location name to match our multipliers
        clean_loc = loc.replace(' Area', '').title()
        if clean_loc.lower() in [k.lower() for k in neighborhood_crime_multiplier.keys()]:
            # Find the matching key (case insensitive)
            for key in neighborhood_crime_multiplier.keys():
                if key.lower() == clean_loc.lower():
                    crime_rates.append(base_crime_rate[i] * neighborhood_crime_multiplier[key])
                    break
        else:
            crime_rates.append(base_crime_rate[i])
    
    crime_rates = np.array(crime_rates)
    
    # Calculate price based on features (with some noise)
    base_price = (
        sqft * 150 +  # $150 per sqft
        bedrooms * 10000 +  # $10k per bedroom
        bathrooms * 8000 +  # $8k per bathroom
        school_rating * 5000 +  # $5k per rating point
        (age < 10) * 20000 +  # $20k bonus for new houses
        has_garage * 15000 -  # $15k for garage
        crime_rates * 5000  # $5k penalty per crime rate point
    )
    
    # Add neighborhood premium/discount
    neighborhood_multipliers = {
        'Waterfront': 1.4, 'Historic': 1.2, 'Downtown': 1.1,
        'University District': 1.05, 'Shopping District': 1.0,
        'Suburbs': 0.95, 'Industrial': 0.8
    }
    
    prices = []
    for i, loc in enumerate(messy_locations):
        clean_loc = loc.replace(' Area', '').title()
        multiplier = 1.0
        for key in neighborhood_multipliers.keys():
            if key.lower() == clean_loc.lower():
                multiplier = neighborhood_multipliers[key]
                break
        prices.append(base_price[i] * multiplier)
    
    prices = np.array(prices)
    
    # Add random noise
    prices += np.random.normal(0, 20000, n_samples)
    prices = np.maximum(prices, 50000)  # minimum price
    
    # Create DataFrame
    df = pd.DataFrame({
        'neighborhood': messy_locations,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'square_feet': sqft.round(0).astype(int),
        'year_built': year_built,
        'property_type': messy_prop_types,
        'garage': garage_strings,
        'school_rating': school_rating,
        'crime_rate': crime_rates.round(2),
        'price': prices.round(0).astype(int)
    })
    
    # Introduce missing values strategically
    missing_indices = {
        'school_rating': np.random.choice(n_samples, 80, replace=False),  # 8% missing
        'garage': np.random.choice(n_samples, 30, replace=False),         # 3% missing
        'year_built': np.random.choice(n_samples, 50, replace=False),     # 5% missing
        'crime_rate': np.random.choice(n_samples, 60, replace=False),     # 6% missing
    }
    
    for col, indices in missing_indices.items():
        df.loc[indices, col] = np.nan
    
    # Add some duplicate rows (data entry errors)
    duplicate_indices = np.random.choice(n_samples, 15, replace=False)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Add some impossible values (data entry errors)
    error_indices = np.random.choice(len(df), 10, replace=False)
    df.loc[error_indices[:5], 'bedrooms'] = 0  # 0 bedrooms (impossible)
    df.loc[error_indices[5:], 'bathrooms'] = 0  # 0 bathrooms (very rare)
    
    return df

if __name__ == "__main__":
    # Create the dataset
    print("Creating messy house price dataset...")
    df = create_messy_house_dataset()
    
    # Save to CSV
    df.to_csv('house_prices_raw.csv', index=False)
    
    print(f"Dataset created with {len(df)} rows and {len(df.columns)} columns")
    print(f"Saved as 'house_prices_raw.csv'")
    
    # Display basic info about the dataset
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nUnique values in categorical columns:")
    for col in ['neighborhood', 'property_type', 'garage']:
        print(f"{col}: {df[col].unique()}") 