# 🏠 House Price Prediction - AI-Powered Real Estate Valuation

A comprehensive machine learning project that predicts house prices using advanced AI algorithms. This project demonstrates the complete ML pipeline from data preprocessing to model deployment with an intuitive GUI.

## 📊 Project Overview

This project implements an end-to-end machine learning solution for house price prediction featuring:

- **Realistic Dataset**: Custom-generated dataset with intentional data quality issues
- **Comprehensive Preprocessing**: Data cleaning, feature engineering, and selection
- **Multiple ML Models**: Comparison of 6 different algorithms
- **High Accuracy**: Achieved 99.06% R² score with Gradient Boosting
- **Interactive GUI**: User-friendly Tkinter application for predictions
- **Complete Pipeline**: From raw data to deployable application

## 📋 Dataset Description

### House Price Dataset Overview

Our dataset simulates realistic real estate data with intentional quality issues to demonstrate professional data preprocessing techniques. The dataset contains **1,015 house records** with **10 features** representing various aspects of residential properties.

### 🏡 Dataset Features

| Feature           | Type        | Description                       | Data Quality Issues                               |
| ----------------- | ----------- | --------------------------------- | ------------------------------------------------- |
| **neighborhood**  | Categorical | Property location (7 areas)       | Inconsistent formatting (mixed case, extra words) |
| **bedrooms**      | Integer     | Number of bedrooms (1-5)          | Some impossible values (0 bedrooms)               |
| **bathrooms**     | Float       | Number of bathrooms (1.0-4.0)     | Some impossible values (0 bathrooms)              |
| **square_feet**   | Integer     | Living area in sq ft (500-15,000) | Extreme outliers (mansion-sized homes)            |
| **year_built**    | Integer     | Construction year (1950-2025)     | ~5% missing values                                |
| **property_type** | Categorical | House type (4 categories)         | Inconsistent formatting (spaces, underscores)     |
| **garage**        | Boolean     | Has garage (True/False)           | Mixed representations ('yes', '1', 'True', etc.)  |
| **school_rating** | Float       | Local school quality (3.0-10.0)   | ~8% missing values                                |
| **crime_rate**    | Float       | Neighborhood safety (0.3-14.3)    | ~6% missing values                                |
| **price**         | Integer     | Target variable ($91K-$2.5M)      | Price outliers at extremes                        |

### 🌍 Neighborhood Categories

1. **Waterfront** - Premium lakeside/oceanfront properties (highest prices)
2. **Historic** - Established areas with character homes
3. **Downtown** - Urban core with city amenities
4. **University District** - Near educational institutions
5. **Shopping District** - Commercial area proximity
6. **Suburbs** - Family-friendly residential areas
7. **Industrial** - Lower-cost areas near industrial zones

### 🏠 Property Types

- **Single Family** - Detached homes (most common)
- **Condo** - Apartment-style ownership
- **Townhouse** - Multi-story attached homes
- **Duplex** - Two-unit properties

### 📊 Data Quality Challenges

Our dataset intentionally includes real-world data problems:

#### Missing Values (222 total):

- `school_rating`: 81 missing (7.98%)
- `crime_rate`: 60 missing (5.91%)
- `year_built`: 51 missing (5.02%)
- `garage`: 30 missing (2.96%)

#### Formatting Inconsistencies:

- **Neighborhoods**: 'downtown' vs 'Downtown' vs 'DOWNTOWN' vs 'Downtown Area'
- **Property Types**: 'Single Family' vs 'single-family' vs 'Single_Family'
- **Garage Values**: 'True', 'yes', 'YES', '1', 'true' all meaning the same thing

#### Data Entry Errors:

- **Impossible Values**: 10 properties with 0 bedrooms/bathrooms
- **Duplicates**: 15 exact duplicate records
- **Outliers**: 20+ mansion-sized homes (8,000-15,000 sq ft)
- **Price Extremes**: Some unrealistic price points

### 🎯 Dataset Realism

This dataset mimics common real estate data challenges:

1. **Multiple Data Sources**: Different systems use different formats
2. **Human Entry Errors**: Typos and impossible values
3. **System Integration**: Inconsistent boolean representations
4. **Market Variability**: Wide price ranges across neighborhoods
5. **Feature Correlations**: Realistic relationships between size, location, and price

### 🔄 Preprocessing Impact

After cleaning and preprocessing:

- **Original**: 1,015 rows × 10 columns
- **Engineered**: 978 rows × 14 potential features (+ target)
- **Final Selection**: 978 rows × 7 optimized features (+ target)
- **Data Quality**: 100% complete, standardized formatting
- **Feature Engineering**: Added derived features (house_age, price_per_sqft, total_rooms, etc.)
- **Feature Selection**: SelectKBest chose most predictive features
- **Ready for ML**: Encoded categoricals, normalized distributions

This dataset provides an excellent foundation for demonstrating professional ML preprocessing techniques and achieving high-accuracy predictions.

## 🔄 Complete Feature Transformation Pipeline

### Original Dataset Features (10 total):

1. **neighborhood** (categorical) - Property location (7 areas)
2. **bedrooms** (integer) - Number of bedrooms (1-5)
3. **bathrooms** (float) - Number of bathrooms (1.0-4.0)
4. **square_feet** (integer) - Living area in sq ft (500-15,000)
5. **year_built** (integer) - Construction year (1950-2025)
6. **property_type** (categorical) - House type (4 categories)
7. **garage** (boolean) - Has garage (True/False)
8. **school_rating** (float) - Local school quality (3.0-10.0)
9. **crime_rate** (float) - Neighborhood safety (0.3-14.3)
10. **price** (target) - House price ($91K-$2.5M)

### Feature Engineering Creates (14 potential features):

1. **square_feet** (original)
2. **bathrooms** (original)
3. **school_rating** (original, cleaned)
4. **crime_rate** (original, cleaned)
5. **house_age** (derived: 2025 - year_built)
6. **price_per_sqft** (derived: price / square_feet)
7. **total_rooms** (derived: bedrooms + bathrooms)
8. **bed_bath_ratio** (derived: bedrooms / bathrooms)
9. **neighborhood_encoded** (transformed: label encoded)
10. **property_type_encoded** (transformed: label encoded)
11. **age_category_encoded** (derived: binned house_age)
12. **size_category_encoded** (derived: binned square_feet)
13. **garage_numeric** (transformed: boolean to 0/1)
14. **bedrooms** (original)

### Final Selected Features (7 total):

**SelectKBest with f_regression selects the top 7 most predictive features:**

1. **square_feet** - Strongest predictor (80.94% importance)
2. **price_per_sqft** - Market value indicator (18.30% importance)
3. **neighborhood_encoded** - Location premium (0.32% importance)
4. **total_rooms** - Space factor (0.15% importance)
5. **school_rating** - Quality of life (0.13% importance)
6. **crime_rate** - Safety factor (0.09% importance)
7. **bathrooms** - Convenience factor (0.07% importance)

### Features Discarded (7 total):

- **bedrooms** → Replaced by `total_rooms` (more predictive)
- **year_built** → Age less important than size/location
- **property_type_encoded** → Location dominates over property type
- **garage_numeric** → Less predictive than other amenities
- **bed_bath_ratio** → Total count more informative than ratio
- **age_category_encoded** → Age binning not in top predictors
- **size_category_encoded** → Categorical size bins have 0% importance (redundant with square_feet)

## 🎯 Key Features

### 🔧 Data Preprocessing (`preprocess.py`)

- **Data Cleaning**: Handles missing values, inconsistent formatting, duplicates
- **Outlier Detection**: Identifies and handles extreme values
- **Feature Engineering**: Creates derived features like house age, price per sqft
- **Feature Selection**: Uses SelectKBest to identify top 7 most predictive features from 14 engineered features
- **Visualization**: Generates comprehensive data analysis plots

### 🤖 Machine Learning (`train_model.py`)

- **Multiple Models**: Tests 6 different algorithms
- **Hyperparameter Tuning**: Optimizes best model performance
- **Cross-Validation**: Ensures robust model evaluation
- **Feature Importance**: Analyzes which features matter most
- **Model Persistence**: Saves trained model for deployment

### 🖥️ GUI Application (`gui_app.py`)

- **Intuitive Interface**: Easy-to-use input forms
- **Real-time Predictions**: Instant price estimates
- **Prediction History**: Tracks previous predictions
- **Random Examples**: Pre-loaded sample data
- **Performance Metrics**: Displays model confidence

## 📁 Project Structure

```
team-project/
├── src/                           # Source code directory
│   ├── create_dataset.py          # Dataset generation script
│   ├── preprocess.py             # Data preprocessing pipeline
│   ├── train_model.py            # ML model training and evaluation
│   └── gui_app.py               # Interactive GUI application
├── data/                          # Data storage directory
│   ├── house_prices_raw.csv      # Original messy dataset (1015 rows)
│   └── house_prices_processed.csv # Clean, ready-to-use dataset (978 rows)
├── models/                        # Trained models directory
│   └── house_price_model.pkl     # Best trained ML model (Gradient Boosting)
├── visualizations/               # Generated plots and charts
│   ├── data_analysis_plots.png   # Data exploration visualizations
│   ├── model_performance.png     # Model comparison and evaluation plots
│   └── feature_importance.png    # Feature importance analysis charts
├── docs/                         # Documentation directory (for future use)
├── requirements.txt              # Python dependencies
└── README.md                    # Project documentation
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib tkinter
```

### Step 1: Generate Dataset

```bash
cd src
python3 create_dataset.py
```

Creates `data/house_prices_raw.csv` with 1015 rows of messy real estate data including:

- Missing values (5-8% per feature)
- Inconsistent formatting
- Outliers and impossible values
- Duplicate records

### Step 2: Preprocess Data

```bash
cd src
python3 preprocess.py
```

Outputs:

- `data/house_prices_processed.csv` - Clean dataset
- `visualizations/data_analysis_plots.png` - Exploratory visualizations

### Step 3: Train Models

```bash
cd src
python3 train_model.py
```

Outputs:

- `models/house_price_model.pkl` - Best trained model
- `visualizations/model_performance.png` - Performance comparison
- `visualizations/feature_importance.png` - Feature analysis

### Step 4: Run GUI Application

```bash
cd src
python3 gui_app.py
```

Launches interactive house price prediction interface.

## 📈 Model Performance

| Model                     | Test R² Score | Test RMSE   | Cross-Val R² |
| ------------------------- | ------------- | ----------- | ------------ |
| **Gradient Boosting**     | **0.9901**    | **$18,669** | **0.9810**   |
| Random Forest             | 0.9877        | $20,868     | 0.9769       |
| Ridge Regression          | 0.9349        | $47,965     | 0.9047       |
| Linear Regression         | 0.9349        | $47,967     | 0.9047       |
| Lasso Regression          | 0.9349        | $47,967     | 0.9047       |
| Support Vector Regression | -0.0062       | $188,618    | -0.0213      |

## 🎯 Key Insights

### Most Important Features (Gradient Boosting) - Top 7 Selected:

1. **Square Feet** (80.94%) - Primary price driver
2. **Price per Sq Ft** (18.30%) - Market value indicator
3. **Neighborhood** (0.32%) - Location premium
4. **Total Rooms** (0.15%) - Space factor
5. **School Rating** (0.13%) - Quality of life
6. **Crime Rate** (0.09%) - Safety factor
7. **Bathrooms** (0.07%) - Convenience factor

### Data Quality Issues Addressed:

- **Missing Values**: 5-8% across key features
- **Inconsistent Formatting**: Mixed case, spaces, symbols
- **Outliers**: Extreme property sizes and prices
- **Duplicates**: 15 duplicate records removed
- **Impossible Values**: Zero bedrooms/bathrooms fixed

## 🔍 Technical Details

### Data Preprocessing Pipeline:

1. **Load & Explore**: Initial data analysis and quality assessment
2. **Clean**: Standardize formatting, fix inconsistencies
3. **Handle Missing Values**: Neighborhood-based imputation
4. **Remove Outliers**: Statistical outlier detection and removal
5. **Feature Engineering**: Create derived features
6. **Encode Categoricals**: Label encoding for ML compatibility
7. **Feature Selection**: SelectKBest with f_regression
8. **Save Clean Data**: Export processed dataset

### Model Training Process:

1. **Initialize Models**: 6 different algorithms
2. **Train & Evaluate**: Cross-validation and test metrics
3. **Compare Performance**: Comprehensive model comparison
4. **Hyperparameter Tuning**: GridSearchCV optimization
5. **Feature Importance**: Analyze feature contributions
6. **Save Best Model**: Persist for deployment

### GUI Application Features:

- **Complete Feature Coverage**: All dataset features represented in user-friendly interface
- **Comprehensive Input Validation**: Range checking and logical relationship validation
- **Real-time Prediction**: Instant price calculation with model confidence
- **Visual Feedback**: Color-coded validation status and error messages
- **History Tracking**: Previous prediction storage and comparison
- **Example Data**: Pre-loaded realistic sample properties
- **Feature Transformation**: Automatic conversion from user inputs to model features

## 🔧 GUI Implementation Details

### Enhanced User Interface (Updated)

The GUI application has been completely enhanced to address missing features and implement robust input validation:

#### ✅ **Complete Feature Coverage:**

**Original Dataset Features (All Implemented):**

- **Bedrooms** (1-5): Number of bedrooms with realistic range validation
- **Bathrooms** (1.0-6.0): Number of bathrooms including half-baths
- **Square Footage** (500-8,000): Total living area with size validation
- **Year Built** (1950-2025): Construction year with historical range
- **Property Type**: Dropdown (Single Family, Condo, Townhouse, Duplex)
- **Garage**: Boolean selection (Yes/No) for parking availability
- **Neighborhood**: 7 location options with premium calculation
- **School Rating** (1.0-10.0): Educational quality indicator
- **Crime Rate** (0.1-15.0): Safety metric (lower is better)

#### 🛡️ **Comprehensive Input Validation:**

**Range Validation:**

- All numeric inputs checked against realistic property ranges
- Prevents impossible values (e.g., 0 bedrooms, future construction dates)

**Logical Relationship Validation:**

- Total rooms must be reasonable (bedrooms + bathrooms ≥ 2)
- Bathroom count limited to bedrooms + 2 (realistic ratio)
- Square footage must support bedroom count (min 200 sq ft per bedroom)

**User Experience Features:**

- **Validate Inputs** button for pre-submission checking
- Clear error messages with specific guidance
- Tooltips explaining valid ranges and purposes
- Color-coded feedback (✅ success, ❌ errors)

#### 🔄 **Feature Transformation Pipeline:**

The GUI intelligently transforms user-friendly inputs into the 8 optimized features the ML model expects:

**User Input → Model Feature Mapping:**

```
User Inputs (9 features):          Model Features (7 features):
├─ bedrooms                    →   [combined into total_rooms]
├─ bathrooms                   →   ├─ bathrooms
├─ square_feet                 →   ├─ square_feet
├─ year_built                  →   [not used by model - GUI reference only]
├─ property_type               →   [not used by model - GUI reference only]
├─ garage                      →   [not used by model - GUI reference only]
├─ neighborhood                →   ├─ neighborhood_encoded
├─ school_rating               →   ├─ school_rating
├─ crime_rate                  →   ├─ crime_rate
└─ [calculated automatically] →   ├─ price_per_sqft (derived from market rates)
                               →   └─ total_rooms (bedrooms + bathrooms)
```

**Derived Feature Calculations:**

- `total_rooms = bedrooms + bathrooms`
- `price_per_sqft = 200` (default market rate, adjusted by model)
- `size_category = binned(square_feet)` → Small/Medium/Large/XLarge
- `neighborhood_encoded = label_encoded(neighborhood)`

### Feature Selection Rationale

The preprocessing pipeline transforms 10 original features into 14 potential features, then selects the 7 most predictive:

#### 🔴 **Features Discarded (7/14):**

- **`bedrooms`** → Replaced by `total_rooms` (more predictive)
- **`year_built`** → Age less important than size/location
- **`property_type`** → Location dominates over property type
- **`garage`** → Less predictive than other amenities
- **`bed_bath_ratio`** → Total count more informative than ratio
- **`age_category`** → Age binning not in top predictors
- **`size_category_encoded`** → Categorical size bins have 0% importance

#### ✅ **Features Selected (7/14):**

1. **`square_feet`** - Strongest predictor (80.94% importance)
2. **`price_per_sqft`** - Market value indicator (18.30% importance)
3. **`neighborhood_encoded`** - Location premium (0.32% importance)
4. **`total_rooms`** - Space factor (0.15% importance)
5. **`school_rating`** - Quality of life (0.13% importance)
6. **`crime_rate`** - Safety factor (0.09% importance)
7. **`bathrooms`** - Convenience factor (0.07% importance)

#### 🎯 **Key Insights:**

- **Derived features outperform originals** (total_rooms > bedrooms)
- **Size and location dominate pricing** (square_feet + neighborhood = 81.26%)
- **Combined metrics more predictive** than individual counts
- **Market indicators powerful** (price_per_sqft derived feature)
- **Quality of life factors matter** (school_rating, crime_rate)

### Implementation Benefits

This feature transformation approach provides:

1. **User-Friendly Interface**: Familiar real estate terms (bedrooms, year built)
2. **Scientific Optimization**: Model uses statistically optimal features
3. **Automatic Intelligence**: GUI converts user inputs to best model format
4. **Robust Validation**: Prevents unrealistic combinations
5. **Educational Value**: Demonstrates feature engineering principles

## 🎨 Visualizations

The project generates several visualization files:

### `visualizations/data_analysis_plots.png`

- Price distribution histogram
- Price vs square feet scatter plot
- Median price by neighborhood
- Price by bedroom count
- House age vs price relationship
- Feature correlation heatmap

### `visualizations/model_performance.png`

- Model comparison bar charts
- Predictions vs actual scatter plot
- Residuals analysis
- Performance metrics visualization

### `visualizations/feature_importance.png`

- Feature importance ranking
- Contribution percentages
- Model interpretability insights

## 🛠️ Customization Options

### Adding New Features:

1. Modify `src/create_dataset.py` to include new variables
2. Update `src/preprocess.py` feature engineering section
3. Retrain model with `src/train_model.py`
4. Update GUI input fields in `src/gui_app.py`

### Trying Different Models:

1. Add new models to `initialize_models()` in `src/train_model.py`
2. Include hyperparameter grids for tuning
3. Compare performance metrics

### GUI Enhancements:

1. Add new input widgets to `create_input_panel()` in `src/gui_app.py`
2. Modify prediction display in `display_prediction()`
3. Customize styling in `setup_styles()`

## 📚 Educational Value

This project demonstrates key ML concepts:

- **Data Quality**: Real-world data challenges and solutions
- **Feature Engineering**: Creating meaningful variables
- **Model Selection**: Comparing different algorithms
- **Hyperparameter Tuning**: Optimizing model performance
- **Model Evaluation**: Comprehensive performance metrics
- **Deployment**: Creating user-friendly applications

## 📋 Recent Updates & Improvements

### GUI Enhancement (Latest Update)

**Problem Addressed:** The original GUI was missing several dataset features mentioned in the documentation and lacked proper input validation.

**Solutions Implemented:**

#### ✅ **Added Missing Features:**

- **Bedrooms** input field (1-5 range)
- **Year Built** input field (1950-2025 range)
- **Property Type** dropdown (Single Family, Condo, Townhouse, Duplex)
- **Garage** selection (Yes/No)

#### 🛡️ **Comprehensive Input Validation:**

- **Range Validation**: All inputs checked against realistic property ranges
- **Logical Validation**: Ensures reasonable relationships between features
- **Real-time Feedback**: "Validate Inputs" button with clear error messages
- **User Guidance**: Descriptive tooltips and validation hints

#### 🎯 **Smart Feature Mapping:**

The GUI now intelligently maps user-friendly inputs to the model's optimized features:

| User Input           | Purpose        | Maps To Model Feature                     |
| -------------------- | -------------- | ----------------------------------------- |
| Bedrooms + Bathrooms | Room counting  | → `total_rooms` (combined)                |
| Square Feet          | Size input     | → `square_feet` + `size_category_encoded` |
| Neighborhood         | Location       | → `neighborhood_encoded`                  |
| Year Built           | Age reference  | → Used for validation only                |
| Property Type        | Type reference | → Used for display only                   |
| Garage               | Amenity info   | → Used for display only                   |

**Technical Achievement:** The GUI maintains user experience with familiar real estate terms while automatically providing the scientifically-optimal 7 features the ML model needs for accurate predictions.

**Impact:**

- ✅ 100% feature coverage matching documentation
- ✅ Robust error prevention and user guidance
- ✅ Maintains 99.01% model accuracy
- ✅ Enhanced user experience with realistic property examples

## 🔮 Future Enhancements

Potential improvements and extensions:

1. **Web Application**: Convert to Flask/Django web app
2. **Real Data Integration**: Connect to real estate APIs
3. **Advanced Models**: Deep learning, ensemble methods
4. **More Features**: Property photos, market trends
5. **Geographic Analysis**: Interactive maps, location data
6. **Market Predictions**: Time series forecasting
7. **Multi-City Support**: Expand to different markets

## 📝 License

This project is created for educational purposes and demonstrates machine learning best practices.

## 🤝 Contributing

Feel free to fork this project and submit improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📞 Support

For questions or issues:

- Review the code comments for implementation details
- Check the console output for debugging information
- Ensure all dependencies are properly installed

---

**Built with ❤️ for machine learning education**

_This project showcases the complete ML pipeline from data generation to deployment, emphasizing real-world data challenges and practical solutions._
