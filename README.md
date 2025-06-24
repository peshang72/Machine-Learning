# 🏠 House Price Prediction - AI-Powered Real Estate Valuation

A comprehensive machine learning project that predicts house prices using advanced AI algorithms. This project demonstrates the complete ML pipeline from data preprocessing to model deployment with an intuitive GUI.

## 📊 Project Overview

This project implements an end-to-end machine learning solution for house price prediction featuring:

- **Realistic Dataset**: Custom-generated dataset with intentional data quality issues
- **Comprehensive Preprocessing**: Data cleaning, feature engineering, and selection
- **Multiple ML Models**: Comparison of 6 different algorithms
- **High Accuracy**: Achieved 99.01% R² score with Gradient Boosting
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
| **year_built**    | Integer     | Construction year (1950-2024)     | ~5% missing values                                |
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
- **Processed**: 978 rows × 8 features (+ target)
- **Data Quality**: 100% complete, standardized formatting
- **Feature Engineering**: Added derived features (house_age, price_per_sqft, total_rooms)
- **Ready for ML**: Encoded categoricals, normalized distributions

This dataset provides an excellent foundation for demonstrating professional ML preprocessing techniques and achieving high-accuracy predictions.

## 🎯 Key Features

### 🔧 Data Preprocessing (`preprocess.py`)

- **Data Cleaning**: Handles missing values, inconsistent formatting, duplicates
- **Outlier Detection**: Identifies and handles extreme values
- **Feature Engineering**: Creates derived features like house age, price per sqft
- **Feature Selection**: Selects top 8 most important features
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
├── create_dataset.py          # Dataset generation script
├── preprocess.py             # Data preprocessing pipeline
├── train_model.py            # ML model training and evaluation
├── gui_app.py               # Interactive GUI application
├── house_prices_raw.csv     # Original messy dataset
├── house_prices_processed.csv # Clean, ready-to-use dataset
├── house_price_model.pkl    # Trained ML model
├── data_analysis_plots.png  # Data exploration visualizations
├── model_performance.png    # Model comparison plots
├── feature_importance.png   # Feature importance analysis
└── README.md               # This documentation
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib tkinter
```

### Step 1: Generate Dataset

```bash
python3 create_dataset.py
```

Creates `house_prices_raw.csv` with 1015 rows of messy real estate data including:

- Missing values (5-8% per feature)
- Inconsistent formatting
- Outliers and impossible values
- Duplicate records

### Step 2: Preprocess Data

```bash
python3 preprocess.py
```

Outputs:

- `house_prices_processed.csv` - Clean dataset
- `data_analysis_plots.png` - Exploratory visualizations

### Step 3: Train Models

```bash
python3 train_model.py
```

Outputs:

- `house_price_model.pkl` - Best trained model
- `model_performance.png` - Performance comparison
- `feature_importance.png` - Feature analysis

### Step 4: Run GUI Application

```bash
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

### Most Important Features (Gradient Boosting):

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

- **Input Validation**: Ensures valid input ranges
- **Real-time Prediction**: Instant price calculation
- **Visual Feedback**: Color-coded confidence indicators
- **History Tracking**: Previous prediction storage
- **Example Data**: Pre-loaded sample inputs

## 🎨 Visualizations

The project generates several visualization files:

### `data_analysis_plots.png`

- Price distribution histogram
- Price vs square feet scatter plot
- Median price by neighborhood
- Price by bedroom count
- House age vs price relationship
- Feature correlation heatmap

### `model_performance.png`

- Model comparison bar charts
- Predictions vs actual scatter plot
- Residuals analysis
- Performance metrics visualization

### `feature_importance.png`

- Feature importance ranking
- Contribution percentages
- Model interpretability insights

## 🛠️ Customization Options

### Adding New Features:

1. Modify `create_dataset.py` to include new variables
2. Update `preprocess.py` feature engineering section
3. Retrain model with `train_model.py`
4. Update GUI input fields in `gui_app.py`

### Trying Different Models:

1. Add new models to `initialize_models()` in `train_model.py`
2. Include hyperparameter grids for tuning
3. Compare performance metrics

### GUI Enhancements:

1. Add new input widgets to `create_input_panel()`
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
