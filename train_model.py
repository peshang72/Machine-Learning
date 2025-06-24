import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class HousePriceModelTrainer:
    """
    A comprehensive ML model trainer for house price prediction.
    Trains multiple models, performs hyperparameter tuning, and evaluates performance.
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, filepath='house_prices_processed.csv'):
        """Load the preprocessed dataset"""
        print("="*50)
        print("LOADING PROCESSED DATA")
        print("="*50)
        
        self.df = pd.read_csv(filepath)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Features: {list(self.df.columns[:-1])}")
        print(f"Target: {self.df.columns[-1]}")
        
        # Separate features and target
        self.X = self.df.drop('price', axis=1)
        self.y = self.df['price']
        
        print(f"\nFeature matrix shape: {self.X.shape}")
        print(f"Target vector shape: {self.y.shape}")
        print(f"Target statistics:")
        print(f"  Mean: ${self.y.mean():,.2f}")
        print(f"  Median: ${self.y.median():,.2f}")
        print(f"  Std: ${self.y.std():,.2f}")
        print(f"  Min: ${self.y.min():,.2f}")
        print(f"  Max: ${self.y.max():,.2f}")
        
        return self.X, self.y
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """Scale features for algorithms that require normalization"""
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Features scaled for algorithms requiring normalization")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def initialize_models(self):
        """Initialize various ML models"""
        print("\n" + "="*50)
        print("INITIALIZING MODELS")
        print("="*50)
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf')
        }
        
        print("Initialized models:")
        for i, model_name in enumerate(self.models.keys(), 1):
            print(f"{i}. {model_name}")
        
        return self.models
    
    def train_models(self):
        """Train all models and evaluate their performance"""
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        self.results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # Use scaled features for SVR, original features for tree-based models
            if model_name == 'Support Vector Regression':
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_test_use = self.X_test
            
            # Train the model
            model.fit(X_train_use, self.y_train)
            
            # Make predictions
            train_pred = model.predict(X_train_use)
            test_pred = model.predict(X_test_use)
            
            # Calculate metrics
            train_mse = mean_squared_error(self.y_train, train_pred)
            test_mse = mean_squared_error(self.y_test, test_pred)
            train_mae = mean_absolute_error(self.y_train, train_pred)
            test_mae = mean_absolute_error(self.y_test, test_pred)
            train_r2 = r2_score(self.y_train, train_pred)
            test_r2 = r2_score(self.y_test, test_pred)
            
            # Cross-validation score
            if model_name == 'Support Vector Regression':
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                          cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=5, scoring='r2')
            
            # Store results
            self.results[model_name] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'train_rmse': np.sqrt(train_mse),
                'test_rmse': np.sqrt(test_mse)
            }
            
            print(f"  Train R² Score: {train_r2:.4f}")
            print(f"  Test R² Score: {test_r2:.4f}")
            print(f"  Test RMSE: ${np.sqrt(test_mse):,.2f}")
            print(f"  CV R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return self.results
    
    def compare_models(self):
        """Compare model performance and select the best one"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Train R²': metrics['train_r2'],
                'Test R²': metrics['test_r2'],
                'Test RMSE': metrics['test_rmse'],
                'Test MAE': metrics['test_mae'],
                'CV R² Mean': metrics['cv_r2_mean'],
                'CV R² Std': metrics['cv_r2_std']
            })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        self.comparison_df = self.comparison_df.sort_values('Test R²', ascending=False)
        
        print("Model Performance Comparison:")
        print(self.comparison_df.round(4))
        
        # Select best model based on test R² score
        self.best_model_name = self.comparison_df.iloc[0]['Model']
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"Test R² Score: {self.comparison_df.iloc[0]['Test R²']:.4f}")
        print(f"Test RMSE: ${self.comparison_df.iloc[0]['Test RMSE']:,.2f}")
        
        return self.best_model, self.best_model_name
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on the best model"""
        print(f"\n" + "="*50)
        print(f"HYPERPARAMETER TUNING - {self.best_model_name}")
        print("="*50)
        
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42)
            X_use = self.X_train
            
        elif self.best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
            model = GradientBoostingRegressor(random_state=42)
            X_use = self.X_train
            
        elif self.best_model_name == 'Ridge Regression':
            param_grid = {
                'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
            }
            model = Ridge()
            X_use = self.X_train_scaled
            
        elif self.best_model_name == 'Support Vector Regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'epsilon': [0.01, 0.1, 0.2]
            }
            model = SVR(kernel='rbf')
            X_use = self.X_train_scaled
            
        else:
            print("No hyperparameter tuning defined for this model.")
            return self.best_model
        
        print(f"Searching best parameters...")
        print(f"Parameter grid: {param_grid}")
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_use, self.y_train)
        
        # Get best model
        self.best_tuned_model = grid_search.best_estimator_
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate tuned model
        if self.best_model_name == 'Support Vector Regression':
            test_pred = self.best_tuned_model.predict(self.X_test_scaled)
        else:
            test_pred = self.best_tuned_model.predict(self.X_test)
        
        tuned_r2 = r2_score(self.y_test, test_pred)
        tuned_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        
        print(f"\nTuned model performance:")
        print(f"Test R² Score: {tuned_r2:.4f}")
        print(f"Test RMSE: ${tuned_rmse:,.2f}")
        
        # Compare with original model
        original_r2 = self.results[self.best_model_name]['test_r2']
        improvement = tuned_r2 - original_r2
        print(f"\nImprovement: {improvement:+.4f} R² points")
        
        return self.best_tuned_model
    
    def feature_importance(self):
        """Analyze feature importance for tree-based models"""
        if hasattr(self.best_tuned_model, 'feature_importances_'):
            print(f"\n" + "="*50)
            print(f"FEATURE IMPORTANCE - {self.best_model_name}")
            print("="*50)
            
            importances = self.best_tuned_model.feature_importances_
            feature_names = self.X.columns
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Feature Importance Ranking:")
            for i, (_, row) in enumerate(importance_df.iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<25} {row['importance']:.4f}")
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance - {self.best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return importance_df
        else:
            print(f"\nFeature importance not available for {self.best_model_name}")
            return None
    
    def save_model(self, filename='house_price_model.pkl'):
        """Save the trained model and scaler"""
        print(f"\n" + "="*50)
        print("SAVING MODEL")
        print("="*50)
        
        # Save both the model and scaler (for models that need scaling)
        model_data = {
            'model': self.best_tuned_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'feature_names': list(self.X.columns),
            'performance_metrics': self.results[self.best_model_name],
            'uses_scaling': self.best_model_name == 'Support Vector Regression'
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved as: {filename}")
        print(f"Model type: {self.best_model_name}")
        print(f"Features: {list(self.X.columns)}")
        print(f"Performance (Test R²): {self.results[self.best_model_name]['test_r2']:.4f}")
        
        return filename
    
    def create_performance_plots(self):
        """Create visualization of model performance"""
        print(f"\n" + "="*50)
        print("CREATING PERFORMANCE PLOTS")
        print("="*50)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Model Comparison - R² Scores
        models = self.comparison_df['Model']
        test_r2 = self.comparison_df['Test R²']
        
        axes[0, 0].bar(range(len(models)), test_r2, color='skyblue')
        axes[0, 0].set_title('Model Comparison - Test R² Score')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(test_r2):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. RMSE Comparison
        test_rmse = self.comparison_df['Test RMSE']
        axes[0, 1].bar(range(len(models)), test_rmse, color='coral')
        axes[0, 1].set_title('Model Comparison - Test RMSE')
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('RMSE ($)')
        axes[0, 1].set_xticks(range(len(models)))
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        
        # 3. Predictions vs Actual (Best Model)
        if self.best_model_name == 'Support Vector Regression':
            best_pred = self.best_tuned_model.predict(self.X_test_scaled)
        else:
            best_pred = self.best_tuned_model.predict(self.X_test)
        
        axes[1, 0].scatter(self.y_test, best_pred, alpha=0.6, color='green')
        axes[1, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1, 0].set_title(f'Predictions vs Actual - {self.best_model_name}')
        axes[1, 0].set_xlabel('Actual Price ($)')
        axes[1, 0].set_ylabel('Predicted Price ($)')
        
        # 4. Residuals Plot
        residuals = self.y_test - best_pred
        axes[1, 1].scatter(best_pred, residuals, alpha=0.6, color='purple')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title(f'Residuals Plot - {self.best_model_name}')
        axes[1, 1].set_xlabel('Predicted Price ($)')
        axes[1, 1].set_ylabel('Residuals ($)')
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Performance plots saved as 'model_performance.png'")

def main():
    """Main training pipeline"""
    print("🏠 HOUSE PRICE PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    # Initialize trainer
    trainer = HousePriceModelTrainer()
    
    # Step 1: Load data
    X, y = trainer.load_data()
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = trainer.split_data()
    
    # Step 3: Scale features
    X_train_scaled, X_test_scaled = trainer.scale_features()
    
    # Step 4: Initialize models
    models = trainer.initialize_models()
    
    # Step 5: Train models
    results = trainer.train_models()
    
    # Step 6: Compare models
    best_model, best_model_name = trainer.compare_models()
    
    # Step 7: Hyperparameter tuning
    best_tuned_model = trainer.hyperparameter_tuning()
    
    # Step 8: Feature importance analysis
    importance_df = trainer.feature_importance()
    
    # Step 9: Save model
    model_filename = trainer.save_model()
    
    # Step 10: Create performance plots
    trainer.create_performance_plots()
    
    print("\n" + "="*60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*60)
    print("Files created:")
    print(f"- {model_filename} (trained model)")
    print("- model_performance.png (performance plots)")
    if importance_df is not None:
        print("- feature_importance.png (feature importance plot)")
    
    final_r2 = trainer.results[best_model_name]['test_r2']
    final_rmse = trainer.results[best_model_name]['test_rmse']
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"📊 Final Performance:")
    print(f"   R² Score: {final_r2:.4f}")
    print(f"   RMSE: ${final_rmse:,.2f}")
    
    return trainer

if __name__ == "__main__":
    trainer = main() 