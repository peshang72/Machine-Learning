import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import joblib
import numpy as np
import pandas as pd
from tkinter import font
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictorGUI:
    """
    A comprehensive GUI application for house price prediction.
    Loads the trained ML model and provides an intuitive interface for predictions.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("🏠 House Price Predictor - AI Powered Real Estate Valuation")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f0f0')
        
        # Load the trained model
        self.load_model()
        
        # Setup the GUI
        self.setup_styles()
        self.create_widgets()
        
        # Initialize prediction history
        self.prediction_history = []
        
    def load_model(self):
        """Load the trained machine learning model"""
        try:
            model_data = joblib.load('../models/house_price_model.pkl')
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_name = model_data['model_name']
            self.feature_names = model_data['feature_names']
            self.uses_scaling = model_data['uses_scaling']
            self.performance_metrics = model_data['performance_metrics']
            
            print(f"✅ Model loaded successfully: {self.model_name}")
            print(f"📊 Model Performance - R² Score: {self.performance_metrics['test_r2']:.4f}")
            
        except FileNotFoundError:
            messagebox.showerror("Error", "Model file '../models/house_price_model.pkl' not found!\nPlease run train_model.py first.")
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
            self.root.destroy()
    
    def setup_styles(self):
        """Setup custom styles for the GUI"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Custom colors
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'background': '#f0f0f0',
            'white': '#ffffff'
        }
        
        # Configure styles
        self.style.configure('Title.TLabel', 
                           font=('Arial', 20, 'bold'),
                           foreground=self.colors['primary'])
        
        self.style.configure('Heading.TLabel',
                           font=('Arial', 14, 'bold'),
                           foreground=self.colors['secondary'])
        
        self.style.configure('Info.TLabel',
                           font=('Arial', 10),
                           foreground='#666666')
        
        self.style.configure('Predict.TButton',
                           font=('Arial', 12, 'bold'),
                           foreground='white')
        
    def create_widgets(self):
        """Create and layout all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="🏠 AI House Price Predictor", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Model info
        model_info = f"Powered by {self.model_name} | Accuracy: {self.performance_metrics['test_r2']:.1%} | RMSE: ${self.performance_metrics['test_rmse']:,.0f}"
        info_label = ttk.Label(main_frame, text=model_info, style='Info.TLabel')
        info_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Input fields
        self.create_input_panel(main_frame)
        
        # Right panel - Results and visualization
        self.create_results_panel(main_frame)
        
    def create_input_panel(self, parent):
        """Create the input panel with all house feature inputs"""
        # Input frame
        input_frame = ttk.LabelFrame(parent, text="House Features", padding="15")
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Initialize input variables
        self.input_vars = {}
        
        # Input fields configuration
        inputs_config = [
            {
                'name': 'bedrooms',
                'label': 'Number of Bedrooms',
                'type': 'int',
                'min': 1,
                'max': 5,
                'step': 1,
                'default': 3,
                'tooltip': 'Number of bedrooms (1-5)'
            },
            {
                'name': 'bathrooms',
                'label': 'Number of Bathrooms',
                'type': 'float',
                'min': 1.0,
                'max': 6.0,
                'step': 0.5,
                'default': 2.5,
                'tooltip': 'Number of bathrooms (including half-baths)'
            },
            {
                'name': 'square_feet',
                'label': 'Square Footage',
                'type': 'int',
                'min': 500,
                'max': 8000,
                'step': 100,
                'default': 2000,
                'tooltip': 'Total living area in square feet (500-8,000)'
            },
            {
                'name': 'year_built',
                'label': 'Year Built',
                'type': 'int',
                'min': 1950,
                'max': 2024,
                'step': 1,
                'default': 2000,
                'tooltip': 'Year the house was constructed (1950-2024)'
            },
            {
                'name': 'school_rating',
                'label': 'School Rating (1-10)',
                'type': 'float',
                'min': 1.0,
                'max': 10.0,
                'step': 0.1,
                'default': 7.0,
                'tooltip': 'Local school district rating (1=poor, 10=excellent)'
            },
            {
                'name': 'crime_rate',
                'label': 'Crime Rate',
                'type': 'float',
                'min': 0.1,
                'max': 15.0,
                'step': 0.1,
                'default': 3.0,
                'tooltip': 'Local crime rate (0.1-15.0, lower is better)'
            }
        ]
        
        # Create input widgets
        for i, config in enumerate(inputs_config):
            # Label
            label = ttk.Label(input_frame, text=config['label'])
            label.grid(row=i, column=0, sticky=tk.W, pady=5, padx=(0, 10))
            
            # Input widget
            if config['type'] == 'int':
                var = tk.IntVar(value=config['default'])
                widget = tk.Spinbox(input_frame, 
                                  from_=config['min'], 
                                  to=config['max'],
                                  increment=config['step'],
                                  textvariable=var,
                                  width=15)
            else:  # float
                var = tk.DoubleVar(value=config['default'])
                widget = tk.Spinbox(input_frame,
                                  from_=config['min'],
                                  to=config['max'],
                                  increment=config['step'],
                                  textvariable=var,
                                  width=15,
                                  format="%.1f")
            
            widget.grid(row=i, column=1, sticky=tk.W, pady=5)
            self.input_vars[config['name']] = var
            
            # Tooltip (displayed as info label)
            tooltip_label = ttk.Label(input_frame, text=config['tooltip'], style='Info.TLabel')
            tooltip_label.grid(row=i, column=2, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Dropdown inputs for categorical variables
        categorical_row = len(inputs_config)
        
        # Neighborhood
        ttk.Label(input_frame, text="Neighborhood").grid(row=categorical_row, column=0, sticky=tk.W, pady=5, padx=(0, 10))
        self.neighborhood_var = tk.StringVar(value="Suburbs")
        neighborhood_combo = ttk.Combobox(input_frame, textvariable=self.neighborhood_var, width=12)
        neighborhood_combo['values'] = ['Downtown', 'Suburbs', 'Waterfront', 'Industrial', 
                                       'Historic', 'University District', 'Shopping District']
        neighborhood_combo.grid(row=categorical_row, column=1, sticky=tk.W, pady=5)
        neighborhood_combo.state(['readonly'])
        ttk.Label(input_frame, text="Property location (7 areas)", style='Info.TLabel').grid(row=categorical_row, column=2, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Property Type
        ttk.Label(input_frame, text="Property Type").grid(row=categorical_row+1, column=0, sticky=tk.W, pady=5, padx=(0, 10))
        self.property_type_var = tk.StringVar(value="Single Family")
        property_combo = ttk.Combobox(input_frame, textvariable=self.property_type_var, width=12)
        property_combo['values'] = ['Single Family', 'Condo', 'Townhouse', 'Duplex']
        property_combo.grid(row=categorical_row+1, column=1, sticky=tk.W, pady=5)
        property_combo.state(['readonly'])
        ttk.Label(input_frame, text="Type of residential property", style='Info.TLabel').grid(row=categorical_row+1, column=2, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Garage
        ttk.Label(input_frame, text="Has Garage").grid(row=categorical_row+2, column=0, sticky=tk.W, pady=5, padx=(0, 10))
        self.garage_var = tk.StringVar(value="Yes")
        garage_combo = ttk.Combobox(input_frame, textvariable=self.garage_var, width=12)
        garage_combo['values'] = ['Yes', 'No']
        garage_combo.grid(row=categorical_row+2, column=1, sticky=tk.W, pady=5)
        garage_combo.state(['readonly'])
        ttk.Label(input_frame, text="Property has garage/parking", style='Info.TLabel').grid(row=categorical_row+2, column=2, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=categorical_row+3, column=0, columnspan=3, pady=20)
        
        # Predict button
        self.predict_btn = ttk.Button(button_frame, 
                                    text="🔮 Predict Price", 
                                    command=self.predict_price,
                                    style='Predict.TButton')
        self.predict_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Validate inputs button
        validate_btn = ttk.Button(button_frame,
                                text="✅ Validate Inputs",
                                command=self.validate_inputs_display)
        validate_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Random example button
        random_btn = ttk.Button(button_frame,
                              text="🎲 Random Example",
                              command=self.load_random_example)
        random_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear button
        clear_btn = ttk.Button(button_frame,
                             text="🗑️ Clear",
                             command=self.clear_inputs)
        clear_btn.pack(side=tk.LEFT)
    
    def create_results_panel(self, parent):
        """Create the results panel with prediction display and history"""
        # Results frame
        results_frame = ttk.LabelFrame(parent, text="Prediction Results", padding="15")
        results_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Prediction display
        self.prediction_frame = ttk.Frame(results_frame)
        self.prediction_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial prediction display
        self.create_prediction_display()
        
        # History frame
        history_frame = ttk.LabelFrame(results_frame, text="Prediction History", padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # History treeview
        columns = ('Price', 'SqFt', 'Bathrooms', 'Neighborhood')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=8)
        
        # Define headings
        self.history_tree.heading('Price', text='Predicted Price')
        self.history_tree.heading('SqFt', text='Sq Ft')
        self.history_tree.heading('Bathrooms', text='Bathrooms')
        self.history_tree.heading('Neighborhood', text='Neighborhood')
        
        # Define column widths
        self.history_tree.column('Price', width=120)
        self.history_tree.column('SqFt', width=80)
        self.history_tree.column('Bathrooms', width=80)
        self.history_tree.column('Neighborhood', width=120)
        
        # Scrollbar for history
        history_scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scrollbar.set)
        
        # Pack history widgets
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_prediction_display(self):
        """Create the main prediction display area"""
        # Clear existing widgets
        for widget in self.prediction_frame.winfo_children():
            widget.destroy()
        
        # Welcome message
        welcome_label = ttk.Label(self.prediction_frame, 
                                text="Enter house features and click 'Predict Price' to get started!",
                                font=('Arial', 12),
                                foreground='#666666')
        welcome_label.pack(pady=50)
        
        # Model info
        model_info_text = f"""
Model: {self.model_name}
Accuracy: {self.performance_metrics['test_r2']:.1%}
RMSE: ${self.performance_metrics['test_rmse']:,.0f}
        """
        
        info_label = ttk.Label(self.prediction_frame, 
                             text=model_info_text.strip(),
                             font=('Arial', 10),
                             foreground='#888888')
        info_label.pack(pady=10)
    
    def validate_inputs(self):
        """Validate all input values and return error messages if any"""
        errors = []
        
        try:
            # Get values
            bedrooms = self.input_vars['bedrooms'].get()
            bathrooms = self.input_vars['bathrooms'].get()
            square_feet = self.input_vars['square_feet'].get()
            year_built = self.input_vars['year_built'].get()
            school_rating = self.input_vars['school_rating'].get()
            crime_rate = self.input_vars['crime_rate'].get()
            
            # Validate ranges
            if not (1 <= bedrooms <= 5):
                errors.append("Bedrooms must be between 1 and 5")
            
            if not (1.0 <= bathrooms <= 6.0):
                errors.append("Bathrooms must be between 1.0 and 6.0")
            
            if not (500 <= square_feet <= 8000):
                errors.append("Square footage must be between 500 and 8,000")
            
            if not (1950 <= year_built <= 2024):
                errors.append("Year built must be between 1950 and 2024")
            
            if not (1.0 <= school_rating <= 10.0):
                errors.append("School rating must be between 1.0 and 10.0")
            
            if not (0.1 <= crime_rate <= 15.0):
                errors.append("Crime rate must be between 0.1 and 15.0")
            
            # Validate logical relationships
            total_rooms = bedrooms + bathrooms
            if total_rooms < 2:
                errors.append("Total rooms (bedrooms + bathrooms) must be at least 2")
            
            if bathrooms > bedrooms + 2:
                errors.append("Number of bathrooms cannot exceed bedrooms + 2")
            
            # Validate square footage vs bedrooms
            min_sqft_per_bedroom = 200
            if square_feet < bedrooms * min_sqft_per_bedroom:
                errors.append(f"Square footage too small for {bedrooms} bedrooms (minimum {bedrooms * min_sqft_per_bedroom} sq ft)")
                
        except (ValueError, TypeError) as e:
            errors.append(f"Invalid input values: {str(e)}")
        
        return errors

    def validate_inputs_display(self):
        """Display validation results to user"""
        errors = self.validate_inputs()
        if not errors:
            messagebox.showinfo("Input Validation", "✅ All inputs are valid!\n\nYou can now proceed with price prediction.")
        else:
            error_message = "❌ Please fix the following issues:\n\n" + "\n".join(f"• {error}" for error in errors)
            messagebox.showwarning("Input Validation Issues", error_message)

    def get_encoded_values(self, neighborhood, property_type, garage):
        """Get encoded values for categorical features"""
        # Neighborhood encoding (based on alphabetical order from preprocessing)
        neighborhood_encoding = {
            'Downtown': 0,
            'Historic': 1,
            'Industrial': 2,
            'Shopping District': 3,
            'Suburbs': 4,
            'University District': 5,
            'Waterfront': 6
        }
        
        # Property type encoding
        property_type_encoding = {
            'Condo': 0,
            'Duplex': 1,
            'Single Family': 2,
            'Townhouse': 3
        }
        
        # Garage encoding
        garage_encoding = {'Yes': 1, 'No': 0}
        
        return (neighborhood_encoding.get(neighborhood, 4), 
                property_type_encoding.get(property_type, 2),
                garage_encoding.get(garage, 1))
    
    def predict_price(self):
        """Make a price prediction based on input values"""
        try:
            # Validate inputs first
            validation_errors = self.validate_inputs()
            if validation_errors:
                error_message = "Please fix the following issues:\n\n" + "\n".join(f"• {error}" for error in validation_errors)
                messagebox.showerror("Input Validation Error", error_message)
                return
            
            # Get input values
            bedrooms = self.input_vars['bedrooms'].get()
            bathrooms = self.input_vars['bathrooms'].get()
            square_feet = self.input_vars['square_feet'].get()
            year_built = self.input_vars['year_built'].get()
            school_rating = self.input_vars['school_rating'].get()
            crime_rate = self.input_vars['crime_rate'].get()
            neighborhood = self.neighborhood_var.get()
            property_type = self.property_type_var.get()
            garage = self.garage_var.get()
            
            # Calculate derived features
            current_year = 2024
            house_age = current_year - year_built
            total_rooms = bedrooms + bathrooms
            price_per_sqft = 200  # Reasonable default, will be adjusted by model
            
            # Get encoded categorical values
            neighborhood_encoded, property_type_encoded, garage_encoded = self.get_encoded_values(
                neighborhood, property_type, garage)
            
            # Create feature array in the same order as training data
            # Based on the processed CSV: bathrooms, square_feet, school_rating, crime_rate, 
            # price_per_sqft, total_rooms, neighborhood_encoded (7 features total)
            features = np.array([[
                bathrooms,
                square_feet,
                school_rating,
                crime_rate,
                price_per_sqft,
                total_rooms,
                neighborhood_encoded
            ]])
            
            # Make prediction
            predicted_price = self.model.predict(features)[0]
            
            # Display prediction
            self.display_prediction(predicted_price, {
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'square_feet': square_feet,
                'year_built': year_built,
                'house_age': house_age,
                'school_rating': school_rating,
                'crime_rate': crime_rate,
                'total_rooms': total_rooms,
                'price_per_sqft': price_per_sqft,
                'neighborhood': neighborhood,
                'property_type': property_type,
                'garage': garage
            })
            
            # Add to history
            self.add_to_history(predicted_price, square_feet, bathrooms, neighborhood)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error making prediction: {str(e)}")
    
    def display_prediction(self, price, features):
        """Display the prediction result with detailed breakdown"""
        # Clear existing widgets
        for widget in self.prediction_frame.winfo_children():
            widget.destroy()
        
        # Main prediction
        price_label = ttk.Label(self.prediction_frame,
                              text=f"${price:,.0f}",
                              font=('Arial', 24, 'bold'),
                              foreground=self.colors['primary'])
        price_label.pack(pady=20)
        
        # Subtitle
        subtitle_label = ttk.Label(self.prediction_frame,
                                 text="Estimated Market Value",
                                 font=('Arial', 12),
                                 foreground='#666666')
        subtitle_label.pack()
        
        # Confidence indicator
        confidence_text = f"Model Confidence: {self.performance_metrics['test_r2']:.1%}"
        confidence_label = ttk.Label(self.prediction_frame,
                                   text=confidence_text,
                                   font=('Arial', 10),
                                   foreground='#888888')
        confidence_label.pack(pady=(10, 20))
        
        # Feature summary
        summary_frame = ttk.Frame(self.prediction_frame)
        summary_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create feature summary
        summary_text = f"""
Property Details:
• {features['square_feet']:,} sq ft {features['property_type'].lower()}
• {features['bedrooms']} bedrooms, {features['bathrooms']} bathrooms ({features['total_rooms']} total rooms)
• Built in {features['year_built']} ({features['house_age']} years old)
• Located in {features['neighborhood']}
• Garage: {features['garage']}
• School rating: {features['school_rating']}/10
• Crime rate: {features['crime_rate']}/10
• Price per sq ft: ${features['price_per_sqft']:.0f}
        """
        
        summary_label = ttk.Label(summary_frame,
                                text=summary_text.strip(),
                                font=('Arial', 10),
                                foreground='#333333')
        summary_label.pack(anchor=tk.W)
        
        # Price range estimate
        margin = price * 0.1  # 10% margin of error
        range_text = f"Price range: ${price-margin:,.0f} - ${price+margin:,.0f}"
        range_label = ttk.Label(self.prediction_frame,
                              text=range_text,
                              font=('Arial', 10, 'italic'),
                              foreground='#666666')
        range_label.pack(pady=(20, 0))
    
    def add_to_history(self, price, sqft, bathrooms, neighborhood):
        """Add prediction to history"""
        # Add to internal history
        self.prediction_history.append({
            'price': price,
            'sqft': sqft,
            'bathrooms': bathrooms,
            'neighborhood': neighborhood
        })
        
        # Add to treeview
        self.history_tree.insert('', 0, values=(
            f"${price:,.0f}",
            f"{sqft:,}",
            f"{bathrooms}",
            neighborhood
        ))
        
        # Keep only last 20 predictions
        if len(self.prediction_history) > 20:
            self.prediction_history = self.prediction_history[-20:]
            # Clear and repopulate treeview
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            
            for pred in reversed(self.prediction_history):
                self.history_tree.insert('', 0, values=(
                    f"${pred['price']:,.0f}",
                    f"{pred['sqft']:,}",
                    f"{pred['bathrooms']}",
                    pred['neighborhood']
                ))
    
    def load_random_example(self):
        """Load a random example for demonstration"""
        import random
        
        examples = [
            {
                'bedrooms': 3, 'bathrooms': 2.5, 'square_feet': 1800, 'year_built': 2000, 'school_rating': 8.5,
                'crime_rate': 2.1, 'neighborhood': 'Suburbs', 'property_type': 'Single Family', 'garage': 'Yes'
            },
            {
                'bedrooms': 4, 'bathrooms': 3.0, 'square_feet': 2800, 'year_built': 2015, 'school_rating': 9.2,
                'crime_rate': 1.5, 'neighborhood': 'Waterfront', 'property_type': 'Single Family', 'garage': 'Yes'
            },
            {
                'bedrooms': 2, 'bathrooms': 1.5, 'square_feet': 1200, 'year_built': 1980, 'school_rating': 6.8,
                'crime_rate': 4.5, 'neighborhood': 'Downtown', 'property_type': 'Condo', 'garage': 'No'
            },
            {
                'bedrooms': 5, 'bathrooms': 4.0, 'square_feet': 4200, 'year_built': 1990, 'school_rating': 9.8,
                'crime_rate': 0.8, 'neighborhood': 'Historic', 'property_type': 'Single Family', 'garage': 'Yes'
            },
            {
                'bedrooms': 3, 'bathrooms': 2.0, 'square_feet': 1600, 'year_built': 2005, 'school_rating': 7.5,
                'crime_rate': 3.2, 'neighborhood': 'University District', 'property_type': 'Townhouse', 'garage': 'Yes'
            }
        ]
        
        example = random.choice(examples)
        
        # Set input values
        self.input_vars['bedrooms'].set(example['bedrooms'])
        self.input_vars['bathrooms'].set(example['bathrooms'])
        self.input_vars['square_feet'].set(example['square_feet'])
        self.input_vars['year_built'].set(example['year_built'])
        self.input_vars['school_rating'].set(example['school_rating'])
        self.input_vars['crime_rate'].set(example['crime_rate'])
        self.neighborhood_var.set(example['neighborhood'])
        self.property_type_var.set(example['property_type'])
        self.garage_var.set(example['garage'])
    
    def clear_inputs(self):
        """Clear all input fields to default values"""
        self.input_vars['bedrooms'].set(3)
        self.input_vars['bathrooms'].set(2.5)
        self.input_vars['square_feet'].set(2000)
        self.input_vars['year_built'].set(2000)
        self.input_vars['school_rating'].set(7.0)
        self.input_vars['crime_rate'].set(3.0)
        self.neighborhood_var.set("Suburbs")
        self.property_type_var.set("Single Family")
        self.garage_var.set("Yes")
        
        # Reset prediction display
        self.create_prediction_display()

def main():
    """Main function to run the GUI application"""
    print("🏠 Starting House Price Predictor GUI...")
    
    # Create and run the application
    root = tk.Tk()
    app = HousePricePredictorGUI(root)
    
    print("✅ GUI application started successfully!")
    print("📱 Application ready for use.")
    
    root.mainloop()

if __name__ == "__main__":
    main() 