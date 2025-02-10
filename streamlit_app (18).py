import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set the page configuration
st.set_page_config(
    page_title="Drug Inventory Tracker",
    page_icon=":pill:",
)

# Google Sheets setup
SHEET_URL = "https://docs.google.com/spreadsheets/d/1RECeYLwS84sKoVUs0Us8wKy6PTY4BfFIMcxyN6NF2UQ"


def get_gsheet_client():
    creds_dict = st.secrets.get("gspread_credentials")
    if creds_dict:
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scopes)
        return gspread.authorize(creds)
    return None
    
def connect_to_gsheet():
    try:
        client = get_gsheet_client()  # Always creates a new connection
        if client:
            return client.open_by_url(SHEET_URL).get_worksheet(0)
    except gspread.exceptions.APIError as e:
        st.error(f"Google Sheets API error: {e}")
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
    return None
    
def save_data(new_data):
    sheet = connect_to_gsheet()
    if sheet:
        try:
            # Convert the new data to match raw data format
            data_to_save = pd.DataFrame([new_data])  # Only save the new row
            data_list = data_to_save.values.tolist()
            sheet.append_rows(data_list, value_input_option="RAW")
            
            # Update session state
            st.session_state.raw_data = pd.concat([st.session_state.raw_data, data_to_save], ignore_index=True)
            # Reprocess the data
            st.session_state.processed_data = load_and_prepare_data(st.session_state.raw_data)
            
            st.success("New data successfully saved to Google Sheets!")
        except Exception as e:
            st.error(f"Error saving data: {e}")

def load_data():
    sheet = connect_to_gsheet()
    if sheet:
        try:
            # Get all values including headers
            all_values = sheet.get_all_values()
            if not all_values:
                st.error("Sheet is empty")
                return pd.DataFrame()
            
            # Extract headers and data
            headers = all_values[0]
            data_values = all_values[1:]
            
            # Create a list of unique headers for duplicate columns
            unique_headers = []
            header_count = {}
            
            for header in headers:
                if header in header_count:
                    header_count[header] += 1
                    unique_headers.append(f"{header}_{header_count[header]}")
                else:
                    header_count[header] = 1
                    unique_headers.append(header)
            
            # Create DataFrame with unique headers
            df = pd.DataFrame(data_values, columns=unique_headers)
            
            # Clean up empty rows and columns - Updated approach
            df = df.replace({'': None})  # Replace empty strings with None
            df = df.astype(object)  # Ensure consistent dtype
            df = df.infer_objects()  # Infer better dtypes where possible
            df = df.dropna(how='all')  # Drop rows that are all NA
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def rename_duplicate_columns(df):
    col_count = {}
    new_columns = []
    for col in df.columns:
        col_count[col] = col_count.get(col, 0) + 1
        new_columns.append(f"{col}_{col_count[col]}") if col_count[col] > 1 else new_columns.append(col)
    df.columns = new_columns
    return df

# Load and preprocess dataset
def load_and_prepare_data(original_df):
    """
    Load and preprocess dataset with enhanced validation for missing unit weights
    and system flags for data correction.
    """
    try:
        df = original_df.copy()
        validation_flags = {
            'missing_unit_weights': False,
            'corrected_values': False,
            'data_warnings': []
        }
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Convert column names to lowercase for case-insensitive matching
        df.columns = df.columns.str.lower()
        
        # Define essential columns with validation requirements
        essential_cols = {
            'number of tablets': {'type': 'numeric', 'required': True},
            'average weight of one loose cut tablet': {'type': 'numeric', 'required': True},
            'active pharmaceutical ingredient strength (mg)': {'type': 'numeric', 'required': True}
        }
        
        # Validate essential columns
        for col, requirements in essential_cols.items():
            if col not in df.columns:
                validation_flags['data_warnings'].append(f"Missing essential column: {col}")
                continue
                
            # Convert to numeric with validation
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Check for missing values
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    if col == 'average weight of one loose cut tablet':
                        validation_flags['missing_unit_weights'] = True
                    validation_flags['data_warnings'].append(
                        f"Found {missing_count} missing values in {col}"
                    )
                    
                    # Apply correction logic for missing unit weights
                    if col == 'average weight of one loose cut tablet':
                        # Calculate median weight by dosage form
                        median_weights = df.groupby('dosage form')[col].transform('median')
                        df[col].fillna(median_weights, inplace=True)
                        validation_flags['corrected_values'] = True
                        
            except Exception as e:
                validation_flags['data_warnings'].append(
                    f"Error converting {col}: {str(e)}"
                )

        # Handle remaining numeric columns
        numeric_columns = [
            'total weight of tablets without mixed packaging',
            'weight-to-strength ratio'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate total weight if missing
        if 'total weight of tablets without mixed packaging' not in df.columns:
            df['total weight of tablets without mixed packaging'] = (
                df['number of tablets'] * df['average weight of one loose cut tablet']
            )
            validation_flags['data_warnings'].append(
                "Total weight column added based on calculations"
            )

        # Remove outliers using the IQR method with validation
        def remove_outliers(df, columns):
            df_clean = df.copy()
            outliers_removed = False
            
            for column in columns:
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                outliers = df_clean[
                    (df_clean[column] < lower) | 
                    (df_clean[column] > upper)
                ]
                
                if not outliers.empty:
                    outliers_removed = True
                    validation_flags['data_warnings'].append(
                        f"Removed {len(outliers)} outliers from {column}"
                    )
                    
                df_clean = df_clean[
                    (df_clean[column] >= lower) & 
                    (df_clean[column] <= upper)
                ]
                
            return df_clean, outliers_removed

        df, had_outliers = remove_outliers(df, list(essential_cols.keys()))
        
        if had_outliers:
            validation_flags['corrected_values'] = True

        # Safe division for Weight-to-Strength Ratio with validation
        if 'weight-to-strength ratio' not in df.columns:
            df['weight-to-strength ratio'] = np.where(
                df['active pharmaceutical ingredient strength (mg)'] != 0,
                df['total weight of tablets without mixed packaging'] / 
                df['active pharmaceutical ingredient strength (mg)'],
                np.nan
            )
            
            if df['weight-to-strength ratio'].isna().any():
                validation_flags['data_warnings'].append(
                    "Found invalid weight-to-strength ratios (division by zero)"
                )

        # Store validation flags in the DataFrame metadata
        df.attrs['validation_flags'] = validation_flags
        
        if df.empty:
            raise ValueError("Data processing resulted in an empty DataFrame")
            
        return df
        
    except Exception as e:
        raise Exception(f"Error in data processing: {str(e)}")
        
# Initialize session state data
def initialize_session_data():
    try:
        # Initialize session state variables if they don't exist
        if 'raw_data' not in st.session_state:
            st.session_state.raw_data = pd.DataFrame()
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = pd.DataFrame()
            
        # Load raw data if empty
        if st.session_state.raw_data.empty:
            raw_data = load_data()
            if raw_data.empty:
                st.error("Failed to load raw data from Google Sheets")
                return
            st.session_state.raw_data = raw_data
        
        # Process data if needed
        if st.session_state.processed_data.empty:
            processed_data = load_and_prepare_data(st.session_state.raw_data)
            if processed_data.empty:
                st.error("Failed to process data")
                return
            st.session_state.processed_data = processed_data
            
    except Exception as e:
        st.error(f"Error initializing session data: {str(e)}")


# Main UI Section
st.title("Drug Inventory Tracker")

if st.button("Reload Data", key="reload_data_button"):
    # Clear session state
    st.session_state.raw_data = pd.DataFrame()
    st.session_state.processed_data = pd.DataFrame()
    st.rerun()  # Rerun the app to reload data

# Initialize session state data
initialize_session_data()

# Check if data is available before proceeding
if 'processed_data' in st.session_state and not st.session_state.processed_data.empty:
    st.success("Data loaded and processed successfully!")
else:
    st.warning("No data available after processing. Please check the data.")
    # Exit early if no data is available
    st.stop()

def create_strength_based_models(data, model_type, test_size, hyperparams={}):
    """
    Creates and trains models for different strength categories of medications.
    Returns a dictionary of models and their metrics for each strength category.
    """
    # Create category masks
    low_strength_mask = data['active pharmaceutical ingredient strength (mg)'] < 10
    medium_strength_mask = (data['active pharmaceutical ingredient strength (mg)'] >= 10) & \
                          (data['active pharmaceutical ingredient strength (mg)'] <= 100)
    high_strength_mask = data['active pharmaceutical ingredient strength (mg)'] > 100
    
    # Dictionary to store models and metrics
    models = {}
    metrics = {}
    
    # Dictionary of datasets
    strength_categories = {
        'Low Strength (<10mg)': data[low_strength_mask],
        'Medium Strength (10-100mg)': data[medium_strength_mask],
        'High Strength (>100mg)': data[high_strength_mask]
    }
    
    # Train models for each category
    for category, category_data in strength_categories.items():
        if not category_data.empty:
            model, mse, r2 = train_model(
                category_data,
                model_type,
                test_size,
                hyperparams
            )
            
            if model is not None:
                models[category] = model
                metrics[category] = {
                    'mse': mse,
                    'r2': r2,
                    'sample_size': len(category_data)
                }
    
    return models, metrics

def train_model(data, model_type, test_size, hyperparams={}):
    try:
        # Check for target column
        if 'number of tablets' not in data.columns:
            st.error("The target column 'number of tablets' is missing from the dataset.")
            return None, None, None

        # Feature selection - only use most relevant features
        important_features = [
            'active pharmaceutical ingredient strength (mg)',
            'average weight of one loose cut tablet',
            'total weight of tablets without mixed packaging',
            'number of rubber bands'
        ]
        
        # Add binary features
        binary_features = [
            'box (y/n)', 
            'loose cut (y/n)', 
            'full strips (y/n)'
        ]
        
        # Combine features
        selected_features = important_features + binary_features
        
        # Select only available features
        X = data[[col for col in selected_features if col in data.columns]]
        y = data['number of tablets']

        # Handle missing values
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna('unknown')

        # Convert categorical variables to dummies
        X = pd.get_dummies(X, drop_first=True)

        # Split with stratification based on quantity ranges
        y_bins = pd.qcut(y, q=5, duplicates='drop')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42,
            stratify=y_bins
        )

        # Select and train the model with adjusted parameters
        if model_type == 'Linear Regression':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
        elif model_type == 'Gradient Boosting':
            # Reduced complexity to prevent overfitting
            model = GradientBoostingRegressor(
                n_estimators=min(hyperparams.get("n_estimators", 100), 200),
                learning_rate=max(hyperparams.get("learning_rate", 0.1), 0.05),
                max_depth=min(hyperparams.get("max_depth", 3), 4),
                min_samples_split=max(hyperparams.get("min_samples_split", 5), 4),
                min_samples_leaf=max(hyperparams.get("min_samples_leaf", 3), 2),
                subsample=min(hyperparams.get("subsample", 0.8), 0.85),
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elif model_type == 'XGBoost':
            # Adjusted parameters to reduce overfitting
            model = XGBRegressor(
                n_estimators=min(hyperparams.get("n_estimators", 100), 150),
                learning_rate=max(hyperparams.get("learning_rate", 0.08), 0.05),
                max_depth=min(hyperparams.get("max_depth", 3), 4),
                min_child_weight=3,
                subsample=min(hyperparams.get("subsample", 0.8), 0.85),
                colsample_bytree=0.8,
                reg_alpha=0.01,
                reg_lambda=1.0,
                random_state=42
            )
            # Remove early_stopping_rounds and eval_set parameters
            model.fit(X_train, y_train, verbose=False)
            y_pred = model.predict(X_test)

        elif model_type == 'Random Forest':
            # Adjusted parameters to prevent overfitting
            model = RandomForestRegressor(
                n_estimators=min(hyperparams.get("n_estimators", 100), 150),
                max_depth=min(hyperparams.get("max_depth", 6), 8),
                min_samples_split=max(hyperparams.get("min_samples_split", 4), 3),
                min_samples_leaf=max(hyperparams.get("min_samples_leaf", 2), 2),
                max_features='sqrt',
                bootstrap=True,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        else:
            st.error("Invalid model type selected.")
            return None, None, None

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return model, mse, r2
        
    except Exception as e:
        st.error(f"An error occurred during model training: {str(e)}")
        return None, None, None

"""
---
"""

# Display the title and introductory information
# Display the title and introductory information
"""
**Track your drug inventory with ease!**
This dashboard displays inventory data directly from the uploaded datasheet.
"""

st.info("Below is the current inventory data in view-only mode.")

# Check for validation flags and display warnings
if not st.session_state.processed_data.empty:
    # Check for validation flags
    validation_flags = st.session_state.processed_data.attrs.get('validation_flags', {})
    
    # Display validation warnings if present
    if validation_flags.get('missing_unit_weights'):
        st.warning("⚠️ Missing unit weight values detected and corrected using median values.")
        
    if validation_flags.get('corrected_values'):
        st.info("ℹ️ Some values were automatically corrected during preprocessing.")
        
    if validation_flags.get('data_warnings'):
        with st.expander("View Data Quality Warnings"):
            for warning in validation_flags['data_warnings']:
                st.write(f"- {warning}")

    # Dynamically set columns_to_display to include all columns from the cleansed dataset
    columns_to_display = list(st.session_state.processed_data.columns)

    # View-only data table
    st.dataframe(
        st.session_state.processed_data[columns_to_display],
        column_config={
            "Active Pharmaceutical Ingredient Strength (mg)": st.column_config.NumberColumn(format="%.2f mg"),
            "Total weight of counted drug with mixed packing": st.column_config.NumberColumn(format="%.2f g"),
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.warning("No columns available to display")

"""
---
"""

# Model Training Section
st.subheader("Train a Model")

if not st.session_state.processed_data.empty:
    # Strength category selection
    training_mode = st.radio(
        "Select Training Mode",
        ["Single Model (All Data)", "Strength-Based Models"]
    )
    
    # Test size selection
    test_size = st.slider(
        "Select Test Size",
        min_value=0.00,
        max_value=1.00,
        value=0.20,
        step=0.01,
        help="Proportion of data to use for testing (0.00 - 1.00)"
    )
    
    # Model selection
    model_type = st.selectbox(
        "Select Model Type",
        ["Linear Regression", "Gradient Boosting", "XGBoost", "Random Forest"]
    )
    
    # Hyperparameter configuration
    hyperparams = {}
    if model_type in ["Gradient Boosting", "XGBoost", "Random Forest"]:
        with st.expander("Model Hyperparameters"):
            try:
                if model_type in ["Gradient Boosting", "XGBoost"]:
                    n_estimators = st.text_input("Number of Trees (100-1000)", "300")
                    hyperparams["n_estimators"] = int(n_estimators)
                    if not (100 <= hyperparams["n_estimators"] <= 1000):
                        st.warning("Number of trees should be between 100 and 1000")
                        
                    learning_rate = st.text_input("Learning Rate (0.01-0.3)", "0.1")
                    hyperparams["learning_rate"] = float(learning_rate)
                    if not (0.01 <= hyperparams["learning_rate"] <= 0.3):
                        st.warning("Learning rate should be between 0.01 and 0.3")
                        
                    max_depth = st.text_input("Max Depth (3-10)", "6")
                    hyperparams["max_depth"] = int(max_depth)
                    if not (3 <= hyperparams["max_depth"] <= 10):
                        st.warning("Max depth should be between 3 and 10")
                        
                    subsample = st.text_input("Subsample Ratio (0.5-1.0)", "0.8")
                    hyperparams["subsample"] = float(subsample)
                    if not (0.5 <= hyperparams["subsample"] <= 1.0):
                        st.warning("Subsample ratio should be between 0.5 and 1.0")
                        
                elif model_type == "Random Forest":
                    n_estimators = st.text_input("Number of Trees (100-500)", "200")
                    hyperparams["n_estimators"] = int(n_estimators)
                    if not (100 <= hyperparams["n_estimators"] <= 500):
                        st.warning("Number of trees should be between 100 and 500")
                        
                    max_depth = st.text_input("Max Depth (3-20)", "10")
                    hyperparams["max_depth"] = int(max_depth)
                    if not (3 <= hyperparams["max_depth"] <= 20):
                        st.warning("Max depth should be between 3 and 20")
                        
                    min_samples_split = st.text_input("Min Samples Split (2-10)", "2")
                    hyperparams["min_samples_split"] = int(min_samples_split)
                    if not (2 <= hyperparams["min_samples_split"] <= 10):
                        st.warning("Min samples split should be between 2 and 10")
                        
                    min_samples_leaf = st.text_input("Min Samples Leaf (1-4)", "1")
                    hyperparams["min_samples_leaf"] = int(min_samples_leaf)
                    if not (1 <= hyperparams["min_samples_leaf"] <= 4):
                        st.warning("Min samples leaf should be between 1 and 4")
            except ValueError as e:
                st.error(f"Please enter valid numeric values for all hyperparameters: {str(e)}")
    
    # Train model button
    if st.button("Train Model", key="train_model_button"):
        with st.spinner("Training model..."):
            if training_mode == "Single Model (All Data)":
                # Existing single model training
                model, mse, r2 = train_model(
                    st.session_state.processed_data,
                    model_type,
                    test_size,
                    hyperparams
                )
                
                if model is not None:
                    st.success("Model trained successfully!")
                    st.write(f"Mean Squared Error: {mse:.4f}")
                    st.write(f"R² Score: {r2:.4f}")
                    
                    # Save model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_metrics = {
                        'mse': mse,
                        'r2': r2
                    }
            
            else:  # Strength-Based Models
                models, metrics = create_strength_based_models(
                    st.session_state.processed_data,
                    model_type,
                    test_size,
                    hyperparams
                )
                
                if models:
                    st.success("Strength-based models trained successfully!")
                    
                    # Display metrics for each category
                    for category, category_metrics in metrics.items():
                        st.write(f"\n**{category}**")
                        st.write(f"Sample size: {category_metrics['sample_size']}")
                        st.write(f"Mean Squared Error: {category_metrics['mse']:.4f}")
                        st.write(f"R² Score: {category_metrics['r2']:.4f}")
                    
                    # Save models in session state
                    st.session_state.strength_models = models
                    st.session_state.strength_metrics = metrics
                    
                    # Create a summary visualization
                    summary_data = []
                    for category, metric in metrics.items():
                        summary_data.append({
                            'Category': category,
                            'MSE': metric['mse'],
                            'R²': metric['r2'],
                            'Sample Size': metric['sample_size']
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.write("\n**Model Performance Summary**")
                    st.dataframe(summary_df)
else:
    st.error("No data available for training. Please check your dataset.")
    
"""
---
"""

# Add the Predict Quantity Section
st.subheader("Predict Quantity")
st.markdown("Predict medication quantity based on weight and packaging details.")

with st.form("predict_quantity_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Step 1: Input Medication Details")
        total_weight = st.number_input(
            "Total Weight (g)",
            min_value=0.0,
            value=50.0,
            format="%.4f",  # Increased precision to 4 decimal places
            help="Enter the total weight of the medication including packaging and accessories in grams"
        )
        
        unit_weight = st.number_input(
            "Unit Weight (g)",
            min_value=0.0,
            value=0.5,
            format="%.4f",  # Increased precision to 4 decimal places
            help="Enter the weight of a single tablet/capsule or weight per unit of packaging in grams"
        )

        num_rubber_bands = st.number_input(
            "Number of Rubber Bands",
            min_value=0,
            value=0,
            step=1,
            help="Enter the number of rubber bands"
        )
        
        st.markdown("#### Packaging Type")
        loose_cut = st.checkbox("Loose Cut", help="Check if the medication is in loose cut form")
        loose_strip = st.checkbox("Loose Strip", help="Check if the medication is in loose strip form")
        full_strip = st.checkbox("Full Strip", help="Check if the medication is in full strip form")
        box = st.checkbox("Box", help="Check if the medication is in a box")
        
        # Accessories section removed since we only handle rubber bands in the number input above
    
    with col2:
        st.markdown("#### Step 2: Predict Quantity")
        st.markdown("Click the button below to predict the quantity based on your inputs.")
        
        if 'trained_model' not in st.session_state and 'strength_models' not in st.session_state:
            st.warning("Please train a model first before making predictions.")
        
        predict_button = st.form_submit_button("Predict Quantity")
        
        if predict_button:
            if total_weight <= 0 or unit_weight <= 0:
                st.error("Please enter valid weights greater than 0.")
            else:
                try:
                    # Get the feature names from the trained model
                    if 'trained_model' in st.session_state:
                        model = st.session_state.trained_model
                        if hasattr(model, 'feature_names_in_'):
                            feature_names = model.feature_names_in_
                        else:
                            X = st.session_state.processed_data.drop(columns=['number of tablets'])
                            feature_names = X.columns
                            
                        # Create DataFrame with all features initialized to 0
                        input_data = pd.DataFrame(0, index=[0], columns=feature_names)
                        
                        # Update features
                        input_data['total weight of tablets without mixed packaging'] = total_weight
                        input_data['average weight of one loose cut tablet'] = unit_weight
                        
                        # Set packaging type
                        if 'box (y/n)' in input_data.columns:
                            input_data['box (y/n)'] = 1 if box else 0
                        if 'loose cut (y/n)' in input_data.columns:
                            input_data['loose cut (y/n)'] = 1 if loose_cut else 0
                        if 'loose strip (y/n)' in input_data.columns:
                            input_data['loose strip (y/n)'] = 1 if loose_strip else 0
                        if 'full strip (y/n)' in input_data.columns:
                            input_data['full strip (y/n)'] = 1 if full_strip else 0
                            
                        # Set accessories
                        if 'number of rubber bands' in input_data.columns:
                            input_data['number of rubber bands'] = num_rubber_bands
                        if 'rubber band (y/n)' in input_data.columns:
                            input_data['rubber band (y/n)'] = 1 if num_rubber_bands > 0 else 0
                            
                        # Make prediction
                        prediction = model.predict(input_data)[0]
                        st.success(f"Predicted Quantity: {int(round(prediction))} units")
                        
                        # Display confidence metrics if available
                        if 'model_metrics' in st.session_state:
                            st.info(f"Model R² Score: {st.session_state.model_metrics['r2']:.4f}")
                            
                    elif 'strength_models' in st.session_state:
                        # Similar logic for strength-based models
                        default_category = list(st.session_state.strength_models.keys())[0]
                        model = st.session_state.strength_models[default_category]
                        
                        if hasattr(model, 'feature_names_in_'):
                            feature_names = model.feature_names_in_
                            input_data = pd.DataFrame(0, index=[0], columns=feature_names)
                            
                            # Update features
                            input_data['total weight of tablets without mixed packaging'] = total_weight
                            input_data['average weight of one loose cut tablet'] = unit_weight
                            
                            # Update packaging type and accessories
                            for col in feature_names:
                                if 'box' in col.lower():
                                    input_data[col] = 1 if box else 0
                                elif 'loose cut' in col.lower():
                                    input_data[col] = 1 if loose_cut else 0
                                elif 'loose strip' in col.lower():
                                    input_data[col] = 1 if loose_strip else 0
                                elif 'full strip' in col.lower():
                                    input_data[col] = 1 if full_strip else 0
                                elif 'number of rubber bands' in col.lower():
                                    input_data[col] = num_rubber_bands
                                elif col.lower() == 'rubber band (y/n)':
                                    input_data[col] = 1 if num_rubber_bands > 0 else 0
                            
                            prediction = model.predict(input_data)[0]
                            st.success(f"Predicted Quantity: {int(round(prediction))} units")
                            
                            if 'strength_metrics' in st.session_state:
                                st.info(f"Model R² Score: {st.session_state.strength_metrics[default_category]['r2']:.4f}")
                    else:
                        st.error("No trained model available. Please train a model first.")
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.error("Please ensure all required fields are filled correctly.")
                    
"""
---
"""

# Section: Add New Data
st.subheader("Add New Data")
with st.form("add_data_form", clear_on_submit=True):
    # Required fields marked with asterisk
    brand_name = st.text_input("Brand Name *")
    drug_code = st.text_input("Drug Code *")
    active_ingredient = st.text_input("Active Pharmaceutical Ingredients *")
    strength_mg = st.text_input("Active Pharmaceutical Ingredient Strength (mg) *")
    dosage_form = st.text_input("Dosage Form *")
    combination_drug = st.selectbox("Combination Drug (Y/N) *", options=["", "Y", "N"])
    special_formulation = st.selectbox("Special Formulation (Y/N) *", options=["", "Y", "N"])
    tablets_per_box = st.text_input("No. of Tablets/Unit per Box", value="N.A.")
    avg_weight_box = st.text_input("Average Weight of One Box of Medication with PIL (g)", value="N.A.")
    tablets_per_strip = st.text_input("No. of Tablets/Unit per Strip *")
    avg_weight_strip = st.text_input("Average Weight of One Strip/Unit of Medication (g)", value="N.A.")
    avg_weight_tablet = st.text_input("Average Weight of One Loose Cut Tablet", value="N.A.")
    full_strips = st.selectbox("Full Strips (Y/N) *", options=["", "Y", "N"])
    box = st.selectbox("Box (Y/N) *", options=["", "Y", "N"])
    loose_cut = st.selectbox("Loose Cut (Y/N) *", options=["", "Y", "N"])
    rubber_bands = st.text_input("Number of Rubber Bands *")
    total_tablets = st.text_input("Number of Tablets *")
    total_weight = st.text_input("Total Weight of Counted Drug with Mixed Packing (g) *")
    
    submitted = st.form_submit_button("Add Data")
    if submitted:
        # Validate required fields
        required_fields = {
            "Brand Name": brand_name,
            "Drug Code": drug_code,
            "Active Pharmaceutical Ingredients": active_ingredient,
            "Active Pharmaceutical Ingredient Strength (mg)": strength_mg,
            "Dosage Form": dosage_form,
            "Combination Drug (Y/N)": combination_drug,
            "Special Formulation (Y/N)": special_formulation,
            "no. of tablet/unit per strip": tablets_per_strip,
            "Full Strips (Y/N)": full_strips,
            "Box (Y/N)": box,
            "Loose Cut (Y/N)": loose_cut,
            "Number of Rubber Bands": rubber_bands,
            "Number of Tablets": total_tablets,
            "Total Weight of Counted Drug with Mixed Packing (g)": total_weight
        }
        
        # Check if any required field is empty
        empty_fields = [field for field, value in required_fields.items() if not value or value.isspace()]
        
        if empty_fields:
            st.error(f"Please fill in all required fields marked with *: {', '.join(empty_fields)}")
        else:
            new_row = {
                "Brand Name": brand_name,
                "Drug Code": drug_code,
                "Active Pharmaceutical Ingredients": active_ingredient,
                "Active Pharmaceutical Ingredient Strength (mg)": strength_mg,
                "Dosage Form": dosage_form,
                "Combination Drug (Y/N)": combination_drug,
                "Special Formulation (Y/N)": special_formulation,
                "no. of tablets/unit per box": tablets_per_box,
                "average weight of one box of medication with PIL (g)": avg_weight_box,
                "no. of tablet/unit per strip": tablets_per_strip,
                "Average weight of one strip/unit of medication (g)": avg_weight_strip,
                "Average weight of one loose cut tablet": avg_weight_tablet,
                "Full Strips (Y/N)": full_strips,
                "Box (Y/N)": box,
                "Loose Cut (Y/N)": loose_cut,
                "Number of Rubber Bands": rubber_bands,
                "Number of Tablets": total_tablets,
                "Total Weight of Counted Drug with Mixed Packing (g)": total_weight
            }
            
            # Save to Google Sheets
            save_data(new_row)
            st.success("Data added successfully!")
