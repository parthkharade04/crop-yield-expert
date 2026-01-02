import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def train_models():
    print("Loading master dataset...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'master_dataset.csv')
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data verification failed. {DATA_PATH} not found.")
        
    df = pd.read_csv(DATA_PATH)
    
    # Feature Engineering Check
    # We want to predict 'Production' or 'Yield' (Production/Area)?
    # The prompt implies 'Production' directly, but Yield is usually better.
    # However, let's stick to Production as the target variable as per original project, 
    # BUT add Area as a feature. 
    # Actually, Neural/ML models often struggle with extensive vs intensive properties.
    # Let's target Production.
    
    # Features & Target
    X = df.drop(columns=['Production'])
    y = df['Production']
    
    # Preprocessing Pipeline
    # Categorical: District_Name, Season, Crop
    # Numerical: Area, Actual_Rainfall, pH_min, pH_max, pH_min_Range, pH_max_Range
    
    cat_features = ['District_Name', 'Season', 'Crop']
    num_features = ['Area', 'Actual_Rainfall', 'pH_min', 'pH_max', 'pH_min_Range', 'pH_max_Range']
    
    # Check if we have missing values in features
    # df.isnull().sum()
    
    from sklearn.impute import SimpleImputer
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1), # 50 trees for speed initially
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, n_jobs=-1)
    }
    
    best_model = None
    best_score = -np.inf
    
    results = []
    
    print("\nTraining Models...")
    print("-" * 50)
    for name, model in models.items():
        start = time.time()
        
        # Create full pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        
        pipeline.fit(X_train, y_train)
        
        preds = pipeline.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        duration = time.time() - start
        print(f"{name:20} | R2: {r2:.4f} | RMSE: {rmse:.0f} | Time: {duration:.2f}s")
        
        results.append({'Model': name, 'R2': r2, 'RMSE': rmse})
        
        if r2 > best_score:
            best_score = r2
            best_model = pipeline
            
    print("-" * 50)
    print(f"Best Model: {best_model['model']} with R2: {best_score:.4f}")
    
    # Save Best Model
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    save_path = os.path.join(MODEL_DIR, 'crop_yield_model.pkl')
    joblib.dump(best_model, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_models()
