import pandas as pd
import joblib
import os
import sys

def load_resources():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # The app.py is inside 'python_implementation', so models is at 'python_implementation/models'
        # which is just os.path.join(base_dir, 'models')
        model_path = os.path.join(base_dir, 'models', 'crop_yield_model.pkl')
            
        print(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print("Error: Model file not found. Please run 'src/training.py' first.")
        sys.exit(1)

def get_user_input():
    print("\n--- Crop Yield Estimator ---")
    print("Please enter the following details:")
    
    district = input("District Name (e.g., AHMEDNAGAR, PUNE): ").strip().upper()
    season = input("Season (Kharif, Rabi, Summer, Whole Year): ").strip()
    crop = input("Crop Name (e.g., Maize, Wheat, Cotton): ").strip()
    try:
        area = float(input("Area (in Hectares): "))
        rainfall = float(input("Estimated Rainfall (mm) [Typical Kharif: 500-1000]: "))
        ph = float(input("Soil pH (Typical: 6.5 - 7.5): "))
    except ValueError:
        print("Invalid number entered.")
        return None

    # Construct DataFrame with exactly the same columns as training
    # Columns: Area, Actual_Rainfall, pH_min, pH_max, pH_min_Range, pH_max_Range, District_Name, Season, Crop
    # Note: The model pipeline expects these specific columns.
    
    # We will use the user's pH for all pH columns for simplicity in this demo, 
    # or ideal vals if we had the lookup loaded. 
    # For a robust app, we'd load the District_ph.csv lookup again.
    # Let's simplify: Set min/max to the input pH (assuming homogenous soil).
    
    data = {
        'District_Name': [district],
        'Season': [season],
        'Crop': [crop],
        'Area': [area],
        'Actual_Rainfall': [rainfall],
        'pH_min': [ph],
        'pH_max': [ph],
        'pH_min_Range': [5.5], # Default dummy valid range
        'pH_max_Range': [8.5]
    }
    
    return pd.DataFrame(data)

def main():
    model = load_resources()
    
    while True:
        df = get_user_input()
        if df is not None:
            try:
                prediction = model.predict(df)[0]
                print(f"\n>>> Predicted Production: {prediction:,.2f} Tonnes")
                print(f">>> Predicted Yield: {prediction/df['Area'][0]:,.2f} Tonnes/Hectare")
            except Exception as e:
                print(f"Prediction Error: {e}")
                print("Make sure the District/Crop names match the training data (e.g., Use 'WHEAT' not 'wheat' if case sensitive, though pipeline handles some).")
        
        cont = input("\nTry another? (y/n): ")
        if cont.lower() != 'y':
            break

if __name__ == "__main__":
    main()
