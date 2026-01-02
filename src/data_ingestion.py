import pandas as pd
import numpy as np
import os

def load_data():
    """Lengths raw data and performs merging."""
    print("Loading raw data...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    
    crop_data = pd.read_csv(os.path.join(RAW_DIR, 'CropData.csv'))
    rainfall_data = pd.read_csv(os.path.join(RAW_DIR, 'MaharashtrastateRainfall.csv'))
    district_ph = pd.read_csv(os.path.join(RAW_DIR, 'District_ph.csv'))
    crop_ph = pd.read_csv(os.path.join(RAW_DIR, 'Crop_ph.csv'))
    
    # Clean column names
    crop_data.columns = [c.strip() for c in crop_data.columns]
    rainfall_data.columns = [c.strip() for c in rainfall_data.columns]
    
    # 1. Standardize District Names
    # CropData uses CAPS (AHMEDNAGAR), Rainfall uses CAPS, District_ph uses CAPS.
    # Just ensure consistency.
    crop_data['District_Name'] = crop_data['District_Name'].str.upper().str.strip()
    rainfall_data['District'] = rainfall_data['District'].str.upper().str.strip()
    district_ph['City'] = district_ph['City'].str.upper().str.strip() # 'City' is actually District
    
    # 2. Calculate Seasonal Rainfall
    print("Calculating Seasonal Rainfall...")
    rainfall_data = rainfall_data.replace('N.A.', np.nan)
    month_cols = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    for col in month_cols:
        rainfall_data[col] = pd.to_numeric(rainfall_data[col], errors='coerce').fillna(0)
        
    # Create Next Year's Jan/Feb/Mar for Rabi calculation
    # We shift the dataframe by -1 year to align next year's early months to current rows
    next_year_rain = rainfall_data.copy()
    next_year_rain['Year'] = next_year_rain['Year'] - 1 # Shift year purely for join
    next_year_rain = next_year_rain[['District', 'Year', 'January', 'February', 'March']]
    next_year_rain.columns = ['District', 'Year', 'Jan_Next', 'Feb_Next', 'Mar_Next']
    
    rainfall_expanded = pd.merge(rainfall_data, next_year_rain, on=['District', 'Year'], how='left')
    
    # Seasonal Logic
    # Kharif: Jun-Oct
    rainfall_expanded['Rain_Kharif'] = (rainfall_expanded['June'] + rainfall_expanded['July'] + 
                                       rainfall_expanded['August'] + rainfall_expanded['September'] + 
                                       rainfall_expanded['October'])
    
    # Rabi: Oct-Dec + Jan_Next-Mar_Next (Standard broad definition for India)
    rainfall_expanded['Rain_Rabi'] = (rainfall_expanded['October'] + rainfall_expanded['November'] + 
                                     rainfall_expanded['December'] + rainfall_expanded['Jan_Next'].fillna(0) + 
                                     rainfall_expanded['Feb_Next'].fillna(0) + rainfall_expanded['Mar_Next'].fillna(0))
    
    # Summer: Mar-May
    rainfall_expanded['Rain_Summer'] = (rainfall_expanded['March'] + rainfall_expanded['April'] + 
                                       rainfall_expanded['May'])
    
    # Whole Year: Annual Total
    rainfall_expanded['Rain_WholeYear'] = rainfall_expanded['Annual_Total'] # or sum(Jan-Dec)
    
    # 3. Merge Rainfall into Crop Data
    # Map Crop Data Seasons to Rainfall Columns
    # CropData Seasons: 'Kharif     ', 'Rabi       ', 'Summer     ', 'Whole Year ', 'Autumn     '
    crop_data['Season'] = crop_data['Season'].str.strip()
    
    season_map = {
        'Kharif': 'Rain_Kharif',
        'Autumn': 'Rain_Kharif', # Treat as Kharif
        'Rabi': 'Rain_Rabi',
        'Summer': 'Rain_Summer',
        'Whole Year': 'Rain_WholeYear'
    }
    
    # We need a way to join effectively.
    # Let's pivot/melt the rainfall or just lookup. Lookup is safer for irregular matches.
    # Join on District + Year, then select column based on Season.
    
    merged_df = pd.merge(crop_data, rainfall_expanded, 
                         left_on=['District_Name', 'Crop_Year'], 
                         right_on=['District', 'Year'], 
                         how='left')
    
    def get_rain(row):
        season_col = season_map.get(row['Season'])
        if season_col and season_col in row:
            return row[season_col]
        return np.nan

    merged_df['Actual_Rainfall'] = merged_df.apply(get_rain, axis=1)
    
    # 4. Merge Soil pH
    # CropData District -> District_ph City
    merged_df = pd.merge(merged_df, district_ph, left_on='District_Name', right_on='City', how='left')
    
    # 5. Merge Crop pH requirements (Optional features)
    crop_ph['Crop'] = crop_ph['Crop'].str.strip()
    merged_df['Crop'] = merged_df['Crop'].str.strip()
    merged_df = pd.merge(merged_df, crop_ph, on='Crop', how='left')
    
    # Drop redundant columns
    cols_to_keep = ['District_Name', 'Crop_Year', 'Season', 'Crop', 'Area', 'Production', 
                    'Actual_Rainfall', 'pH_min', 'pH_max', 'pH_min_Range', 'pH_max_Range']
    
    final_df = merged_df[cols_to_keep].copy()
    
    # Handle pH missing values (fill with global means if district missing)
    final_df['pH_min'] = final_df['pH_min'].fillna(final_df['pH_min'].mean())
    final_df['pH_max'] = final_df['pH_max'].fillna(final_df['pH_max'].mean())
    
    # Drop rows where Actual_Rainfall is massive outlier or NaN (unless we want to fill it)
    # Be strict for now
    before_len = len(final_df)
    final_df = final_df.dropna(subset=['Actual_Rainfall', 'Production'])
    print(f"Dropped {before_len - len(final_df)} rows due to missing rainfall/production data.")
    
    # Save
    out_path = os.path.join(BASE_DIR, 'data', 'processed', 'master_dataset.csv')
    final_df.to_csv(out_path, index=False)
    print(f"Saved processed dataset to {out_path} with {len(final_df)} rows.")

if __name__ == "__main__":
    load_data()
