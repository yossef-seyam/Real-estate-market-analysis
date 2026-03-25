import pandas as pd
import numpy as np
import re
import os
import glob

def clean_data(output_file="cleaned_properties.csv"):
    """
    Reads the raw scraped listings from ALL region files, cleans the data, 
    merges them, and prepares it for ML modeling.
    This fulfills the Data Integration & Flexible Schema step of our architecture.
    """
    # Find all CSV files that start with aqarmap_listings_
    all_files = glob.glob("aqarmap_listings_*.csv")
    
    # Also include the original aqarmap_listings.csv if it exists from previous runs
    if os.path.exists("aqarmap_listings.csv") and "aqarmap_listings.csv" not in all_files:
        all_files.append("aqarmap_listings.csv")
    
    if not all_files:
        print(f"❌ Error: No aqarmap_listings_*.csv files found. Please run scraper.py first!")
        return
        
    print(f"-> Found {len(all_files)} raw data files. Loading and merging...")
    
    df_list = []
    for file in all_files:
        print(f"   - Loading {file}...")
        # low_memory=False prevents pandas from guessing data types in chunks, removing DtypeWarnings
        df_list.append(pd.read_csv(file, low_memory=False))
        
    df = pd.concat(df_list, ignore_index=True)
    print(f"   Combined Original Data Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # ==========================================
    # 1. DROP DUPLICATES & MISSING CORE DATA
    # ==========================================
    # We can't train an ML model on a house without a price or area!
    df = df.drop_duplicates(subset=['url'], keep='last')
    
    # Drop rows where essential ML features are completely missing
    essential_cols = ['price', 'area']
    existing_essentials = [col for col in essential_cols if col in df.columns]
    df = df.dropna(subset=existing_essentials)

    # ==========================================
    # 2. STANDARDIZE COLUMN NAMES
    # ==========================================
    # This prepares the data for integration with Property Finder later
    rename_mapping = {
        'price': 'unified_price',
        'area': 'unified_area',
        'rooms': 'unified_rooms',
        'bathrooms': 'unified_bathrooms'
    }
    df = df.rename(columns={k: v for k, v in rename_mapping.items() if k in df.columns})

    # ==========================================
    # 3. CLEAN NUMERIC FEATURES (Price, Area, etc.)
    # ==========================================
    def extract_numbers(text):
        if pd.isna(text):
            return np.nan
        # Fix the "one extra zero" bug: drop any decimal points and everything after it
        # E.g. "90,000,000.00" -> "90,000,000"
        text_no_decimal = str(text).split('.')[0].replace(',', '')
        # Find all remaining numbers
        numbers = re.findall(r'\d+', text_no_decimal)
        return int(''.join(numbers)) if numbers else np.nan

    print("-> Cleaning numeric features (Price, Area, Rooms)...")
    if 'unified_price' in df.columns:
        df['unified_price'] = df['unified_price'].apply(extract_numbers)
    
    if 'unified_area' in df.columns:
        df['unified_area'] = df['unified_area'].apply(extract_numbers)

    if 'unified_rooms' in df.columns:
        # Sometimes it says "10+", we just want the 10
        df['unified_rooms'] = df['unified_rooms'].apply(extract_numbers)

    if 'unified_bathrooms' in df.columns:
        df['unified_bathrooms'] = df['unified_bathrooms'].apply(extract_numbers)

    # ==========================================
    # 4. HANDLE MISSING VALUES (IMPUTATION)
    # ==========================================
    # For ML, we fill missing categoricals with 'Unknown' and median for numericals 
    # if it makes sense (e.g., if bathrooms is missing, assume 1)
    if 'unified_bathrooms' in df.columns:
        df['unified_bathrooms'] = df['unified_bathrooms'].fillna(1)
        
    if 'unified_rooms' in df.columns:
        # If rooms are missing, we might use median rooms for that area
        df['unified_rooms'] = df['unified_rooms'].fillna(df['unified_rooms'].median())

    # ==========================================
    # 5. FEATURE ENGINEERING: AMENITIES FLAGS
    # ==========================================
    # Machine learning algorithms need numbers, not strings like "Balcony, Elevator"
    # We create a column for key amenities. 1 if present, 0 if not.
    if 'amenities' in df.columns:
        print("-> Engineering amenities into ML features...")
        df['amenities'] = df['amenities'].fillna('')
        
        # Define the most important amenities for property valuation
        key_amenities = ['elevator', 'security', 'balcony', 'pool', 'garden', 'parking']
        
        for amenity in key_amenities:
            # Check if the amenity exists in the string (case insensitive)
            df[f'has_{amenity}'] = df['amenities'].str.lower().str.contains(amenity).astype(int)

    # Drop fully null columns
    df = df.dropna(axis=1, how='all')

    print("-> Data harmonization complete!")
    print(f"   Final Cleaned Data Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # ==========================================
    # 6. EXPORT CLEANED DATA
    # ==========================================
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("="*50)
    print(f"✅ Data ready for Machine Learning / Dashboard!")
    print(f"✅ Cleaned data saved securely to: {output_file}")
    print("="*50)

if __name__ == "__main__":
    clean_data()
