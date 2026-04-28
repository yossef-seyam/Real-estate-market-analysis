import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import sys
import os

def train_and_export_models():
    print("🤖 Initializing Deep ML Training Sequence...")
    
    # 1. Load the pristine Parquet output
    parquet_path = "final_ml_dataset.parquet"
    if not os.path.exists(parquet_path):
        print(f"❌ Critical Error: {parquet_path} not found. Please run the cleaner.py pipeline first.")
        sys.exit(1)
        
    print(f"📂 Loading core dataset: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"   -> Successfully loaded {len(df)} records in memory.")

    # 2. Engineer Target Feature Vector
    # Ensuring our feature space aligns precisely with the V2 recommender footprint
    features = ['latitude', 'longitude', 'unified_price', 'unified_area', 'amenity_score']
    
    amenity_cols = [col for col in df.columns if col.startswith('has_')]
    if amenity_cols:
        print("   -> Calculating holistic amenity scores...")
        df['amenity_score'] = df[amenity_cols].sum(axis=1)
    else:
        df['amenity_score'] = 0

    # Strict isolation: We cannot fit a KNN against NaN features
    initial_len = len(df)
    df_clean = df.dropna(subset=features).copy()
    print(f"   -> Scrubbed {initial_len - len(df_clean)} unusable artifacts.")
    
    if df_clean.empty:
        print("❌ Critical Error: Filter bounds completely decimated the DataFrame. Aborting.")
        sys.exit(1)

    # 3. Model Preprocessing: MinMax Scaling
    print("⚖️ Initializing Vector Constraints (MinMaxScaler)...")
    scaler = MinMaxScaler()
    X = df_clean[features].values
    X_scaled = scaler.fit_transform(X)

    # 4. Applying Business Domain Logic (Baking Weights intimately into the structure)
    # Using the standard configuration requested in the architecture
    print("🧠 Forging weighted multi-dimensional vector array...")
    weight_loc = 0.4
    weight_price = 0.4
    weight_area = 0.1
    weight_amenities = 0.1
    
    weights = []
    for col in features:
        if col in ['latitude', 'longitude']: weights.append(weight_loc / 2) # 20% each
        elif col == 'unified_price': weights.append(weight_price)           # 40%
        elif col == 'unified_area': weights.append(weight_area)             # 10%
        elif col == 'amenity_score': weights.append(weight_amenities)       # 10%
        else: weights.append(0.0)
    
    weight_array = np.array(weights)
    X_scaled_weighted = X_scaled * weight_array

    # 5. Core Model Training
    print("🏃 Executing unsupervised Nearest Neighbors tree construction...")
    # Using 5 neighbors minimum, with aggressive Euclidean checks
    n_neighbors = min(5, len(X_scaled_weighted))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm='auto')
    knn.fit(X_scaled_weighted)

    # 6. Binary Artefact Serialisation
    print("💾 Saving core decoupling artifacts securely to disk via Joblib...")
    try:
        joblib.dump(knn, 'knn_model.pkl', compress=3)
        joblib.dump(scaler, 'scaler.pkl', compress=3)
        # We save df_clean as the meta lookup so indices returned by model.kneighbors perfectly align
        joblib.dump(df_clean, 'knn_meta.pkl', compress=3)
    except Exception as e:
        print(f"❌ Failed to write Pickles to OS: {e}")
        sys.exit(1)

    print("=" * 50)
    print("✅ Model Decoupling Successful!")
    print("   Artifacts securely generated:")
    print("   1. knn_model.pkl (Topological Data)")
    print("   2. scaler.pkl (Dimensionality normalizer)")
    print("   3. knn_meta.pkl (Referential Mapping Array)")
    print("=" * 50)


if __name__ == "__main__":
    train_and_export_models()
