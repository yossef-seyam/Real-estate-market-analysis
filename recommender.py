import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

def get_recommendations(df, target_lat, target_lon, target_price, min_rooms, weight_loc=0.4, weight_price=0.4, weight_area=0.1, weight_amenities=0.1):
    """
    Returns the top 5 property recommendations based on Weighted KNN (Scaled).
    """
    # 1. Filter by strict minimum requirements (e.g. at least N rooms)
    # We only want to search among properties that actually fit the core criteria.
    df_filtered = df[df['unified_rooms'] >= min_rooms].copy()
    
    if df_filtered.empty:
        return pd.DataFrame() # No properties found

    # 2. Select ML Features
    # Latitude, Longitude, Price, Area, and maybe some key amenities
    features = ['latitude', 'longitude', 'unified_price', 'unified_area']
    
    # Optional: include amenity scores if they exist
    amenity_cols = [col for col in df.columns if col.startswith('has_')]
    if amenity_cols:
        # Create an aggregated "Amenities Score" (how many amenities it has)
        df_filtered['amenity_score'] = df_filtered[amenity_cols].sum(axis=1)
        features.append('amenity_score')
    else:
        df_filtered['amenity_score'] = 0

    # Ensure no NaN in our feature columns (from cleaning, should be fine, but just in case)
    df_filtered = df_filtered.dropna(subset=features)
    if df_filtered.empty:
        return pd.DataFrame()

    # 3. Scaling (Critical Step for KNN)
    # Price is in Millions, Lat/Long is in decimals. MinMaxScaler makes everything 0 to 1.
    scaler = MinMaxScaler()
    X = df_filtered[features].values
    X_scaled = scaler.fit_transform(X)

    # 4. Preparing the Target Request
    # We create a dummy "Ideal Property" row
    # For area and amenities, we assume the user wants the maximum possible naturally
    target_area = df_filtered['unified_area'].max()
    target_amenity = df_filtered['amenity_score'].max() if 'amenity_score' in features else 0
    
    target_row = []
    for col in features:
        if col == 'latitude': target_row.append(target_lat)
        elif col == 'longitude': target_row.append(target_lon)
        elif col == 'unified_price': target_row.append(target_price)
        elif col == 'unified_area': target_row.append(target_area)
        elif col == 'amenity_score': target_row.append(target_amenity)
        else: target_row.append(0)
    
    # Scale the target row using the SAME scaler
    target_scaled = scaler.transform([target_row])[0]

    # 5. Applying Business Logic Weights
    # Multiply the scaled variables by their importance
    weights = []
    for col in features:
        if col in ['latitude', 'longitude']: weights.append(weight_loc / 2) # Split 40% between lat/long
        elif col == 'unified_price': weights.append(weight_price) # 40%
        elif col == 'unified_area': weights.append(weight_area) # 10%
        elif col == 'amenity_score': weights.append(weight_amenities) # 10%
        else: weights.append(0.0)
    
    weight_array = np.array(weights)
    
    X_scaled_weighted = X_scaled * weight_array
    target_scaled_weighted = target_scaled * weight_array

    # 6. K-Nearest Neighbors Execution
    n_neighbors = min(5, len(X_scaled_weighted))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(X_scaled_weighted)

    distances, indices = knn.kneighbors([target_scaled_weighted])

    # 7. Format output
    recommendations = df_filtered.iloc[indices[0]].copy()
    recommendations['match_distance'] = distances[0]
    
    return recommendations
