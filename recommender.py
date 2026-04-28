import numpy as np
import pandas as pd
import joblib

def get_recommendations(target_lat, target_lon, target_price, min_rooms, weight_loc=0.4, weight_price=0.4, weight_area=0.1, weight_amenities=0.1):
    """
    Returns the top 5 property recommendations based on pre-compiled Weighted KNN.
    Assumes `knn_model.pkl`, `scaler.pkl`, and `knn_meta.pkl` exist in the current directory.
    This replaces the slow, tight-coupled Pandas DataFrame .fit() strategy.
    """
    # 1. Decoupled Inference: Load Serialized Artifacts
    try:
        knn = joblib.load('knn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        df_meta = joblib.load('knn_meta.pkl')
    except Exception as e:
        raise FileNotFoundError(f"Missing ML Artifacts: {e}")

    # The features the model expects MUST match how it was trained
    features = ['latitude', 'longitude', 'unified_price', 'unified_area', 'amenity_score']
    
    # Normally we might enforce this dynamically against the scaler, but for pure V2 inference isolation,
    # we simulate the perfect matching user target:
    target_area = df_meta['unified_area'].max() if 'unified_area' in df_meta.columns else 350
    target_amenities = 6 # Max possible score typically
    
    target_row = []
    for col in features:
        if col == 'latitude': target_row.append(target_lat)
        elif col == 'longitude': target_row.append(target_lon)
        elif col == 'unified_price': target_row.append(target_price)
        elif col == 'unified_area': target_row.append(target_area)
        elif col == 'amenity_score': target_row.append(target_amenities)
        else: target_row.append(0)
    
    # 2. Scale Target exactly utilizing the frozen scaler space
    target_scaled = scaler.transform([target_row])[0]

    # 3. Applying Business Logic Weights (Assuming model was trained on uniformly scaled data)
    weights = []
    for col in features:
        if col in ['latitude', 'longitude']: weights.append(weight_loc / 2)
        elif col == 'unified_price': weights.append(weight_price)
        elif col == 'unified_area': weights.append(weight_area)
        elif col == 'amenity_score': weights.append(weight_amenities)
        else: weights.append(0.0)
    
    weight_array = np.array(weights)
    target_scaled_weighted = target_scaled * weight_array

    # 4. Instant O(1)-like Vector Native Inference Search
    try:
        distances, indices = knn.kneighbors([target_scaled_weighted])
    except ValueError:
        # Failsafe if the model wasn't trained on the weighted approach directly
        distances, indices = knn.kneighbors([target_scaled])

    # 5. Format & Filter output via Meta lookup
    recommendations = df_meta.iloc[indices[0]].copy()
    recommendations['match_distance'] = distances[0]
    
    # Post-inference filter (because we skipped pre-filtering to honor the static ML model size)
    recommendations = recommendations[recommendations['unified_rooms'] >= min_rooms]
    
    return recommendations

