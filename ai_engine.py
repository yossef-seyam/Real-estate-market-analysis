"""
AI recommendation engine using the trained Random Forest model
and Google Gemini for natural language investment advice.
"""

import os
import pickle
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_model.pkl")
PPSM_PATH = os.path.join(os.path.dirname(__file__), "district_ppsm.pkl")

# Yield table matching clean_data.py
YIELD_TABLE = {
    "New Cairo": 7.5,
    "Maadi": 8.0,
    "Heliopolis": 7.0,
    "Nasr City": 8.5,
    "Sheikh Zayed": 7.2,
    "6th October": 8.0,
    "Rehab City": 7.8,
    "Madinaty": 7.5,
    "New Capital City": 7.0,
    "Shorouk City": 6.8,
    "Mostakbal City": 7.0,
    "Alexandria": 6.5,
    "Smouha": 7.0,
    "Downtown Cairo": 5.5,
    "Garden City": 5.0,
    "Zamalek": 5.5,
    "Mokattam": 6.0,
    "Noor City": 7.0,
    "El Nozha": 6.8,
    "Badr City": 6.5,
    "El Manial": 7.0,
    "Other": 6.5,
    "unknown": 6.5,
}


def load_model():
    """Load the trained Random Forest model."""
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_district_ppsm():
    """Load the district price-per-sqm mapping."""
    if not os.path.exists(PPSM_PATH):
        return {}
    with open(PPSM_PATH, "rb") as f:
        ppsm = pickle.load(f)
    if isinstance(ppsm, pd.Series):
        return ppsm.to_dict()
    return dict(ppsm)


def predict_price(
    area,
    rooms,
    bathrooms,
    district,
    has_elevator=0,
    has_security=0,
    has_balcony=0,
    has_pool=0,
    has_garden=0,
    has_parking=0,
):
    """
    Predict property price using the trained model.
    Returns dict with prediction details or None on error.
    """
    model = load_model()
    if model is None:
        return None

    district_ppsm = load_district_ppsm()
    ppsm_value = district_ppsm.get(district)

    if ppsm_value is None:
        all_values = list(district_ppsm.values())
        ppsm_value = float(np.median(all_values)) if all_values else 20000.0

    # Feature vector matching model training order
    features = np.array(
        [[rooms, area, bathrooms, has_elevator, has_security, has_balcony,
          has_pool, has_garden, has_parking, ppsm_value]]
    )

    try:
        predicted_price = model.predict(features)[0]
    except Exception as e:
        return {"error": str(e)}

    predicted_price = max(predicted_price, 0)

    roi = YIELD_TABLE.get(district, 6.5)
    annual_rent = predicted_price * (roi / 100)
    monthly_rent = annual_rent / 12

    return {
        "predicted_price": predicted_price,
        "price_per_sqm": predicted_price / area if area > 0 else 0,
        "district_avg_ppsm": ppsm_value,
        "estimated_roi": roi,
        "estimated_annual_rent": annual_rent,
        "estimated_monthly_rent": monthly_rent,
    }


def get_gemini_recommendation(api_key, district, area, rooms, bathrooms,
                               amenities, prediction, df):
    """
    Use Google Gemini to generate a natural-language investment recommendation
    based on the model prediction and market data context.
    Returns the AI text on success, or an error string starting with 'ERROR:' on failure.
    """
    # Build market context from the data
    district_data = df[df["district"] == district]
    known = df[~df["district"].isin(["unknown", "Other"])]

    context_lines = []
    context_lines.append(f"District: {district}")
    context_lines.append(f"Property specs: {area} sqm, {rooms} rooms, {bathrooms} bathrooms")

    amenity_list = [k for k, v in amenities.items() if v == 1]
    context_lines.append(f"Amenities: {', '.join(amenity_list) if amenity_list else 'None'}")

    context_lines.append(f"\nML Model Prediction:")
    context_lines.append(f"  Predicted price: {prediction['predicted_price']:,.0f} EGP")
    context_lines.append(f"  Price per sqm: {prediction['price_per_sqm']:,.0f} EGP")
    context_lines.append(f"  District avg price/sqm: {prediction['district_avg_ppsm']:,.0f} EGP")
    context_lines.append(f"  Estimated ROI: {prediction['estimated_roi']:.1f}%")
    context_lines.append(f"  Estimated monthly rent: {prediction['estimated_monthly_rent']:,.0f} EGP")

    if len(district_data) > 0:
        context_lines.append(f"\nDistrict Market Data ({district}):")
        context_lines.append(f"  Total listings: {len(district_data)}")
        context_lines.append(f"  Median price: {district_data['unified_price'].median():,.0f} EGP")
        context_lines.append(f"  Mean price: {district_data['unified_price'].mean():,.0f} EGP")
        context_lines.append(f"  Price std: {district_data['unified_price'].std():,.0f} EGP")
        context_lines.append(f"  Median area: {district_data['unified_area'].median():.0f} sqm")
        context_lines.append(f"  Median price/sqm: {district_data['price_per_sqm'].median():,.0f} EGP")
        context_lines.append(f"  Avg rooms: {district_data['unified_rooms'].mean():.1f}")

    # Top 5 districts for comparison
    top5 = (known.groupby("district").agg(
        count=("unified_price", "size"),
        median_price=("unified_price", "median"),
        median_ppsm=("price_per_sqm", "median"),
        avg_roi=("estimated_roi_percent", "mean"),
    ).reset_index().nlargest(5, "count"))

    context_lines.append("\nTop 5 districts by listings:")
    for _, r in top5.iterrows():
        context_lines.append(
            f"  {r['district']}: {r['count']} listings, "
            f"median {r['median_price']:,.0f} EGP, "
            f"PPSM {r['median_ppsm']:,.0f}, ROI {r['avg_roi']:.1f}%"
        )

    context = "\n".join(context_lines)

    prompt = f"""You are an expert Egyptian real estate investment advisor. Based on the 
following market data and ML model prediction, provide a detailed investment recommendation.

{context}

Please provide your analysis covering:
1. Whether this property is fairly priced, overvalued, or undervalued compared to district data
2. Investment potential and expected returns
3. Risk assessment for this district
4. Comparison with alternative districts that might offer better value
5. Final recommendation (buy/hold/pass) with reasoning

Keep your response concise (under 300 words), professional, and data-driven.
Do not use emojis. Format with clear numbered points."""

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        # Try multiple models in case of quota exhaustion
        models_to_try = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]
        last_error = None

        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text
            except Exception as model_err:
                last_error = model_err
                continue

        return f"ERROR: All models exhausted. Last error: {str(last_error)}"
    except Exception as e:
        return f"ERROR: {str(e)}"


def get_investment_recommendation(df, district, area, rooms, predicted_price):
    """
    Generate data-driven investment recommendations (fallback when no API key).
    """
    recommendations = []
    known = df[~df["district"].isin(["unknown", "Other"])]
    district_data = known[known["district"] == district]

    # 1. Compare with district median
    if len(district_data) > 0:
        median_price = district_data["unified_price"].median()
        diff_pct = ((predicted_price - median_price) / median_price) * 100

        if diff_pct > 15:
            recommendations.append(
                f"The predicted price is {diff_pct:.0f}% above the district median of "
                f"{median_price/1e6:.2f}M EGP. This suggests a premium property. "
                f"Verify that the amenities and specifications justify the higher price."
            )
        elif diff_pct < -15:
            recommendations.append(
                f"The predicted price is {abs(diff_pct):.0f}% below the district median of "
                f"{median_price/1e6:.2f}M EGP. This could indicate an undervalued opportunity."
            )
        else:
            recommendations.append(
                f"The predicted price aligns well with the district median of "
                f"{median_price/1e6:.2f}M EGP (within {abs(diff_pct):.0f}%). Fairly priced for {district}."
            )

    # 2. ROI assessment
    roi = YIELD_TABLE.get(district, 6.5)
    if roi >= 8.0:
        recommendations.append(
            f"{district} offers one of the highest estimated rental yields at {roi}%. "
            f"Attractive for buy-to-let investors."
        )
    elif roi >= 7.0:
        recommendations.append(
            f"{district} has a moderate estimated rental yield of {roi}%. "
            f"Balanced investment with reasonable returns."
        )
    else:
        recommendations.append(
            f"{district} has a lower estimated rental yield of {roi}%. "
            f"Better suited for long-term capital appreciation."
        )

    # 3. Price consistency
    if len(district_data) >= 10:
        cv = district_data["unified_price"].std() / district_data["unified_price"].mean()
        if cv > 1.0:
            recommendations.append(
                f"Prices in {district} show high variability (CV: {cv:.2f}). "
                f"Conduct thorough due diligence."
            )
        elif cv < 0.5:
            recommendations.append(
                f"Prices in {district} are consistent (CV: {cv:.2f}). Stable market."
            )

    # 4. Alternative districts
    district_roi = known.groupby("district").agg(
        median_ppsm=("price_per_sqm", "median"),
        roi=("estimated_roi_percent", "mean"),
        count=("unified_price", "size"),
    ).reset_index()
    district_roi = district_roi[district_roi["count"] >= 10]

    if district in district_roi["district"].values:
        current_ppsm = district_roi[district_roi["district"] == district]["median_ppsm"].values[0]
        cheaper = district_roi[
            (district_roi["median_ppsm"] < current_ppsm * 0.8) & (district_roi["roi"] >= roi)
        ].sort_values("roi", ascending=False)

        if len(cheaper) > 0:
            top_alt = cheaper.iloc[0]
            recommendations.append(
                f"Consider {top_alt['district']} as an alternative: "
                f"lower price/sqm ({top_alt['median_ppsm']:,.0f} vs {current_ppsm:,.0f} EGP) "
                f"with comparable ROI ({top_alt['roi']:.1f}%)."
            )

    return recommendations


def model_available():
    """Check if the model file exists."""
    return os.path.exists(MODEL_PATH)
