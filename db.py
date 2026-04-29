"""
MongoDB connection helper for the Property Analysis Dashboard.
"""

import pandas as pd
from pymongo import MongoClient
import os
from urllib.parse import quote_plus

db_password = os.getenv("MONGO_PASSWORD", "seyam")
encoded_pwd = quote_plus(db_password)
MONGO_URI = os.getenv("MONGO_URI", f"mongodb+srv://joseyam:{encoded_pwd}@cluster0.r55mnz5.mongodb.net/?appName=Cluster0")

DB_NAME = "real_estate_db"
COLLECTION_NAME = "properties"

def get_client():
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)


def get_db():
    return get_client()[DB_NAME]


def get_collection():
    return get_db()[COLLECTION_NAME]


def query_properties(filters=None, projection=None, limit=0):
    """
    Query properties from MongoDB with optional filters.
    Returns a pandas DataFrame.
    """
    collection = get_collection()
    filters = filters or {}
    cursor = collection.find(filters, projection)
    if limit > 0:
        cursor = cursor.limit(limit)
    data = list(cursor)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df


def get_all_properties():
    """Load all properties as a DataFrame."""
    return query_properties()


def get_district_list():
    """Get sorted list of unique districts."""
    collection = get_collection()
    districts = collection.distinct("district")
    districts = [d for d in districts if d and d != "unknown" and d != "Other"]
    return sorted(districts)


def get_district_stats():
    """Aggregation pipeline for district-level statistics."""
    collection = get_collection()
    pipeline = [
        {"$match": {"district": {"$nin": ["unknown", "Other"]}}},
        {
            "$group": {
                "_id": "$district",
                "count": {"$sum": 1},
                "mean_price": {"$avg": "$unified_price"},
                "median_price": {"$avg": "$unified_price"},
                "mean_ppsm": {"$avg": "$price_per_sqm"},
                "mean_area": {"$avg": "$unified_area"},
                "mean_rooms": {"$avg": "$unified_rooms"},
                "mean_roi": {"$avg": "$estimated_roi_percent"},
                "std_price": {"$stdDevPop": "$unified_price"},
            }
        },
        {"$sort": {"count": -1}},
    ]
    results = list(collection.aggregate(pipeline))
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df = df.rename(columns={"_id": "district"})
    df["cv"] = df["std_price"] / df["mean_price"]
    return df


def search_properties(
    district=None,
    min_price=None,
    max_price=None,
    min_area=None,
    max_area=None,
    min_rooms=None,
    max_rooms=None,
    min_bathrooms=None,
    max_bathrooms=None,
    amenities=None,
    limit=100,
):
    """Search properties with multiple filter criteria."""
    filters = {}

    if district and district != "All":
        filters["district"] = district

    price_filter = {}
    if min_price is not None:
        price_filter["$gte"] = min_price
    if max_price is not None:
        price_filter["$lte"] = max_price
    if price_filter:
        filters["unified_price"] = price_filter

    area_filter = {}
    if min_area is not None:
        area_filter["$gte"] = min_area
    if max_area is not None:
        area_filter["$lte"] = max_area
    if area_filter:
        filters["unified_area"] = area_filter

    rooms_filter = {}
    if min_rooms is not None:
        rooms_filter["$gte"] = min_rooms
    if max_rooms is not None:
        rooms_filter["$lte"] = max_rooms
    if rooms_filter:
        filters["unified_rooms"] = rooms_filter

    bath_filter = {}
    if min_bathrooms is not None:
        bath_filter["$gte"] = min_bathrooms
    if max_bathrooms is not None:
        bath_filter["$lte"] = max_bathrooms
    if bath_filter:
        filters["unified_bathrooms"] = bath_filter

    if amenities:
        for amenity in amenities:
            col = f"has_{amenity.lower()}"
            filters[col] = 1

    return query_properties(filters=filters, limit=limit)


def check_connection():
    """Check if MongoDB is reachable."""
    try:
        client = get_client()
        client.admin.command("ping")
        return True
    except Exception:
        return False


def get_document_count():
    """Get total number of documents in the collection."""
    try:
        return get_collection().count_documents({})
    except Exception:
        return 0
