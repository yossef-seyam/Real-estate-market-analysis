"""
Script to load the cleaned property data into MongoDB.
Run this once before starting the Streamlit app.

Usage:
    python load_data_to_mongo.py
"""

import os
import pandas as pd
from pymongo import MongoClient, ASCENDING

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "property_analysis"
COLLECTION_NAME = "properties"

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Cleaning Data", "final_unified_property_data.csv"
)


def load_data():
    print("=" * 60)
    print("PROPERTY DATA LOADER")
    print("=" * 60)

    # Check file exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        return False

    # Read CSV
    print(f"\nReading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Filter to Property Finder data only
    df = df[df["source"] == "property_finder"].reset_index(drop=True)
    print(f"Filtered to Property Finder only: {len(df)} rows")

    # Connect to MongoDB
    print(f"\nConnecting to MongoDB at {MONGO_URI}")
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command("ping")
        print("MongoDB connection successful")
    except Exception as e:
        print(f"ERROR: Could not connect to MongoDB: {e}")
        print("Make sure MongoDB is running on localhost:27017")
        return False

    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Drop existing collection
    print(f"\nDropping existing collection '{COLLECTION_NAME}'...")
    collection.drop()

    # Convert DataFrame to list of dicts
    records = df.to_dict("records")

    # Insert data
    print(f"Inserting {len(records)} documents...")
    collection.insert_many(records)
    print(f"Inserted {collection.count_documents({})} documents")

    # Create indexes for fast querying
    print("\nCreating indexes...")
    collection.create_index([("district", ASCENDING)])
    collection.create_index([("unified_price", ASCENDING)])
    collection.create_index([("unified_area", ASCENDING)])
    collection.create_index([("unified_rooms", ASCENDING)])
    collection.create_index([("source", ASCENDING)])
    print("Indexes created")

    print("\n" + "=" * 60)
    print("DATA LOADING COMPLETE")
    print("=" * 60)

    # Print summary
    print(f"\nDatabase: {DB_NAME}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents: {collection.count_documents({})}")
    print(f"Districts: {len(collection.distinct('district'))}")
    print(f"Sources: {collection.distinct('source')}")

    client.close()
    return True


if __name__ == "__main__":
    load_data()
