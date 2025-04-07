from pymongo import MongoClient

def get_db():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["heart_disease_db"]
    return db["predictions"]  # Collection name
