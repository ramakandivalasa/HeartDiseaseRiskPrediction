from pymongo import MongoClient

def get_db():
    client = MongoClient("mongodb+srv://testuser:testpass123@cluster-1.n8r9dct.mongodb.net/")
    db = client["heart_disease_db"]
    return db["predictions"]  # Collection name
