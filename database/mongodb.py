"""
MongoDB database connection and collection helper functions for LinkLens AI.
"""

from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, PyMongoError

# Module-level database and client references
client: Optional[MongoClient] = None
db: Optional[Database] = None


def init_db(app) -> Optional[Database]:
    """
    Connect to MongoDB using app configuration, initialize database, and create required indexes.

    Args:
        app: Flask application instance containing configuration.

    Returns:
        Database instance or None if connection fails.
    """
    global client, db

    mongo_uri = app.config.get("MONGO_URI", "mongodb://localhost:27017/")
    db_name = app.config.get("DATABASE_NAME", "linklens_ai")

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # Test connection with ping command
        client.admin.command("ping")
        db = client[db_name]

        # Create indexes
        # users: unique index on 'email'
        db["users"].create_index("email", unique=True)

        # scan_history: index on 'user_id' and 'scanned_at'
        db["scan_history"].create_index("user_id")
        db["scan_history"].create_index("scanned_at")
        db["scan_history"].create_index([("user_id", 1), ("scanned_at", -1)])

        print(f"[LinkLens AI] Successfully connected to MongoDB database '{db_name}'.")
        return db
    except (ConnectionFailure, PyMongoError) as err:
        print(f"[LinkLens AI] Warning: Failed to connect to MongoDB at {mongo_uri}: {err}")
        return None
    except Exception as exc:
        print(f"[LinkLens AI] Unexpected error during MongoDB initialization: {exc}")
        return None


def get_db() -> Optional[Database]:
    """
    Return the active MongoDB database instance.

    Returns:
        pymongo.database.Database or None.
    """
    return db


def get_users_collection() -> Optional[Collection]:
    """
    Return the 'users' collection.

    Returns:
        pymongo.collection.Collection or None if db is not connected.
    """
    if db is not None:
        return db["users"]
    return None


def get_scans_collection() -> Optional[Collection]:
    """
    Return the 'scan_history' collection.

    Returns:
        pymongo.collection.Collection or None if db is not connected.
    """
    if db is not None:
        return db["scan_history"]
    return None
