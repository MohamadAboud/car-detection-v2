# Local imports
from config import Config


# SQLAlchemy Config
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

sqlalchemy_database_uri = Config.SQLALCHEMY_DATABASE_URI

# Create the SQLAlchemy engine
engine = create_engine(sqlalchemy_database_uri)

# Define the Base class for SQLAlchemy models
Base = declarative_base()

# Create a session factory
Session = sessionmaker(bind=engine)

def create_sqlalchemy_tables():
    """
    Create the database tables if they don't exist.
    
    Args:
        None
    
    Returns:
        None
    """
    try:
        # Create the database tables
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        pass