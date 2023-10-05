from typing import Any, List
from utils import ProjectPaths

class Config:
    """
    Configuration constants for Kafka and car detection.
    """

    # Kafka topic name for car detection messages
    TOPIC_NAME: str = 'car_detection'

    # Kafka bootstrap server configuration
    BOOTSTRAP_IP: str = '10.104.93.100'
    BOOTSTRAP_PORT: str = '9092'
    BOOTSTRAP_SERVERS: List[str] = [f'{BOOTSTRAP_IP}:{BOOTSTRAP_PORT}']

    # Postgres configuration
    DB_CONFIG = {
        'db_user': 'postgres',  # Database username
        'db_password': '1234',   # Database password
        'db_host': 'postgres',  #'host.docker.internal',  # Database host (change if hosted elsewhere)
        'db_name': 'AIAP_DB',  # Database name
        'db_port': 5432  # PostgreSQL default port
        }
    db_username = DB_CONFIG['db_user']
    db_password = DB_CONFIG['db_password']
    db_host = DB_CONFIG['db_host']
    db_port = DB_CONFIG['db_port']
    db_name = DB_CONFIG['db_name']

    SQLALCHEMY_DATABASE_URI: str = f'postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False