from typing import Any, List


class Config:
    """
    Configuration constants for Kafka and car detection.
    """

    # Kafka topic name for car detection messages
    TOPIC_NAME: str = 'car_detection_topic'

    # Kafka bootstrap server configuration
    BOOTSTRAP_IP: str = '10.104.93.100'
    BOOTSTRAP_PORT: str = '9092'
    BOOTSTRAP_SERVERS: List[str] = [f'{BOOTSTRAP_IP}:{BOOTSTRAP_PORT}']
