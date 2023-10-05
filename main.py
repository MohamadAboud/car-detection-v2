print("[!].This is the main.py file content `4` step.\n\n")

print("[1]. Importing the necessary libraries...")
import os

from app import CarDetectionConsumer
from config import Config
from database import create_sqlalchemy_tables


def main():
    try:
        print("[2]. Checking the environment variables...")
        # Set CUDA_LAUNCH_BLOCKING environment variable to 1
        os.environ['CUDA_LAUNCH_BLOCKING'] = '50'
        # torch.use_deterministic_algorithms(True)
        # Get the kafka topic name from the environment variable if exist.
        kafka_topic: str = os.environ.get('KAFKA_TOPIC', Config.TOPIC_NAME)
        # Set the kafka topic name in the config.
        Config.TOPIC_NAME = kafka_topic
        # Get the show_frame flag from the environment variable if exist.
        show_frame: bool = os.environ.get('SHOW_FRAME', False)
        
        # Print the configuration.
        print(f'KAFKA_SERVERS: {Config.BOOTSTRAP_SERVERS}')
        print(f'KAFKA_TOPIC: {kafka_topic}')
        print(f'SHOW_FRAME: {show_frame}')
        print('-'*25)
        
        print('[3].Creating the database tables if not exist...')
        create_sqlalchemy_tables()
        
        print("[4]. Creating a new instance of the CarDetectionProcessor class...\n\n")
        # Create a new instance of the CarDetectionProcessor class.
        processor = CarDetectionConsumer(show_frame=show_frame)
        # Start processing messages.
        processor.process_messages()
    except KeyboardInterrupt:
        print('Stopping the car detection processor...')
    except Exception as e:
        print(f'An error occurred: {e}')

if __name__ == '__main__':
    main()