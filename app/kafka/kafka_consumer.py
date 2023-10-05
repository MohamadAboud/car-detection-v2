from concurrent.futures import ThreadPoolExecutor
import json
import os
import queue
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import base64
import numpy as np
from kafka import KafkaConsumer
import torch
import torch.multiprocessing as mp
import threading


# Local imports
from app.ai.car_detection import CarDetection
from utils import json_deserializer
from config import Config


class CarDetectionConsumer:
    """
    A class for processing car detection messages from Kafka using multiprocessing and GPU acceleration.

    This class is designed to facilitate the efficient processing of car detection messages obtained from Kafka.
    It leverages multiprocessing to handle multiple messages concurrently and can take advantage of GPU acceleration
    for enhanced AI model inference speed.

    The class provides methods for initializing the processor, processing incoming messages, and displaying or
    storing the results. It encapsulates functionality for decoding frames, running car detection inference,
    displaying processed frames, and printing device information.

    Attributes:
        topic_name (str): The name of the Kafka topic being consumed.
        bootstrap_servers (str): The Kafka broker addresses being used.
        show_frame (bool): Indicates whether processed frames are displayed.

    Notes:
        - This class requires the `torch`, `torch.cuda`, and `cv2` (OpenCV) libraries for proper functionality.
        - The Kafka consumer instance and AI model for car detection must be provided externally.
        - GPU acceleration is leveraged if a compatible CUDA-enabled GPU is available.

    Example:
        >>> processor = CarDetectionConsumer()
        >>> processor.process_messages()
    """
    
    def __init__(self, bootstrap_servers: List[str] = None, topic_name: str =None, show_frame: bool =False) -> None:
        """
        Initializes the CarDetectionConsumer class.

        Args:
            bootstrap_servers (str, optional): Comma-separated list of Kafka broker addresses.
                If not provided, the default value from Config.BOOTSTRAP_SERVERS will be used.
            topic_name (str, optional): The name of the Kafka topic to consume messages from.
                If not provided, the default value from Config.TOPIC_NAME will be used.
            show_frame (bool, optional): Whether to display the processed frames. Default is False.

        Attributes:
            topic_name (str): The name of the Kafka topic being consumed.
            bootstrap_servers (str): The Kafka broker addresses being used.
            show_frame (bool): Indicates whether processed frames are displayed.
            consumer (KafkaConsumer): Kafka consumer instance for consuming messages.
            car_detection (CarDetection): Instance of the CarDetection AI model.
        """
        # Initialize attributes
        self.topic_name:str = Config.TOPIC_NAME if topic_name is None else topic_name
        self.bootstrap_servers = Config.BOOTSTRAP_SERVERS if bootstrap_servers is None else bootstrap_servers
        self.show_frame: bool = show_frame
        # Set the batch size for processing messages
        self.batch_size: int = 30
        # Set the stop event to stop processing
        self.stop_event = threading.Event()
        # Set the number of threads to use for processing messages
        self.num_threads: int = min(mp.cpu_count(), os.environ.get('NUM_THREADS', 4) - 1)
        # Initialize a list to store threads
        self.threads: List[threading.Thread] = []
        # Initialize a queue to store messages
        self.message_queue = queue.Queue()
        # Initialize a thread pool for processing messages
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
        
        # Validate input parameters
        if not self.topic_name or not self.bootstrap_servers:
            raise ValueError("Invalid input parameters. Please provide a valid topic name and Kafka broker addresses.")

        try:
            # Initialize KafkaConsumer with provided configurations
            self.consumer: KafkaConsumer = self.create_kafka_consumer()
        except Exception as e:
            raise RuntimeError("Error initializing KafkaConsumer") from e
    
    def create_kafka_consumer(self) -> KafkaConsumer:
        """
        Creates a Kafka consumer instance.
        
        Returns:
            KafkaConsumer: A Kafka consumer instance.
        """
        return KafkaConsumer(
            self.topic_name,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=json_deserializer,
        )
    
    
    def print_device_info(self):
        """
        Prints information about the computing device and available cores/threads.

        This function detects the availability of a CUDA-enabled GPU and displays relevant
        information about the device type and, if applicable, CUDA version. It also determines
        the number of CPU cores/threads available and prints that information.

        Note:
            This function requires the `torch` and `torch.cuda` modules to be available.

        Example output:
            Device Information:
            -------------------
            Device Type: GPU (NVIDIA GeForce RTX 3080)
            CUDA Version: 11.1
            Number of Cores/Threads: 16

        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("Device Information:")
        print("-------------------")

        if device.type == 'cuda':
            print(f"Device Type: GPU ({torch.cuda.get_device_name(0)})")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("Device Type: CPU")

        num_cores = mp.cpu_count()
        print("Number of Cores/Threads:", num_cores)
        print("You are using {} Core(s)/Thread(s).".format(self.num_threads))
        print('\n\n')


    def _decode_frames(self, frame_base64: List[str]) -> np.ndarray:
        """
        Decodes a base64-encoded frames and returns a list of decoded frames.
        
        Args:
            frame_base64 (list of str): A list of base64-encoded frames.

        Returns:
            list of numpy.ndarray: A list of decoded frames.

        This method decodes a base64-encoded frame into an image using OpenCV. It returns
        the decoded image as a NumPy array. If any errors occur during decoding, None is returned.

        Raises:
            TypeError: If the input frame is not a NumPy array.

        Example:
            Consider an instance of CarDetectionConsumer. Calling _decode_frames with an appropriate list
            of base64-encoded frames:

            >>> frame_base64 = [
            ...     'base64_encoded_data_here',
            ...     # Add more frames here...
            ... ]
            >>> processor._decode_frames(frame_base64)
        """
        try:
            frames = []
            for frame in frame_base64:
                decoded_frame = base64.b64decode(frame)
                decoded_frame = np.frombuffer(decoded_frame, dtype=np.uint8)
                decoded_frame = cv2.imdecode(decoded_frame, cv2.IMREAD_COLOR)
                frames.append(decoded_frame)
            
            return frames
        except Exception as e:
            print("Error decoding frame:", e)
            return []


    def _process_frame(self, frames: List[ np.ndarray]) -> Optional[Tuple[np.ndarray, Optional[List[Dict[str, Any]]]]]:
        """
        Processes a frame using the car detection model.

        Args:
            frame (numpy.ndarray): Input frame as a NumPy array.

        Returns:
            Tuple containing:
                - processed_frame (numpy.ndarray): Processed frame with detection bounding boxes and labels.
                - detected_objects (List[Dict[str, Any]]): List of dictionaries containing object information (name, confidence, coordinates).
                

        This method takes an input frame as a NumPy array and processes it using the car detection model.
        It returns a tuple containing the processed frame, which includes bounding boxes and labels, and a list
        of detected objects, where each object is represented as a dictionary with keys 'confidence' and 'coordinates'.
        The 'coordinates' key further contains 'x_min', 'y_min', 'x_max', and 'y_max' values representing
        the bounding box coordinates.

        Raises:
            TypeError: If the input frame is not a NumPy array.

        """
        try:
            # Run car detection inference
            car_detection: CarDetection = CarDetection()
            processed_frames, detected_objects = car_detection.run_batch(frames)

            return processed_frames, detected_objects
        except Exception as e:
            print("Error processing frame:", e)
            return None, None


    def _show_frame(self, frame: np.ndarray) -> None:
        """
        Displays a frame using OpenCV.

        Args:
            frame (numpy.ndarray): The frame to be displayed.
        
        This function uses the OpenCV library to display a frame. The frame is resized to a
        standard size (640x360) for consistent display dimensions. An exception is caught
        and printed if any errors occur during the display process.

        """
        try:
            cv2.imshow('processed_frame', cv2.resize(frame, (640, 360)))
            cv2.waitKey(1)
        except Exception as e:
            print("Error displaying frame:", e)


    def _process_batch(self, msg_values: List[Dict[str, Any]]) -> None:
        """
        Process a batch of messages containing frame data.

        Args:
            msg_values (list of dict): A list of dictionaries, each containing the following keys:
                - 'id_source' (str): Identifier for the source of the frame.
                - 'frame_number' (int): Number associated with the current frame.
                - 'frame_data' (str): Base64-encoded frame data for one frame.

        Notes:
            The function processes each frame in the input batch, decodes the frames from
            base64 format, processes each frame using the `_process_frame` method, and prints or
            displays relevant information. If enabled, the processed frames are displayed using the
            `_show_frame` method.

        Example:
            Consider an instance of CarDetectionConsumer. Calling _process_batch with an appropriate list
            of message dictionaries and a specified device:

            >>> msg_values = [
            ...     {
            ...         'id_source': 'camera_1',
            ...         'frame_number': 42,
            ...         'frame_data': 'base64_encoded_data_here'
            ...     },
            ...     # Add more frames here...
            ... ]
            >>> processor._process_batch(msg_values)
        """
        print(f"Message received at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            msg_values = [json.loads(msg) for msg in msg_values]
            frames_base64 = [msg['frame'] for msg in msg_values]
            frames = self._decode_frames(frames_base64)
            
            if frames is not None and len(frames) > 0:
                processed_frame, detected_objects = self._process_frame(frames)
                
                if processed_frame is not None and detected_objects is not None:
                    for i in range(len(msg_values)):
                        message = msg_values[i]
                        annotated_frame = processed_frame[i]
                        objects = detected_objects[i]
                        
                        data = {
                            'id_source': message['id_source'],
                            'detected_objects': objects,
                            'timestamp': message['timestamp'],
                        }

                        print(data)

                        if self.show_frame:
                            self._show_frame(annotated_frame)

                        # Optionally release the Kafka message once it's processed
                        # consumer.commit()
                else:
                    print("No processed_frame.")
        except Exception as e:
            print("Error processing message:", e)

    
    def message_consumer(self) -> None:
        """
        A message consumer that collects messages into batches and puts them in the message queue.
        
        Args:
            None
        
        Returns:
            None
            
        Raises:
            If any errors occur during message consumption, the function prints the error and exits.
        """
        try:
            # Initialize a list to store frames
            frames = []
            # Start consuming messages
            while not self.stop_event.is_set():
                # Get the next message
                msg = next(self.consumer)
                # Add the message to the list of frames if it is not None
                if msg is not None and msg.value is not None:
                    # Add the message to the list of frames
                    frames.append(msg.value)
                    # Check if the batch size has been reached
                    if len(frames) >= self.batch_size:
                        # Put the batch of frames in the message queue
                        self.message_queue.put(frames)
                        # Clear the list of frames
                        frames = []
        except Exception as e:
            print("Error consuming message:", e)
            # Set the stop event to stop processing
            self.stop_event.set()
            # Exit the program
            exit(1)

    def process_messages(self) -> None:
        """
        Starts message processing.

        This method starts a message consumer thread and processes messages using a thread pool.
        
        Args:
            None
        
        Returns:
            None
        """
        
        # Print device information
        self.print_device_info()

        # Create a thread for the message consumer
        consumer_thread = threading.Thread(target=self.message_consumer)
        # Start the consumer thread
        consumer_thread.start()

        try:
            # Start processing messages
            while not self.stop_event.is_set():
                # Check if there are fewer than self.num_threads threads running
                if self.executor._work_queue.qsize() < self.num_threads:
                    msg_values = self.message_queue.get()  # Get a message from the queue
                    self.executor.submit(self._process_batch, msg_values)

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping processing.")
            # Set the stop event to stop processing
            self.stop_event.set()

        # Wait for the consumer thread to finish
        self.executor.shutdown()
        print("All threads stopped")
        # Exit the program
        exit(0)
