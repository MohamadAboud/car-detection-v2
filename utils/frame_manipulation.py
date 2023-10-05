import cv2
import base64
import numpy as np
from typing import List

def decode_frame(frame_base64: str) -> np.ndarray:
        """
        Decodes a base64-encoded frame and returns a decoded frame.
        
        Args:
            frame_base64 (str): A base64-encoded frame.

        Returns:
           numpy.ndarray: A decoded frame.

        This method decodes a base64-encoded frame into an image using OpenCV. It returns
        the decoded image as a NumPy array. If any errors occur during decoding, None is returned.

        Raises:
            TypeError: If the input frame is not a NumPy array.

        Example:
            Consider an instance of CarDetectionConsumer. Calling _decode_frames with an appropriate list
            of base64-encoded frames:

            >>> frame_base64 = 'base64_encoded_data_here'
            >>> processor.decode_frames(frame_base64)
        """
        try:
            decoded_frame = base64.b64decode(frame_base64)
            decoded_frame = np.frombuffer(decoded_frame, dtype=np.uint8)
            decoded_frame = cv2.imdecode(decoded_frame, cv2.IMREAD_COLOR)
            
            return decoded_frame
        except Exception as e:
            print("Error decoding frame:", e)
            return []

def encode_frame(frame: np.ndarray) -> str:
        """
        Encodes a frame and returns a base64-encoded frame.
        
        Args:
            frame (numpy.ndarray): A frame.

        Returns:
           str: A base64-encoded frame.

        This method encodes a frame into a base64-encoded frame using OpenCV. It returns
        the base64-encoded frame as a string. If any errors occur during encoding, None is returned.

        Raises:
            TypeError: If the input frame is not a NumPy array.

        Example:
            Consider an instance of CarDetectionConsumer. Calling _encode_frames with an appropriate list
            of frames:

            >>> frame = 'frame_data_here'
            >>> processor.encode_frames(frame)
        """
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            encoded_frame = base64.b64encode(buffer)
            
            return encoded_frame.decode('utf-8')
        except Exception as e:
            print("Error encoding frame:", e)
            return ""
