import random
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

import os
import sys
sys.path.append(os.getcwd())


class CarDetection:
    """
    A class for real-time car detection using the YOLOv8n model.

    Attributes:
        model (YOLO): The YOLOv8n model for car detection.
        class_list (List[str]): List of class names for detection.
        detection_colors (List[Tuple[int, int, int]]): Random colors for each class.

    Methods:
        __init__(self, model_path: Optional[str] = None, epochs: Epochs = Epochs.MEDIUM) -> None:
            Initializes the CarDetection object.

        _generate_random_colors(self) -> List[Tuple[int, int, int]]:
            Generates random colors for the class list.

        _process_frame(self, frame) -> np.ndarray:
            Processes a single frame for car detection.

        run(self, frame) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
            Detects objects in a single frame and returns object information.
    """

    model = None

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the CarDetection object.

        Args:
            model_path (str, optional): Path to the YOLO model weights. If not provided, a default path is used.

        Raises:
            ValueError: If invalid input parameters are provided.
        """

        self.model_path = model_path if model_path else f"./models/car_detection_50_epochs.pt"
        
        # check if model file is exists
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please provide a valid model path.")
        

        try:
            self.create_yolo_model()
            self.class_list = self.model.names
            self.detection_colors = self._generate_random_colors()
        except Exception as e:
            raise Exception(f"Error initializing CarDetection object: {e}")
    
    def create_yolo_model(self) -> YOLO:
        """
        Create YOLO model for car detection.
        
        Returns:
            YOLO: The YOLOv8n model for car detection.
        """
        try:
            # check if model already exists
            if self.model is None:
                # create model
                self.model = YOLO(self.model_path, task='detect')
            
            return self.model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def _generate_random_colors(self) -> List[Tuple[int, int, int]]:
        """Generate random colors for class list."""
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                  for _ in range(len(self.class_list))]
        return colors

    def _draw_bounding_box(self, frame: np.ndarray, bb: Any, clsID: Any, conf:Any):
        """Draw bounding box and label on the frame."""
        cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                      self.detection_colors[int(clsID)], 3)
        label = f"{self.class_list[int(clsID)]} {str(round(conf, 3))}%"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame, label, (int(bb[0]), int(bb[1]) - 10), font, 1, (255, 255, 255), 2)

    def _process_frame(self, frame: np.ndarray, detect_params: Any) -> np.ndarray:
        """
        Process a single frame for car detection.

        Args:
            frame (numpy.ndarray): The input frame to detect objects in.
            detect_params: The detection parameters for a single frame.

        Returns:
            numpy.ndarray: Processed frame with detection bounding boxes and labels.
        """
        for i, box in enumerate(detect_params.boxes.cpu().numpy()):
            clsID = box.cls[0]
            conf = box.conf[0]
            bb = box.xyxy[0]
            self._draw_bounding_box(frame, bb, clsID, conf)

        return frame

    
    def run_batch(self, frames: List[np.ndarray]) -> Tuple[List[np.ndarray], List[List[Dict[str, Any]]]]:
        """
        Detect objects in a single frame and return object information.

        Args:
            frame (numpy.ndarray): The input frame to detect objects in.

        Returns:
            Tuple containing:
                - processed_frame (numpy.ndarray): Processed frame with detection bounding boxes and labels.
                - detected_objects (List[Dict[str, Any]]): List of dictionaries containing object information (name, confidence, coordinates).
        
        Raises:
            Exception: If an error occurs while running car detection.
        """
        try:
            results = self.model.predict(source=frames, conf=0.3, save=False)
            processed_frames = []
            detected_objects_list = []
            for r in results:
                annotated_frame  = r.plot()
                # annotated_frame  = self._process_frame(r.orig_img, r)
                objects = []

                for box in r.boxes.cpu().numpy():
                    class_id = box.cls[0]
                    conf = box.conf[0]
                    # Get the bounding box
                    x, y, w, h = box.xywh[0]
                    
                    conf = conf * 100

                    object_info = {
                        'confidence': conf,
                        'boxes': box,
                        'coordinates': {
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h,
                        }
                    }
                    objects.append(object_info)
                
                processed_frames.append(annotated_frame )
                detected_objects_list.append(objects)

            return processed_frames, detected_objects_list
        except Exception as e:
            raise Exception(f"Error running car detection: {e}")
