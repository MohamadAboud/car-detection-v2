# Real-Time Car Detection using YOLOv8n



## Introduction

Welcome to the Real-Time Car Detection using YOLOv8n project! This code demonstrates how to perform car detection in real-time using the YOLOv8n model. The YOLOv8n model is utilized for object detection, specifically targeting car detection. This readme provides an overview of the code, how to use it, installation instructions, and more.

## Description

This Python script utilizes the YOLOv8n model to perform real-time car detection on a video stream or webcam feed. It draws bounding boxes around detected cars and displays the results in real-time. The script supports customization of the model's epochs and video source.

## Prerequisites

Before using this code, ensure you have the following dependencies installed:

- Python (>=3.6)
- OpenCV (cv2)
- NumPy
- Ultralytics
- Torch (2.0.1+cu117)

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## How to Use

1. Clone this repository or download the script.

2. Open a terminal and navigate to the directory containing the script.

3. Run the script using the following command:

```bash
python main.py
```
This command runs the main.py script with its default parameters. The script will perform some predefined actions or processes according to its logic.

4. Kafka Producer Test
Additionally, there is another command mentioned:

```bash
python .\app\kafka\kafka_producer.py
```

This command seems to be for testing a Kafka producer component of the application, located in the `kafka_producer.py` file within the `app/kafka` directory. This producer may be responsible for sending messages to a Kafka topic.

5. The script will start capturing the video feed from the default source (webcam) and perform real-time car detection. Detected cars will be highlighted with bounding boxes, and the class name and confidence level will be displayed above each box.

6. To exit the real-time display, press the 'q' key.


## Customization

You can customize the behavior of the script by modifying the following parameters in the script:

- `model_path`: Path to the YOLO model weights (default: "models/car_detection_{epochs}_epochs.pt").
- `source`: Video source identifier (default: 0 for webcam). You can also provide a path to a video file (e.g., "./videos/video1.avi").
- `epochs`: Number of epochs the YOLO model is trained for (default: 50).

## Acknowledgments

This code is based on the YOLOv8n model and the Ultralytics library. Special thanks to the developers of YOLOv8n and Ultralytics for their contributions to the field of object detection.

