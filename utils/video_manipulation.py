import os
import cv2
from typing import List
from tqdm import tqdm

def extract_frames_from_video(video_path: str, show_progress: bool = False) -> List:
    """
    Extract frames from a video file and return them as a list of NumPy arrays.

    Args:
        video_path (str): The path to the video file.
        show_progress (bool, optional): Whether to display a progress bar (default is False).

    Returns:
        List: A list of frames as NumPy arrays.

    Raises:
        FileNotFoundError: If the video file does not exist.
    
    Example:
        >>> frames = extract_frames_from_video("video.mp4", True)
        >>> print(len(frames))
        100
    """
    try:
        # Check if the video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create an empty list to store the frames
        frames = []

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video file was opened successfully
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {video_path}")

        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Use tqdm to show progress
        if show_progress:
            progress_bar = tqdm(total=total_frames, unit="frame")

        # Loop over the frames of the video file
        while True:
            # Read the current frame
            ret, frame = cap.read()
            # Check if there is a frame
            if not ret:
                break
            # Add the frame to the list of frames
            frames.append(frame)

            # Update progress bar
            if show_progress:
                progress_bar.update(1)

        # Release the video file
        cap.release()

        # Close the progress bar
        if show_progress:
            progress_bar.close()

        return frames
    except Exception as e:
        raise e

def create_video_from_images(image_folder: str, output_video_path: str = "output.mp4", frame_rate: float = 30, images_extension: str = ".jpg", show_progress: bool = False) -> str:
    """
    Create a video from a folder of images.

    Args:
        image_folder (str): The path to the folder containing the images.
        output_video_path (str, optional): The path to save the output video (default is "output.mp4").
        frame_rate (float, optional): The frame rate of the output video (default is 30).
        images_extension (str, optional): The extension of the images in the folder (default is ".jpg").
        show_progress (bool, optional): Whether to display a progress bar (default is False).

    Returns:
        str: The path to the output video.
    
    Raises:
        ValueError: If no image files are found in the folder.
    
    Example:
        >>> create_video_from_images("images", "video.mp4", 30, ".jpg", True)
        Video saved to `video.mp4`
        'video.mp4'
    """
    try:
        # Get a list of image file names in the folder
        image_files = [f for f in os.listdir(image_folder) if f.endswith(images_extension)]
        
        try:
            # Sort the image file names
            image_files = sorted(image_files, key=lambda x: int(x.split(".")[0]))
        except:
            pass

        if not image_files:
            raise ValueError(f"No image files found in {image_folder}")

        # Get the dimensions of the first image (assuming all images have the same dimensions)
        first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
        height, width, layers = first_image.shape
        frame_size = (width, height)

        # Create a VideoWriter object to save the video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)

        # Use tqdm to show progress
        if show_progress:
            progress_bar = tqdm(total=len(image_files), unit="frame")

        for image_file in sorted(image_files):
            image_path = os.path.join(image_folder, image_file)
            frame = cv2.imread(image_path)

            if frame is not None:
                out.write(frame)

            # Update progress bar
            if show_progress:
                progress_bar.update(1)

        out.release()

        # Close the progress bar
        if show_progress:
            progress_bar.close()

        print(f"Video saved to {output_video_path}")
        return output_video_path
    except Exception as e:
        raise e
