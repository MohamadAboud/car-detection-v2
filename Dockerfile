# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install torch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the folders
COPY . /app

# Run the command to start your application
CMD ["python", "main.py"]
