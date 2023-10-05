import os

class ProjectPaths:
    # Get the current directory (the directory where this script is located)
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Define project folder paths
    project_root = os.path.abspath(os.path.join(current_directory, os.pardir))
    
    # Define instance folder paths relative to the project root
    # database_dir = os.path.join(project_root, "instance")
    
    # Define subfolder paths relative to the upload folder.
    storage_dir = os.path.join(project_root, "storage")
    images_dir = os.path.join(storage_dir, "images")
    # Define temporary folder paths relative to the project root
    temp_dir = os.path.join(storage_dir, "temp")
    
    @classmethod
    def init(cls):
       """
        Create the project folders if they don't exist.
       """
       # os.makedirs(cls.database_dir, exist_ok=True)
       os.makedirs(cls.storage_dir, exist_ok=True)
        
       os.makedirs(cls.images_dir, exist_ok=True)
       os.makedirs(cls.temp_dir, exist_ok=True)
       

# Initialize the project paths
ProjectPaths.init()