import json
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import os
import sys
sys.path.append(os.getcwd())

# Local imports
from utils import encode_frame, decode_frame
from database import Session, create_sqlalchemy_tables
from database.sqlalchemy_db.tables.df_output.output_model import OutputModel

class ValidationException(Exception):
    """Raised when a validation error occurs"""
    
    def __init__(self, message="A validation error occurred"):
        self.message = message
        super().__init__(self.message)

class DfOutputController:
    
    def __init__(self) -> None:
        """
        Initialize the DFTempTable class.
        
        Args:
            None
        
        Returns:
            None
        """
        pass
    
    def create(self, use_case_id: int, source_id: int, output_data: Dict[str, any], img_path: str, created_by: int) -> OutputModel:
        """
        create() method is used to create a output.
        
        Args:
            use_case_id (int): The use case id.
            source_id (int): The source id.
            output_data (Dict[str, any]): The output data.
            img_path (str): The image path.
            created_by (int): The created by.
        
        Returns:
            OutputModel: The created output.
        """
        try:
            if use_case_id is None or not isinstance(use_case_id, int):
                raise ValidationException("The `use case id` is invalid.")
        
            if source_id is None or not isinstance(source_id, int):
                raise ValidationException("The `source id` is invalid.")
            
            if output_data is None or not isinstance(output_data, Dict):
                raise ValidationException(f"The `output data` is invalid ({type(output_data)}).")
            
            if img_path is None or not isinstance(img_path, str):
                raise ValidationException("The `image path` is invalid.")
            
            if created_by is None or not isinstance(created_by, int):
                raise ValidationException("The `created by` is invalid.")
            
            # Open a session
            session = Session()
            
            output = OutputModel(
                UseCase_ID = use_case_id,
                Source_ID = source_id,
                Output_Data = json.dumps(output_data),
                Img_Path = img_path,
                Created_by = created_by,
                Creation_Date = datetime.now()
            )
            
            session.add(output)
            session.commit()
            
            session.close()

            return output
        except Exception as e:
            print(f"Error creating output: {e}")
            raise ValidationException("Error creating output.") from e
