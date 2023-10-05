from datetime import datetime # Import the Datetime to get the current time.
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import relationship # Import the RelationShip to link the tables.
from sqlalchemy import Column, Integer, String, DateTime, JSON # Import the SqlAlchemy Data types.

# database
from database import Base

class OutputModel(Base):
    """
    OutputModel Class represents a output entity in the database.
    
    Attributes:
        Output_ID (int): The output id.
        UseCase_ID (int): The use case id.
        Source_ID (int): The source id.
        Output_Data (str): The output data.
        Img_Path (str): The image path.
        Created_by (int): The created by.
        Creation_Date (datetime): The creation date.
    
    Methods:
        to_map(): Convert the object to a dictionary.
    
    Usage:
        output = OutputModel(
            UseCase_ID = 1,
            Source_ID = 1,
            Output_Data = {
                "id": uuid.uuid4().hex,
                "name": "John Doe"
            },
            Img_Path = "C:/Users/JohnDoe/Desktop/JohnDoe.png",
            Created_by = 0,
            Creation_Date = datetime.now()
        )
    
    """
    
    
    __tablename__ = "DF_AIAP_OUTPUT"
    
    Output_ID             = Column(Integer, primary_key=True, autoincrement=True) # Auto Incremental ID
    
    UseCase_ID            = Column(Integer, nullable=False) # Data Column
    Source_ID             = Column(Integer, nullable=False) # Data Column
    Output_Data           = Column(JSON, nullable=False) # Data Column
    Img_Path              = Column(String, nullable=False) # Data Column
    Created_by            = Column(Integer, nullable=False) # Data Column
    Creation_Date         = Column(DateTime, nullable=False, default=datetime.now) # Created At Column
    
    
    def to_map(self) -> Dict[str, Any]:
        """
        to_map() method is used to convert the object to a dictionary.
        
        Args:
            None
        
        Returns:
            Dict[str, Any]: The object as a dictionary.
        """
        return {
            "Output_ID": self.Output_ID,
            "UseCase_ID": self.UseCase_ID,
            "Source_ID": self.Source_ID,
            "Output_Data": self.Output_Data,
            "Img_Path": self.Img_Path,
            "Created_by": self.Created_by,
            "Creation_Date": self.Creation_Date
        }