from typing import Any
import yaml
import os
from easydict import EasyDict

class YamlParser(EasyDict):
    """
    A class for parsing YAML configuration files.
    
    This class inherits from the EasyDict class, which is a dictionary with attribute-style access.
     
    Attributes:
        None
    
    Methods:
        __init__: Initializes the YamlParser class.
    
    Examples:
        >>> yaml_parser = YamlParser('config.yaml')
        >>> print(yaml_parser)
        {'model': {'name': 'resnet', 'num_classes': 10}, 'optimizer': {'name': 'adam', 'lr': 0.001}}
        >>> print(yaml_parser.model)
        {'name': 'resnet', 'num_classes': 10}
        >>> print(yaml_parser.model.name)
        resnet
        >>> print(yaml_parser.optimizer)
        {'name': 'adam', 'lr': 0.001}
        >>> print(yaml_parser.optimizer.name)
        adam
        >>> print(yaml_parser.optimizer.lr)
        0.001
    """
    def __init__(self, config_file: str) -> None:
        """
        Initializes the YamlParser class.
        
        Args:
            config_file (str): Path to the YAML configuration file.
        
        Returns:
            None
        """
        yaml_dict = {}
        
        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as f:
                yaml_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
            
        super(YamlParser, self).__init__(yaml_dict)
    
    def __repr__(self) -> str:
        """
        Returns the string representation of the YamlParser class.
        
        Args:
            None
        
        Returns:
            str: String representation of the YamlParser class.
        """
        return str(self.__dict__)
    
    def __call__(self) -> Any:
        """
        Returns the dictionary representation of the YamlParser class.
        
        Args:
            None
        
        Returns:
            dict: Dictionary representation of the YamlParser class.
        """
        return self.__dict__
    
    def get_value(self, key: str) -> Any:
        """
        Returns the value of the key.
        
        Args:
            key (str): Key.
        
        Returns:
            Any: Value of the key.
        """
        return self.__dict__[key]
