# core/utils/config_loader.py

import os
import yaml
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Utility class for loading and managing configuration files.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the config loader.
        
        Args:
            config_path: Path to the configuration file or directory
        """
        self.config_path = config_path
        
        # Check if path is a directory or file
        if os.path.isdir(config_path):
            self.config_dir = config_path
            self.config_file = None
        else:
            self.config_dir = os.path.dirname(config_path)
            self.config_file = os.path.basename(config_path)
    
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        if self.config_file:
            # Load specific config file
            config_path = os.path.join(self.config_dir, self.config_file)
            return self._load_yaml(config_path)
        else:
            # Load composite config from directory
            return self._load_composite_config()
    
    def load_project_config(self) -> Dict[str, Any]:
        """
        Load project configuration.
        
        Returns:
            Project configuration dictionary
        """
        config_path = os.path.join(self.config_dir, "project_config.yaml")
        if os.path.exists(config_path):
            return self._load_yaml(config_path)
        else:
            logger.warning(f"Project config not found at {config_path}")
            return {}
    
    def load_feature_config(self) -> Dict[str, Any]:
        """
        Load feature configuration.
        
        Returns:
            Feature configuration dictionary
        """
        config_path = os.path.join(self.config_dir, "feature_config.yaml")
        if os.path.exists(config_path):
            return self._load_yaml(config_path)
        else:
            logger.warning(f"Feature config not found at {config_path}")
            return {}
    
    def load_model_config(self) -> Dict[str, Any]:
        """
        Load model configuration.
        
        Returns:
            Model configuration dictionary
        """
        config_path = os.path.join(self.config_dir, "model_config.yaml")
        if os.path.exists(config_path):
            return self._load_yaml(config_path)
        else:
            logger.warning(f"Model config not found at {config_path}")
            return {}
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """
        Load YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Dictionary from YAML
        """
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            
            logger.debug(f"Loaded config from {path}")
            return data or {}
        except Exception as e:
            logger.error(f"Error loading config from {path}: {str(e)}")
            return {}
    
    def _load_composite_config(self) -> Dict[str, Any]:
        """
        Load composite configuration from directory.
        
        Returns:
            Composite configuration dictionary
        """
        config = {}
        
        # Priority order for loading configs
        config_files = [
            "project_config.yaml",
            "feature_config.yaml",
            "model_config.yaml",
            "pipeline_config.yaml"
        ]
        
        # Load each config file
        for file in config_files:
            path = os.path.join(self.config_dir, file)
            if os.path.exists(path):
                data = self._load_yaml(path)
                config.update(data)
        
        return config