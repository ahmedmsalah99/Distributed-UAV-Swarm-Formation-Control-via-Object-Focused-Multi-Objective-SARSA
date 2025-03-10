import yaml
import os

class ParametersManager:
    instance = None  # Singleton instance

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super(ParametersManager, cls).__new__(cls)
        return cls.instance
    
    def __new__(cls, config_file):
        if cls.instance is None:
            cls.instance = super(ParametersManager, cls).__new__(cls)
            cls.instance.parse(config_file)
        return cls.instance
        

    def parse(self, config_file):
        """Load the YAML configuration file."""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' not found.")
        
        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)

    def get_mode(self):
        """Get the mode (e.g., 'train')."""
        return self.config.get("mode", ["train"])[0]  # Default to 'train' if not specified

    def get_env_params(self):
        """Get environment parameters."""
        return self.config.get("env", {})

    def get_active_modules(self):
        """Get active modules and their parameters."""
        return self.config.get("active_modules", {})
    def get_training_params(self):
        return self.config.get("training_params", {})


