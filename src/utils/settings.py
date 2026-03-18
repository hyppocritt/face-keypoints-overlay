import os
from pathlib import Path
from dotenv import load_dotenv
import copy

from src.utils.io import load_yaml
from src.paths import WEIGHTS_DIR, CONFIG_DIR, DATA_DIR


class Settings(dict):

    """
    Simple settings class to store all parameters.
    """

    def __init__(self, data: dict):
        """
        Makes Settings object out of a dictionary.

        Args:
            data (dict): Input dictionary.
        """
        super().__init__(data)

    def __getattr__(self, key):

        if key not in self:
            raise AttributeError(key)

        value = self[key]
        if isinstance(value, dict):
            value = Settings(value)
            self[key] = value

        return value
    
    @staticmethod
    def _deep_update(base: dict, new: dict) -> dict:

        """
        Recursively updates a dictionary with values from another dictionary.

        Nested dictionaries are merged. Keys with value None are ignored.


        Args:
            base (dict): Base dictionary to update.
            new (dict): Dictionary containing new values.

        Returns:
            dict: Updated dictionary.
        """

        for k, v in new.items():

            if v is None:
                continue

            if (k in base) and isinstance(base[k], dict) and isinstance(v, dict):

                base[k] = Settings._deep_update(base[k], v)

            else:
                base[k] = v

        return base
    
    @staticmethod
    def _parse_value(value: str):

        if value.lower() == 'none':
            return None
        
        if value.lower() in {'true', 'false'}:
            return value.lower() == 'true'
        
        if value.isdigit():
            return int(value)
        
        try:
            return float(value)
        except Exception:
            pass

        return value
    
    
    @staticmethod
    def _parse_overrides(overrides: list[str]) -> dict:

        """
        Turns a list of flat string overrides into a nested dictionary.

        Args:
            overrides (list[str]): List of flat string overrides.

        Returns:
            dict: Nested dictionary of overrides.
        """

        if not overrides:
            return {}
        
        result = {}

        for string in overrides:

            full_key, value = list(map( lambda x: x.strip(), string.split('=', 1) ))
            keys = full_key.split('.')

            current = result

            for key in keys[:-1]:
                    current = current.setdefault(key, {})

            current[keys[-1]] = Settings._parse_value(value)
        
        return result

    
    @classmethod
    def from_sources(cls, config_path: str | Path = None, overrides: list[str] = None):

        """
        Builds Settings object from overrides, YAML config and ENV variables.

        Priority:
        overrides > YAML > ENV > defaults

        Args:
            config_path (str | Path): Path to YAML config file. If None, uses ./configs/default.yaml
            overrides (list[str]): List of overrides in format "key=value" where key may be "key_1.key_2". 
                                   These values can override config settings.

        Returns:
            Settings: Settings object containing merged dictionary of settings.

        Raises:
            FileNotFoundError: If invalid config path is passed.
        """

        load_dotenv()

        env_dict = {
            'data': {
                'path': Path(os.getenv('DATA_DIR', DATA_DIR)).resolve(),
                'metadata_path': Path(os.getenv('METADATA_PATH', None)).resolve() \
                                 if os.getenv('METADATA_PATH', None) is not None \
                                 else None
            },
            'weights_dir': Path(os.getenv('WEIGHTS_DIR', WEIGHTS_DIR)).resolve(),
            'config_dir': Path(os.getenv('CONFIG_DIR', CONFIG_DIR)).resolve()

        }   

        settings_dict = copy.deepcopy(env_dict)

        if config_path is None:
            config_path = settings_dict['config_dir'] / 'default.yaml'

        config_path = Path(config_path).resolve()

        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found at {config_path}. Please specify a valid config path.')

        config_dict = load_yaml(config_path)

        cls._deep_update(settings_dict, config_dict)

        overrides_dict = cls._parse_overrides(overrides)

        cls._deep_update(settings_dict, overrides_dict)

        settings = cls(settings_dict)

        debug_dict = {
            'config_path': config_path,
            'config_name': config_path.name,
            'config_args': config_dict,
            'cli_args': overrides_dict,
            'env_args': env_dict
        }

        settings.debug = debug_dict

        return settings
    
    
    def to_dict(self):

        res = {}
        for k, v in self.items():
            
            if isinstance(v, Settings):
                res[k] = v.to_dict()

            else: res[k] = v

        return res
    

    def resolve_model_path(self):

        model_args = getattr(self, 'model', None)

        if model_args is not None:

            model_path = getattr(model_args, 'path', None)

            if model_path is not None:
                return Path(model_path).resolve()
        
            if getattr(self, 'weights_dir', None) and \
            getattr(model_args, 'filename', None):
            
                return Path(self.weights_dir).resolve() / self.model.filename
        
        raise ValueError('model_path or (weights_dir + model.filename) must be provided.')
