import yaml
from typing import Dict


class ConfigLoader:
    
    def load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_module_config(self, config: Dict, module_name: str) -> Dict:
        if 'modules' not in config or module_name not in config['modules']:
            raise ValueError(f"Module '{module_name}' not found in config")
        return config['modules'][module_name]
