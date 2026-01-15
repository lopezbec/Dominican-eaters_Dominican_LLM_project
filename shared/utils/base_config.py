from pathlib import Path
from typing import Any, Dict
from shared.utils.module_config import ModuleConfig


class BaseConfig:
    
    def __init__(self, module_name: str, input_file: str, **kwargs):
        self._base = ModuleConfig(module_name=module_name, input_file=input_file)
        self.module_name = module_name
        self.input_file = input_file
        self._extra_config = kwargs
    
    def __getattr__(self, name: str) -> Any:
        if name in self._extra_config:
            return self._extra_config[name]
        return getattr(self._base, name)
    
    @property
    def PROJECT_ROOT(self) -> Path:
        return self._base.PROJECT_ROOT
    
    @property
    def INPUT_FILE(self) -> str:
        return self._base.INPUT_FILE
    
    @property
    def OUTPUT_FILE(self) -> str:
        return self._base.OUTPUT_FILE
    
    @property
    def OUTPUT_CSV(self) -> str:
        return self._base.OUTPUT_CSV
    
    def validate(self) -> bool:
        return True
