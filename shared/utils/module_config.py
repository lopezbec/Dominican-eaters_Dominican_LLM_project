from pathlib import Path
from typing import Optional


class ModuleConfig:
    
    def __init__(
        self,
        module_name: str,
        input_file: Optional[str] = None,
        output_subdir: str = "audio_processing/data"
    ):
        self.module_name = module_name
        self._input_file = input_file
        self._output_subdir = output_subdir
        
        self.PROJECT_ROOT = self._get_project_root()
        self.OUTPUT_FILE = self._generate_output_path("xlsx")
        self.OUTPUT_CSV = self._generate_output_path("csv")
        
        if input_file:
            self.INPUT_FILE = input_file
    
    def _get_project_root(self) -> Path:
        current_file = Path(__file__).resolve()
        return current_file.parent.parent.parent
    
    def _generate_output_path(self, extension: str) -> str:
        filename = f"dominican_{self.module_name}.{extension}"
        return f"../{self._output_subdir}/{filename}"
    
    def get_absolute_output_path(self, extension: str) -> Path:
        return self.PROJECT_ROOT / self._output_subdir / f"dominican_{self.module_name}.{extension}"
    
    def get_absolute_input_path(self) -> Optional[Path]:
        if hasattr(self, 'INPUT_FILE'):
            return self.PROJECT_ROOT / self.module_name / self.INPUT_FILE
        return None
