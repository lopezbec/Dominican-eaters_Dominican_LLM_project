import os
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ReportExporter:
    
    def export_json(self, data: Dict, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to: {output_path}")
