import logging
from typing import List
from tqdm import tqdm


class ProgressReporter:
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def report_batch(self, items: List, description: str):
        return tqdm(items, desc=description)
    
    def report_success(self, message: str):
        self.logger.info(message)
    
    def report_error(self, message: str):
        self.logger.error(message)
    
    def report_warning(self, message: str):
        self.logger.warning(message)
