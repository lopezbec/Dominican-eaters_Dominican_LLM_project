import logging
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Dict, Optional, Callable


logger = logging.getLogger(__name__)


class BaseEaterRunner(ABC):
    
    def __init__(self, module_name: str, title: str):
        self.module_name = module_name
        self.title = title
        self.config = self.load_config()
    
    @abstractmethod
    def load_input_data(self) -> List[Any]:
        pass

    @abstractmethod
    def get_client(self) -> Any:
        """
        Abstract method to get the specific client for the module.

        Returns:
            Any: The client instance.
        """
        pass

    @abstractmethod
    def get_service(self, client: Any) -> Any:
        """
        Abstract method to get the specific service for the module.

        Args:
            client (Any): The client instance.

        Returns:
            Any: The service instance.
        """
        pass

    @abstractmethod
    def load_config(self) -> Any:
        """
        Abstract method to load the configuration for the module.

        Returns:
            Any: The configuration object.
        """
        pass

    
    @abstractmethod
    @abstractmethod
    def save_results(self, items: List[Any]) -> None:
        pass
    
    def print_header(self) -> None:
        separator = "=" * len(self.title)
                logger.info("\n{separator}")
        print(self.title)
                logger.info("{separator}\n")
    
    def print_results_header(self) -> None:
        separator = "=" * len(self.title)
                logger.info("\n{separator}")
                logger.info("Guardando resultados...")
                logger.info("{separator}\n")
    
    def run(self) -> None:
        """
        Runs the entire data processing flow.
        """
        self.print_header()
        
        try:
            client = self.get_client()
            service = self.get_service(client)
            data = self.load_input_data()
            
            if not data:
                logger.info("No data to process for %s", self.module_name)
                logger.info("No data to process for {self.module_name}.")
                return
                
                logger.info("\n{'='*60}")
                logger.info("Iniciando búsqueda...")
                logger.info("{'='*60}\n")
            
            results, stats = service.process_multiple(data)
            
            if results:
                self.print_results_header()
                self.save_dual_format(results)
                service.print_statistics(stats)
            else:
                logger.info("No items processed in %s", self.module_name)
                logger.info("\nNo se procesaron elementos en {self.module_name}")
                
        except KeyboardInterrupt:
            logger.info("Process interrupted by user")
                logger.info("\n\nPrograma interrumpido por el usuario")
                logger.info("¡Hasta luego!")
        except Exception as e:
            logger.error("Fatal error in %s runner: %s", self.module_name, e)
                logger.info("\nError fatal: {e}")
            raise
    
    def load_with_fallback(
        self,
        file_path: str,
        fallback_provider: Callable[[], List[Any]],
        file_description: str,
        fallback_description: str
    ) -> List[Any]:
        """
        Load data from file with fallback to provider function.
        
        Args:
            file_path: Path to data file
            fallback_provider: Function that returns default data if file not found
            file_description: Description for file loading messages
            fallback_description: Description for fallback data messages
            
        Returns:
            List of loaded items
        """
        from shared.utils.file_handler import FileHandler
        
        items = FileHandler.load_items_from_file(file_path, self._get_model_class())
        
        if items:
            logger.info("Loaded %d items from '%s'", len(items), file_description)
                logger.info("Cargados {len(items)} {file_description}")
        else:
            logger.info("Using fallback data: %s", fallback_description)
                logger.info("Usando {fallback_description}")
            items = fallback_provider()
                logger.info("{len(items)} en el dataset")
        
        return items
    
    def save_dual_format(self, items: List[Any]) -> bool:
        """
        Save results to both Excel and CSV formats.
        
        Args:
            items: List of items to save
            
        Returns:
            True if successful, False otherwise
        """
        from shared.utils.file_handler import FileHandler
        
        success = False
        
        # Filter out None items (for lyrics-eater)
        valid_items = [item for item in items if item is not None]
        
        if valid_items:
            excel_success = FileHandler.save_to_excel(valid_items, self._get_output_file())
            csv_success = FileHandler.save_to_csv(valid_items, self._get_output_csv())
            
            if excel_success or csv_success:
                logger.info("Results saved successfully")
                self._print_output_files()
                success = True
            else:
                logger.error("Failed to save results")
        else:
            logger.warning("No valid items to save")
                logger.info("\n No items were successfully processed")
        
        return success
    
    @abstractmethod
    def _get_model_class(self) -> type:
        """Return the model class for this module."""
        pass
    
    @abstractmethod
    def _get_output_file(self) -> str:
        """Return the output Excel file path."""
        pass
    
    @abstractmethod
    def _get_output_csv(self) -> str:
        """Return the output CSV file path."""
        pass
    
    def _print_output_files(self):
        """Print generated file information."""
                logger.info("Archivos generados:")
                logger.info("   - {self._get_output_file()}")
                logger.info("   - {self._get_output_csv()}")
    
    @staticmethod
    def create_and_run(runner_class) -> None:
        """
        Factory method to create and run a runner.
        
        Args:
            runner_class: Class that inherits from BaseEaterRunner
        """
        try:
            runner = runner_class()
            runner.run()
        except KeyboardInterrupt:
                logger.info("\n\nPrograma interrumpido por el usuario")
                logger.info("¡Hasta luego!")
        except Exception as e:
                logger.info("\nError fatal: {e}")
            raise
