from typing import List

try:
    import pandas as pd
except ImportError:
    pd = None


class DataLoader:
    
    def load_urls_from_csv(self, csv_path: str, url_column: str) -> List[str]:
        if pd is None:
            raise ImportError("pandas is required. Install with: pip install pandas")
        
        df = pd.read_csv(csv_path)
        urls = []
        for idx, row in df.iterrows():
            url = row.get(url_column, '')
            if isinstance(url, str) and url.startswith('http') and 'youtube.com' in url:
                urls.append(url)
        return urls
    
    def load_urls_from_excel(self, excel_path: str, url_column: str) -> List[str]:
        if pd is None:
            raise ImportError(
                "pandas and openpyxl are required. Install with: pip install pandas openpyxl"
            )
        
        df = pd.read_excel(excel_path)
        urls = []
        for idx, row in df.iterrows():
            url = row.get(url_column, '')
            if isinstance(url, str) and url.startswith('http') and 'youtube.com' in url:
                urls.append(url)
        return urls
