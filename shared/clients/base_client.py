from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


import logging

logger = logging.getLogger(__name__)

class BaseHTTPClient(ABC):
    
    def __init__(
        self,
        base_url: str = "",
        timeout: int = 10,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        user_agent: Optional[str] = None
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.user_agent = user_agent or "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
        self._session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(self._get_default_headers())
        # Configure retry strategy to respect Retry-After header (for 429 responses)
        # and avoid raising immediately on status so urllib3 can consult Retry-After.
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_default_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": self.user_agent
        }
    
    def _make_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Optional[requests.Response]:
        timeout = timeout or self.timeout
        full_url = f"{self.base_url}{url}" if self.base_url and not url.startswith("http") else url
        
        try:
            response = self._session.request(
                method=method,
                url=full_url,
                params=params,
                json=json,
                timeout=timeout,
                **kwargs
            )
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout:
            self._handle_timeout(url, timeout)
            return None
        except requests.exceptions.HTTPError as e:
            self._handle_http_error(e)
            return None
        except requests.exceptions.RequestException as e:
            self._handle_request_error(e)
            return None
    
    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Optional[requests.Response]:
        return self._make_request("GET", url, params=params, timeout=timeout, **kwargs)
    
    def post(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Optional[requests.Response]:
        return self._make_request("POST", url, json=json, timeout=timeout, **kwargs)
    
    def _handle_timeout(self, url: str, timeout: int):
        logger.warning("Timeout: Request to %s exceeded %ss", url, timeout)
    
    def _handle_http_error(self, error: requests.exceptions.HTTPError):
        # If it's a 429 Too Many Requests, try to respect the Retry-After header
        resp = getattr(error, "response", None)
        if resp is not None and resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = None
            try:
                if retry_after is not None:
                    # Retry-After can be seconds or HTTP-date; try seconds first
                    wait = int(retry_after)
            except Exception:
                wait = None

            if wait is None:
                # Fallback heuristic based on backoff_factor
                try:
                    wait = max(1, int(self.backoff_factor * 5))
                except Exception:
                    wait = 5

            logger.warning(
                "HTTP 429 received from %s. Respecting Retry-After: sleeping %s seconds (header=%s)",
                getattr(resp, 'url', 'unknown'), wait, retry_after,
            )
            try:
                time.sleep(wait)
            except Exception:
                pass

        logger.error("HTTP Error: %s", error)
    
    def _handle_request_error(self, error: requests.exceptions.RequestException):
        logger.error("Request Error: %s", error)
    
    def close(self):
        if self._session:
            self._session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class BaseAPIClient(BaseHTTPClient):
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 10,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        user_agent: Optional[str] = None
    ):
        self.api_key = api_key  # Set BEFORE calling super().__init__ to avoid AttributeError
        super().__init__(base_url, timeout, max_retries, backoff_factor, user_agent)
    
    def _get_default_headers(self) -> Dict[str, str]:
        headers = super()._get_default_headers()
        if self.api_key:
            headers.update(self._get_auth_headers())
        return headers
    
    @abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        pass


class BaseScraperClient(BaseHTTPClient):
    
    def __init__(
        self,
        base_url: str = "",
        timeout: int = 10,
        delay: float = 1.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        user_agent: Optional[str] = None
    ):
        default_ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        super().__init__(base_url, timeout, max_retries, backoff_factor, user_agent or default_ua)
        self.delay = delay
    
    def _respect_delay(self):
        if self.delay > 0:
            time.sleep(self.delay)
    
    def get_with_delay(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Optional[requests.Response]:
        response = self.get(url, params, timeout, **kwargs)
        self._respect_delay()
        return response
