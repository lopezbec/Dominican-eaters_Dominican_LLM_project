"""Basic test examples for BaseHTTPClient and adapters.

These are integration examples - not a full test suite since the project
doesn't have pytest configured yet.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_adapter_compliance():
    """Example showing how to test adapter protocol compliance."""
    
    # Test YouTubeSearcherAdapter compliance
    from shared.clients.adapters import YouTubeSearcherAdapter
    
    adapter = YouTubeSearcherAdapter()
    
    # Check required methods exist
    assert hasattr(adapter, 'search_audiobook'), "Should have search_audiobook method"
    assert hasattr(adapter, 'search_poem_recitation'), "Should have search_poem_recitation method"
    assert hasattr(adapter, 'search_music_video'), "Should have search_music_video method"
    
    print("✓ YouTubeSearcherAdapter has required methods")


def test_genius_adapter_compliance():
    """Example showing how to test Genius adapter compliance."""
    
    # Test GeniusSearcherAdapter compliance
    from shared.clients.adapters import GeniusSearcherAdapter
    from shared.clients.genius_client import GeniusAPIClient
    
    # Mock client for testing (would use real token in production)
    mock_client = GeniusAPIClient(access_token="mock_token")
    adapter = GeniusSearcherAdapter(mock_client)
    
    # Check required methods exist
    assert hasattr(adapter, 'search'), "Should have search method"
    assert hasattr(adapter, 'get_song_details'), "Should have get_song_details method"
    assert hasattr(adapter, 'scrape_lyrics'), "Should have scrape_lyrics method"
    
    print("✓ GeniusSearcherAdapter has required methods")


def test_base_http_client_creation():
    """Example showing BaseHTTPClient creation."""
    
    from shared.clients.base_client import BaseHTTPClient
    
    # Test that client can be created
    try:
        client = BaseHTTPClient(
            base_url="https://httpbin.org/status/500",
            max_retries=2,
            backoff_factor=0.1
        )
        print("✓ BaseHTTPClient created successfully")
    except Exception as e:
        print(f"✗ BaseHTTPClient creation failed: {e}")
        raise


if __name__ == "__main__":
    print("Running basic compliance tests...")
    test_adapter_compliance()
    test_genius_adapter_compliance()
    test_base_http_client_creation()
    print("All basic tests passed!")