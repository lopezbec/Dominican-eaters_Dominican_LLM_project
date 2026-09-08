"""Transport adapters shared by collection domains."""

from .youtube import (
    ScrapeTubeSearch,
    YouTubeSearchError,
    YouTubeSearchProvider,
    YouTubeVideo,
    parse_display_duration,
)

__all__ = [
    "ScrapeTubeSearch",
    "YouTubeSearchError",
    "YouTubeSearchProvider",
    "YouTubeVideo",
    "parse_display_duration",
]
