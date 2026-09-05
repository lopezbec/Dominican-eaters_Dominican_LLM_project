"""Transport adapters shared by collection domains."""

from .youtube import YouTubeAPIError, YouTubeDataAPI, YouTubeVideo, parse_youtube_duration

__all__ = ["YouTubeAPIError", "YouTubeDataAPI", "YouTubeVideo", "parse_youtube_duration"]
