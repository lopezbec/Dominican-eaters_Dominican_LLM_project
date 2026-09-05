"""Canonical, restartable lyrics collection domain."""

from .contracts import (
    LYRICS_SCHEMA_VERSION,
    CollectionIssue,
    CollectionResult,
    CollectionStage,
    CollectionStatus,
    LyricsManifest,
    LyricsRequest,
    LyricsValidationError,
    SongRecord,
    VideoMatch,
)
from .genius import GeniusAPI
from .manifest import load_lyrics_manifest, write_lyrics_manifest
from .ports import (
    GeniusCandidate,
    GeniusProvider,
    GeniusSongDetails,
    MusicVideoProvider,
    NoopRateLimiter,
    ProviderError,
    RateLimiter,
)
from .service import LyricsCollectionService, select_candidate
from .store import (
    LEDGER_FILENAME,
    CollectionLedger,
    LedgerConflictError,
    LyricsCollectionRunner,
    LyricsLedgerStore,
)
from .youtube import YouTubeMusicVideoSearch

__all__ = [
    "LEDGER_FILENAME",
    "LYRICS_SCHEMA_VERSION",
    "CollectionIssue",
    "CollectionLedger",
    "CollectionResult",
    "CollectionStage",
    "CollectionStatus",
    "GeniusCandidate",
    "GeniusAPI",
    "GeniusProvider",
    "GeniusSongDetails",
    "LedgerConflictError",
    "LyricsCollectionRunner",
    "LyricsCollectionService",
    "LyricsLedgerStore",
    "LyricsManifest",
    "LyricsRequest",
    "LyricsValidationError",
    "MusicVideoProvider",
    "NoopRateLimiter",
    "ProviderError",
    "RateLimiter",
    "SongRecord",
    "VideoMatch",
    "YouTubeMusicVideoSearch",
    "load_lyrics_manifest",
    "select_candidate",
    "write_lyrics_manifest",
]
