"""Canonical audiobook collection contracts and orchestration."""

from .manifest import (
    BOOK_MANIFEST_SCHEMA_VERSION,
    BookManifest,
    load_book_manifest,
    write_book_manifest,
)
from .models import (
    AudiobookHit,
    BookCollectionCheckpoint,
    BookCollectionStatus,
    BookRecord,
    BookSeed,
    CheckpointState,
    CollectionIssue,
    MatchKind,
)
from .runner import BookCollectionRunner, ConcurrentCollectionError, load_checkpoint
from .service import AudiobookSearchProvider, ProviderError, collect_book
from .youtube import YouTubeAudiobookSearch

__all__ = [
    "AudiobookHit",
    "AudiobookSearchProvider",
    "BOOK_MANIFEST_SCHEMA_VERSION",
    "BookCollectionCheckpoint",
    "BookCollectionRunner",
    "BookCollectionStatus",
    "BookManifest",
    "BookRecord",
    "BookSeed",
    "CheckpointState",
    "CollectionIssue",
    "ConcurrentCollectionError",
    "MatchKind",
    "ProviderError",
    "YouTubeAudiobookSearch",
    "collect_book",
    "load_book_manifest",
    "load_checkpoint",
    "write_book_manifest",
]
