"""Clean poem recitation collection domain."""

from .artifacts import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointState,
    CollectionCheckpoint,
    load_checkpoint,
    write_checkpoint,
)
from .manifest import (
    POEM_MANIFEST_SCHEMA_VERSION,
    PoemManifest,
    load_poem_manifest,
    write_poem_manifest,
)
from .matching import RankedCandidate, classify_content, rank_candidate, select_candidate
from .models import (
    ContentKind,
    OutcomeStatus,
    PoemOutcome,
    PoemSource,
    PoemValidationError,
    VideoCandidate,
    derive_source_id,
    normalized_identity,
)
from .ports import ProviderError, RecitationQuery, RecitationSearch
from .service import PoemCollector
from .youtube import YouTubeRecitationSearch

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointState",
    "CollectionCheckpoint",
    "ContentKind",
    "OutcomeStatus",
    "POEM_MANIFEST_SCHEMA_VERSION",
    "PoemCollector",
    "PoemManifest",
    "PoemOutcome",
    "PoemSource",
    "PoemValidationError",
    "ProviderError",
    "RankedCandidate",
    "RecitationQuery",
    "RecitationSearch",
    "VideoCandidate",
    "YouTubeRecitationSearch",
    "classify_content",
    "derive_source_id",
    "load_checkpoint",
    "load_poem_manifest",
    "normalized_identity",
    "rank_candidate",
    "select_candidate",
    "write_checkpoint",
    "write_poem_manifest",
]
