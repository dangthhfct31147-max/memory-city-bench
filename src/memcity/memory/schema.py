"""Memory City data schemas — nodes, edges, episodes."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    EPISODE = "episode"
    BEGIN = "begin"
    END = "end"
    ENTITY = "entity"
    TOPIC_HUB = "topic_hub"
    COMMUNITY = "community"
    FACT = "fact"
    # Hierarchical summary tier (Phase 6, RAPTOR-style). A SUMMARY node condenses
    # a group of children (a session's episodes, or lower-level summaries) into an
    # extractive digest. Summaries form a navigable tree above the raw episodes;
    # they only ever *route* a global query down to raw evidence, never become
    # final evidence themselves.
    SUMMARY = "summary"


class EdgeType(str, Enum):
    BEGINS = "BEGINS"
    ENDS_WITH = "ENDS_WITH"
    NEXT = "NEXT"
    SAME_SESSION = "SAME_SESSION"
    SEMANTICALLY_RELATED = "SEMANTICALLY_RELATED"
    MENTIONS = "MENTIONS"
    BELONGS_TO_TOPIC = "BELONGS_TO_TOPIC"
    BELONGS_TO_COMMUNITY = "BELONGS_TO_COMMUNITY"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    SUPERSEDES = "SUPERSEDES"
    DERIVED_FROM = "DERIVED_FROM"
    CAUSED_BY = "CAUSED_BY"
    DEPENDS_ON = "DEPENDS_ON"
    # Hierarchical summary tier (Phase 6). SUMMARIZES points summary→child
    # (a summary summarises its children); PARENT_SUMMARY points child→summary
    # (the upward navigation edge). Kept as two directions so the DiGraph carries
    # both traversal orientations without colliding on one (src, dst) pair.
    SUMMARIZES = "SUMMARIZES"
    PARENT_SUMMARY = "PARENT_SUMMARY"


class CreationMethod(str, Enum):
    DETERMINISTIC = "deterministic"
    HEURISTIC = "heuristic"
    LLM = "llm"


class NodeBase(BaseModel):
    id: str
    node_type: NodeType
    text: str
    source_episode_ids: list[str] = Field(default_factory=list)
    session_id: str = ""
    timestamp: float = Field(default_factory=time.time)
    confidence: float = 1.0
    version: int = 1
    creation_method: CreationMethod = CreationMethod.DETERMINISTIC
    content_hash: str = ""
    # Bi-temporal axes (Phase 3). ``valid_from`` / ``valid_to`` are *event time*
    # (when the fact was true in the world); ``observed_at`` / ``ingested_at`` are
    # *system time* (when the conversation stated it / when we indexed it). Keeping
    # both lets a temporal query ask "what was true before May" (event time) rather
    # than merely "what was said most recently" (system time).
    valid_from: float | None = None
    valid_to: float | None = None
    observed_at: float | None = None
    ingested_at: float | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_current(self, at: float | None = None) -> bool:
        t = at if at is not None else time.time()
        if self.valid_to is not None and self.valid_to < t:
            return False
        if self.valid_from is not None and self.valid_from > t:
            return False
        return True


class EpisodeNode(NodeBase):
    node_type: NodeType = NodeType.EPISODE
    user_text: str = ""
    assistant_text: str = ""
    turn_index: int = 0


class BeginNode(NodeBase):
    node_type: NodeType = NodeType.BEGIN
    episode_id: str = ""


class EndNode(NodeBase):
    node_type: NodeType = NodeType.END
    episode_id: str = ""


class EntityNode(NodeBase):
    node_type: NodeType = NodeType.ENTITY
    entity_label: str = ""


class TopicHubNode(NodeBase):
    node_type: NodeType = NodeType.TOPIC_HUB
    top_terms: list[str] = Field(default_factory=list)
    cluster_id: int = -1


class CommunityReadme(BaseModel):
    topic: str = ""
    top_terms: list[str] = Field(default_factory=list)
    top_entities: list[str] = Field(default_factory=list)
    representative_episodes: list[str] = Field(default_factory=list)
    time_range: tuple[float, float] | None = None
    node_count: int = 0
    edge_count: int = 0
    source_episode_ids: list[str] = Field(default_factory=list)
    creation_method: CreationMethod = CreationMethod.DETERMINISTIC


class CommunityNode(NodeBase):
    node_type: NodeType = NodeType.COMMUNITY
    readme: CommunityReadme = Field(default_factory=CommunityReadme)
    member_ids: list[str] = Field(default_factory=list)


class FactNode(NodeBase):
    node_type: NodeType = NodeType.FACT
    subject: str = ""
    predicate: str = ""
    obj: str = ""
    superseded_by: str | None = None


class SummaryNode(NodeBase):
    """A node in the hierarchical summary tree (Phase 6).

    ``level`` is the tree depth: 1 = session-level summary over raw episodes,
    2 = summary over level-1 summaries, and so on up to the root. ``child_ids``
    are the immediate children this node summarises; ``source_episode_ids`` is the
    flattened set of raw episodes reachable beneath it, so a routed summary can
    surface its leaf evidence directly.
    """

    node_type: NodeType = NodeType.SUMMARY
    level: int = 1
    child_ids: list[str] = Field(default_factory=list)


class Edge(BaseModel):
    id: str
    src_id: str
    dst_id: str
    edge_type: EdgeType
    weight: float = 1.0
    source_episode_ids: list[str] = Field(default_factory=list)
    session_id: str = ""
    timestamp: float = Field(default_factory=time.time)
    confidence: float = 1.0
    version: int = 1
    creation_method: CreationMethod = CreationMethod.DETERMINISTIC
    content_hash: str = ""
    valid_from: float | None = None
    valid_to: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
