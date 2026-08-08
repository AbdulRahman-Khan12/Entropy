"""Stage 5 - the event graph.

Fuses the Stage 3 entities, Stage 4 relations and events, and the Stage 5
temporal links into one queryable graph, then exports it in the shapes the
Streamlit app and the report need.

Dependency-free on purpose: plain dicts and an adjacency index rather than
networkx, and DOT output that ``st.graphviz_chart`` renders client-side with
no graphviz binary and no extra package on Community Cloud.

Node types
    entity : one per entity_key ("SPACECRAFT:Cassini")
    event  : one per event_id ("wn-0042:3:17")

Edge types
    RELATION : entity -> entity  (LAUNCHED_BY, ORBITS, ...)
    AGENT    : event  -> entity  (who did it)
    PATIENT  : event  -> entity  (what it was done to)
    LOCATION : event  -> entity  (where)
    TEMPORAL : event  -> event   (BEFORE / AFTER / SIMULTANEOUS)
"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable

__all__ = ["Node", "Edge", "EventGraph"]


# --------------------------------------------------------------------------
# presentation
# --------------------------------------------------------------------------
ENTITY_COLOURS = {
    "SPACECRAFT": "#4C78A8",
    "CELESTIAL": "#B279A2",
    "ORG": "#F58518",
    "PERSON": "#54A24B",
    "GPE": "#9D755D",
    "FAC": "#9D755D",
    "LOC": "#9D755D",
    "NORP": "#E45756",
    "DATE": "#BAB0AC",
    "TIME": "#BAB0AC",
}
EVENT_COLOURS = {
    "launch": "#F58518",
    "landing": "#54A24B",
    "docking": "#4C78A8",
    "flyby": "#B279A2",
    "discovery": "#72B7B2",
    "failure": "#E45756",
    "delay": "#EECA3B",
}
EDGE_STYLES = {
    "RELATION": ("solid", "#333333"),
    "AGENT": ("solid", "#54A24B"),
    "PATIENT": ("solid", "#4C78A8"),
    "LOCATION": ("dotted", "#9D755D"),
    "TEMPORAL": ("dashed", "#888888"),
}


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------
@dataclass
class Node:
    node_id: str
    node_type: str        # entity | event
    label: str            # display name
    category: str         # entity label, or event_type for events
    doc_count: int = 0
    mention_count: int = 0
    time_value: str = ""  # events only
    doc_id: str = ""      # events only (events are document-local)
    negated: bool = False

    @property
    def colour(self) -> str:
        table = ENTITY_COLOURS if self.node_type == "entity" else EVENT_COLOURS
        return table.get(self.category, "#BAB0AC")


@dataclass
class Edge:
    source: str
    target: str
    edge_type: str        # RELATION | AGENT | PATIENT | LOCATION | TEMPORAL
    label: str
    doc_id: str = ""
    sent_id: int = -1
    confidence: float = 0.0
    weight: int = 1


NODE_FIELDS = tuple(f.name for f in fields(Node))
EDGE_FIELDS = tuple(f.name for f in fields(Edge))


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------
@dataclass
class EventGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    _adjacency: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    # ---------------------------------------------------------------- build
    @classmethod
    def build(
        cls,
        relations: Iterable[dict],
        timed_events: Iterable[object] = (),
        temporal_links: Iterable[object] = (),
        min_relation_confidence: float = 0.0,
    ) -> "EventGraph":
        graph = cls()

        for row in relations:
            if float(row.get("confidence", 0) or 0) < min_relation_confidence:
                continue
            subj = graph._ensure_entity(
                row["subj_key"], row.get("subj_canonical") or row.get("subj", ""),
                row.get("subj_label", ""), row.get("doc_id", ""),
            )
            obj = graph._ensure_entity(
                row["obj_key"], row.get("obj_canonical") or row.get("obj", ""),
                row.get("obj_label", ""), row.get("doc_id", ""),
            )
            graph._add_edge(Edge(
                source=subj.node_id, target=obj.node_id, edge_type="RELATION",
                label=row["relation"], doc_id=row.get("doc_id", ""),
                sent_id=int(row.get("sent_id", -1) or -1),
                confidence=float(row.get("confidence", 0) or 0),
            ))

        for event in timed_events:
            data = event if isinstance(event, dict) else asdict(event)
            node = Node(
                node_id=data["event_id"], node_type="event",
                label=f"{data['event_type']}: {data['trigger_word']}",
                category=data["event_type"], doc_count=1, mention_count=1,
                time_value=data.get("time_value", ""), doc_id=data.get("doc_id", ""),
                negated=bool(data.get("negated", False)),
            )
            graph.nodes[node.node_id] = node
            for key_field, text_field, edge_type in (
                ("agent_key", "agent", "AGENT"),
                ("patient_key", "patient", "PATIENT"),
                ("location_key", "location", "LOCATION"),
            ):
                key = (data.get(key_field) or "").strip()
                if not key:
                    continue
                label = (data.get(text_field) or "").strip()
                entity = graph._ensure_entity(
                    key, label, key.split(":", 1)[0], data.get("doc_id", "")
                )
                graph._add_edge(Edge(
                    source=node.node_id, target=entity.node_id, edge_type=edge_type,
                    label=edge_type.title(), doc_id=data.get("doc_id", ""),
                    sent_id=int(data.get("sent_id", -1) or -1),
                    confidence=float(data.get("confidence", 0) or 0),
                ))

        for link in temporal_links:
            data = link if isinstance(link, dict) else asdict(link)
            source, target = data["source_event_id"], data["target_event_id"]
            if source not in graph.nodes or target not in graph.nodes:
                continue
            graph._add_edge(Edge(
                source=source, target=target, edge_type="TEMPORAL",
                label=data["relation"], doc_id=data.get("doc_id", ""),
                confidence=float(data.get("confidence", 0) or 0),
            ))
        return graph

    def _ensure_entity(
        self, key: str, label: str, entity_label: str, doc_id: str
    ) -> Node:
        node = self.nodes.get(key)
        if node is None:
            category = entity_label or (key.split(":", 1)[0] if ":" in key else "")
            display = label or (key.split(":", 1)[-1] if ":" in key else key)
            node = Node(node_id=key, node_type="entity", label=display,
                        category=category)
            self.nodes[key] = node
            node._docs = {doc_id} if doc_id else set()  # type: ignore[attr-defined]
        elif doc_id:
            node._docs.add(doc_id)  # type: ignore[attr-defined]
        node.mention_count += 1
        node.doc_count = len(getattr(node, "_docs", ()) or ())
        return node

    def _add_edge(self, edge: Edge) -> None:
        index = len(self.edges)
        self.edges.append(edge)
        self._adjacency[edge.source].append(index)
        self._adjacency[edge.target].append(index)

    # --------------------------------------------------------------- query
    def neighbours(
        self, node_id: str, edge_types: Iterable[str] | None = None
    ) -> list[tuple[Edge, Node]]:
        wanted = set(edge_types) if edge_types else None
        out: list[tuple[Edge, Node]] = []
        for index in self._adjacency.get(node_id, ()):
            edge = self.edges[index]
            if wanted and edge.edge_type not in wanted:
                continue
            other_id = edge.target if edge.source == node_id else edge.source
            other = self.nodes.get(other_id)
            if other is not None:
                out.append((edge, other))
        return out

    def subgraph(
        self, seeds: Iterable[str], hops: int = 1, max_nodes: int = 60
    ) -> "EventGraph":
        """Breadth-first neighbourhood around one or more nodes.

        ``max_nodes`` keeps the Streamlit DOT render legible - hub entities like
        NASA (772 mentions) would otherwise pull in most of the corpus.
        """
        keep: set[str] = {s for s in seeds if s in self.nodes}
        frontier = set(keep)
        for _ in range(max(hops, 0)):
            nxt: set[str] = set()
            for node_id in frontier:
                for _, other in self.neighbours(node_id):
                    if other.node_id not in keep and len(keep) < max_nodes:
                        keep.add(other.node_id)
                        nxt.add(other.node_id)
            if not nxt:
                break
            frontier = nxt

        out = EventGraph()
        out.nodes = {nid: self.nodes[nid] for nid in keep}
        for edge in self.edges:
            if edge.source in keep and edge.target in keep:
                out._add_edge(edge)
        return out

    def entity_timeline(self, entity_key: str) -> list[Node]:
        """Every event this entity takes part in, in chronological order."""
        events = [
            node for _, node in self.neighbours(
                entity_key, {"AGENT", "PATIENT", "LOCATION"}
            ) if node.node_type == "event"
        ]
        unique = {node.node_id: node for node in events}
        return sorted(
            unique.values(),
            key=lambda n: (n.time_value == "", n.time_value, n.node_id),
        )

    def aggregate_relations(self) -> list[Edge]:
        """Collapse repeated instances into one weighted entity-to-entity edge."""
        buckets: dict[tuple[str, str, str], list[Edge]] = defaultdict(list)
        for edge in self.edges:
            if edge.edge_type == "RELATION":
                buckets[(edge.source, edge.target, edge.label)].append(edge)
        out: list[Edge] = []
        for (source, target, label), group in buckets.items():
            out.append(Edge(
                source=source, target=target, edge_type="RELATION", label=label,
                confidence=round(max(e.confidence for e in group), 3),
                weight=len(group),
                doc_id="", sent_id=-1,
            ))
        return sorted(out, key=lambda e: (-e.weight, e.label))

    def stats(self) -> dict:
        node_types = Counter(n.node_type for n in self.nodes.values())
        categories = Counter(n.category for n in self.nodes.values())
        edge_types = Counter(e.edge_type for e in self.edges)
        degrees = {nid: len(idx) for nid, idx in self._adjacency.items()}
        top = sorted(degrees.items(), key=lambda kv: -kv[1])[:10]
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "node_types": dict(node_types),
            "node_categories": dict(categories),
            "edge_types": dict(edge_types),
            "isolated_nodes": sum(1 for n in self.nodes if n not in self._adjacency),
            "mean_degree": round(sum(degrees.values()) / len(self.nodes), 2)
            if self.nodes else 0.0,
            "top_by_degree": [
                {"node_id": nid, "label": self.nodes[nid].label, "degree": deg}
                for nid, deg in top if nid in self.nodes
            ],
        }

    # -------------------------------------------------------------- export
    def to_dot(
        self, title: str = "", rankdir: str = "LR", show_event_times: bool = True
    ) -> str:
        """Graphviz DOT source for ``st.graphviz_chart``."""
        lines = ["digraph entropy {",
                 f'  rankdir={rankdir};',
                 '  bgcolor="transparent";',
                 '  node [fontname="Helvetica", fontsize=10, style="filled", '
                 'fontcolor="white", penwidth=0];',
                 '  edge [fontname="Helvetica", fontsize=8, color="#666666"];']
        if title:
            lines.append(f'  label="{_dot_escape(title)}"; labelloc=t; fontsize=12;')

        for node in self.nodes.values():
            if node.node_type == "entity":
                shape, parts = "ellipse", [node.label]
            else:
                shape = "box"
                parts = [node.label]
                if show_event_times and node.time_value:
                    parts.append(node.time_value)
                if node.negated:
                    parts.append("(not / delayed)")
            lines.append(
                f'  "{_dot_escape(node.node_id)}" '
                f'[label="{_dot_label(parts)}", shape={shape}, '
                f'fillcolor="{node.colour}"];'
            )

        for edge in self.edges:
            style, colour = EDGE_STYLES.get(edge.edge_type, ("solid", "#666666"))
            label = edge.label if edge.weight == 1 else f"{edge.label} ×{edge.weight}"
            lines.append(
                f'  "{_dot_escape(edge.source)}" -> "{_dot_escape(edge.target)}" '
                f'[label="{_dot_escape(label)}", style={style}, color="{colour}", '
                f'fontcolor="{colour}"];'
            )
        lines.append("}")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "nodes": [
                {k: v for k, v in asdict(node).items()} for node in self.nodes.values()
            ],
            "edges": [asdict(edge) for edge in self.edges],
            "stats": self.stats(),
        }

    def write(self, out_dir: Path, prefix: str = "graph") -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        nodes_path = out_dir / f"{prefix}_nodes.csv"
        with nodes_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(NODE_FIELDS))
            writer.writeheader()
            for node in self.nodes.values():
                writer.writerow({k: getattr(node, k) for k in NODE_FIELDS})
        written.append(nodes_path)

        edges_path = out_dir / f"{prefix}_edges.csv"
        with edges_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(EDGE_FIELDS))
            writer.writeheader()
            writer.writerows(asdict(edge) for edge in self.edges)
        written.append(edges_path)

        json_path = out_dir / f"{prefix}.json"
        json_path.write_text(
            json.dumps(self.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written.append(json_path)
        return written


def _dot_escape(text: str) -> str:
    return html.unescape(str(text)).replace("\\", "\\\\").replace('"', '\\"')


def _dot_label(parts: list[str]) -> str:
    """Join label lines with a real DOT newline.

    Each part is escaped first; the separator is added afterwards so the
    backslash-escaping pass cannot turn the line break into literal "\\n".
    """
    return "\\n".join(_dot_escape(p) for p in parts if p)
