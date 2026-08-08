"""Stage 5 driver - temporal resolution and event graph construction.

Reads the Stage 4 CSVs plus the frozen corpus (for each document's creation
time) and writes every artefact the Streamlit app needs. It never touches the
spaCy cache or loads a model, which is what makes the app deployable inside the
~1 GB Community Cloud budget.

    python scripts/build_graph.py
    python scripts/build_graph.py --min-relation-confidence 0.6
    python scripts/build_graph.py --dot-for "SPACECRAFT:Cassini" --hops 2

Outputs (outputs/):
    timed_events.csv     events with ISO-8601 resolved times
    temporal_links.csv   pairwise BEFORE/AFTER/SIMULTANEOUS assertions
    timeline.csv         corpus-wide chronological event ordering
    graph_nodes.csv      entity + event nodes
    graph_edges.csv      relation / participation / temporal edges
    graph.json           the whole graph, for the app
    graph_stats.json     summary counts for the report
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entropy.graph import EventGraph  # noqa: E402
from entropy.relations import output_dir  # noqa: E402
from entropy.temporal import (  # noqa: E402
    attach_times,
    build_timeline,
    order_events,
    write_temporal_links_csv,
    write_timed_events_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 5: temporal ordering and event graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-relation-confidence", type=float, default=0.0,
                        help="drop weak Stage 4 relations before graph building")
    parser.add_argument("--rule-only", action="store_true", default=True,
                        help="use only rule-based relations (exclude HF baseline rows)")
    parser.add_argument("--include-hf", dest="rule_only", action="store_false",
                        help="also put HF baseline relations in the graph")
    parser.add_argument("--max-events-per-doc", type=int, default=40,
                        help="cap pairwise temporal links in very long documents")
    parser.add_argument("--dot-for", default=None, metavar="ENTITY_KEY",
                        help="also write a DOT file centred on one entity")
    parser.add_argument("--hops", type=int, default=1,
                        help="neighbourhood radius for --dot-for")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _as_date(value) -> date | None:
    """Coerce whatever corpus.Document.dct holds into a date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "")
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%d %B %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(text[: len(fmt) + 8], fmt).date()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text).date()
        except ValueError:
            return None
    return None


def load_dcts() -> dict[str, date]:
    """Document creation times, the anchor for every relative date."""
    from entropy.corpus import iter_corpus

    dcts: dict[str, date] = {}
    for document in iter_corpus():
        resolved = _as_date(getattr(document, "dct", None))
        if resolved is not None:
            dcts[document.doc_id] = resolved
    return dcts


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/extract_relations.py (Stage 4).")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    out_dir = args.out_dir or output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    relations = read_csv(out_dir / "relations.csv")
    events = read_csv(out_dir / "events.csv")
    if args.rule_only:
        relations = [r for r in relations if r.get("source", "rule") == "rule"]

    dcts = load_dcts()
    if not args.quiet:
        print(f"Stage 5: {len(relations)} relations | {len(events)} events | "
              f"{len(dcts)} document creation times")

    # ------------------------------------------------------------- temporal
    timed = attach_times(events, dcts)
    links = order_events(timed, max_events_per_doc=args.max_events_per_doc)
    timeline = build_timeline(timed)

    resolved = sum(1 for e in timed if e.time_method != "dct_fallback")
    if not args.quiet:
        methods: dict[str, int] = {}
        for event in timed:
            methods[event.time_method] = methods.get(event.time_method, 0) + 1
        print(f"\nTemporal: {resolved}/{len(timed)} events resolved from a real "
              f"expression ({resolved / len(timed):.0%}), rest anchored to the DCT")
        for method, count in sorted(methods.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  {method:<20} {count}")
        bases: dict[str, int] = {}
        for link in links:
            bases[link.basis] = bases.get(link.basis, 0) + 1
        print(f"\nOrdering: {len(links)} temporal links")
        for basis, count in sorted(bases.items(), key=lambda kv: -kv[1]):
            print(f"  {basis:<20} {count}")

    # ---------------------------------------------------------------- graph
    graph = EventGraph.build(
        relations, timed, links,
        min_relation_confidence=args.min_relation_confidence,
    )
    stats = graph.stats()
    if not args.quiet:
        print(f"\nGraph: {stats['nodes']} nodes | {stats['edges']} edges | "
              f"mean degree {stats['mean_degree']}")
        print("  node types:", stats["node_types"])
        print("  edge types:", stats["edge_types"])
        print("  most connected:")
        for entry in stats["top_by_degree"][:5]:
            print(f"    {entry['label']:<32} degree {entry['degree']}")

    # -------------------------------------------------------------- outputs
    written = [
        write_timed_events_csv(timed, out_dir / "timed_events.csv"),
        write_temporal_links_csv(links, out_dir / "temporal_links.csv"),
        write_timed_events_csv(timeline, out_dir / "timeline.csv"),
    ]
    written += graph.write(out_dir)
    stats_path = out_dir / "graph_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    written.append(stats_path)

    if args.dot_for:
        if args.dot_for not in graph.nodes:
            print(f"  '{args.dot_for}' is not a node; skipping DOT export",
                  file=sys.stderr)
        else:
            focus = graph.subgraph([args.dot_for], hops=args.hops)
            dot_path = out_dir / "graph_focus.dot"
            dot_path.write_text(
                focus.to_dot(title=args.dot_for), encoding="utf-8"
            )
            written.append(dot_path)

    if not args.quiet:
        print(f"\ndone in {time.perf_counter() - started:.1f}s")
        for path in written:
            print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
