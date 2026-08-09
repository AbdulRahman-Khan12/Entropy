"""Entropy - information extraction over a Wikinews space/astronomy corpus.

Streamlit front end for the Stage 3/4/5 artefacts.

Deployment note (OPEN ISSUE #2): this app reads CSVs only. It never opens the
spaCy DocBin cache and never loads a language model, so the whole thing runs
well inside the ~1 GB Streamlit Community Cloud budget. Run
``scripts/export_artifacts.py`` to publish the CSVs into data/artifacts/, which
*is* committed - that is what Cloud sees.

    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT))

from entropy.graph import EventGraph  # noqa: E402  (path shim must run first)

st.set_page_config(page_title="Entropy - Information Extraction",
                   page_icon="🛰️", layout="wide")

# Small files are loaded whole; the tokens_conll dump is deliberately absent.
DATASETS = {
    "entities": "entities.csv",
    "relations": "relations.csv",
    "relation_summary": "relation_summary.csv",
    "events": "events.csv",
    "event_summary": "event_summary.csv",
    "timed_events": "timed_events.csv",
    "temporal_links": "temporal_links.csv",
    "graph_nodes": "graph_nodes.csv",
    "graph_edges": "graph_edges.csv",
}


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def resolve_data_dir() -> tuple[Path, str]:
    """Prefer a fresh local run, fall back to the committed artefacts."""
    import os

    override = os.environ.get("ENTROPY_DATA_DIR")
    if override and (Path(override) / "relations.csv").exists():
        return Path(override), "ENTROPY_DATA_DIR"
    for candidate, source in (
        (APP_ROOT / "outputs", "outputs/ (local run)"),
        (APP_ROOT / "data" / "artifacts", "data/artifacts/ (committed)"),
    ):
        if (candidate / "relations.csv").exists():
            return candidate, source
    return APP_ROOT / "outputs", "missing"


@st.cache_data(show_spinner="Loading extraction artefacts…")
def load_tables(data_dir_str: str) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir_str)
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in DATASETS.items():
        path = data_dir / filename
        tables[name] = (
            pd.read_csv(path, keep_default_na=False, dtype={"sent_id": "Int64"})
            if path.exists()
            else pd.DataFrame()
        )
    return tables


@st.cache_data(show_spinner=False)
def load_stats(data_dir_str: str) -> dict:
    path = Path(data_dir_str) / "graph_stats.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_resource(show_spinner="Building the event graph…")
def build_graph(data_dir_str: str, min_confidence: float) -> EventGraph:
    tables = load_tables(data_dir_str)
    relations = tables["relations"]
    if not relations.empty and "source" in relations.columns:
        relations = relations[relations["source"] == "rule"]
    return EventGraph.build(
        relations.to_dict("records"),
        tables["timed_events"].to_dict("records"),
        tables["temporal_links"].to_dict("records"),
        min_relation_confidence=min_confidence,
    )


def counted(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame()
    return frame[column].value_counts().rename_axis(column).to_frame("count")


# --------------------------------------------------------------------------
# tabs
# --------------------------------------------------------------------------
def tab_overview(tables: dict[str, pd.DataFrame], stats: dict) -> None:
    entities, relations = tables["entities"], tables["relations"]
    events, timed = tables["events"], tables["timed_events"]
    rule_relations = (
        relations[relations["source"] == "rule"]
        if not relations.empty and "source" in relations.columns else relations
    )

    columns = st.columns(5)
    columns[0].metric("Documents", f"{entities['doc_id'].nunique():,}"
                      if not entities.empty else "—")
    columns[1].metric("Entity mentions", f"{len(entities):,}")
    columns[2].metric("Distinct entities", f"{entities['entity_key'].nunique():,}"
                      if not entities.empty else "—")
    columns[3].metric("Relations (rule)", f"{len(rule_relations):,}")
    columns[4].metric("Events", f"{len(events):,}")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Entity labels")
        st.bar_chart(counted(entities, "label"), horizontal=True)
        st.subheader("Relation types")
        st.bar_chart(counted(rule_relations, "relation"), horizontal=True)
    with right:
        st.subheader("Event types")
        st.bar_chart(counted(events, "event_type"), horizontal=True)
        st.subheader("Events per year (resolved times only)")
        if not timed.empty:
            real = timed[timed["time_method"] != "dct_fallback"].copy()
            real["year"] = real["time_value"].astype(str).str.slice(0, 4)
            real = real[real["year"].str.fullmatch(r"\d{4}", na=False)]
            st.bar_chart(real["year"].value_counts().sort_index()
                         .rename_axis("year").to_frame("events"))

    if not tables["relation_summary"].empty:
        st.subheader("Rule-based vs Hugging Face baseline")
        st.caption(
            "Low agreement is expected: the rules fire on specific syntactic "
            "patterns, while zero-shot NLI reasons semantically. They find "
            "different evidence for overlapping facts."
        )
        st.dataframe(tables["relation_summary"],
                     hide_index=True)

    if stats:
        st.subheader("Graph")
        graph_columns = st.columns(4)
        graph_columns[0].metric("Nodes", f"{stats.get('nodes', 0):,}")
        graph_columns[1].metric("Edges", f"{stats.get('edges', 0):,}")
        graph_columns[2].metric("Mean degree", stats.get("mean_degree", 0))
        graph_columns[3].metric("Isolated", f"{stats.get('isolated_nodes', 0):,}")


def tab_entities(tables: dict[str, pd.DataFrame], graph: EventGraph) -> None:
    entities = tables["entities"]
    if entities.empty:
        st.info("entities.csv not found - run scripts/extract_entities.py (Stage 3).")
        return

    summary = (
        entities.groupby(["entity_key", "label", "canonical"])
        .size().reset_index(name="mentions")
        .sort_values("mentions", ascending=False)
    )
    controls = st.columns([2, 2, 1])
    labels = sorted(summary["label"].unique())
    chosen_labels = controls[0].multiselect("Entity label", labels, default=[])
    query = controls[1].text_input("Search", placeholder="NASA, Cassini, Mars…")
    minimum = controls[2].number_input("Min mentions", 1, 500, 1)

    filtered = summary[summary["mentions"] >= minimum]
    if chosen_labels:
        filtered = filtered[filtered["label"].isin(chosen_labels)]
    if query:
        filtered = filtered[
            filtered["canonical"].str.contains(query, case=False, na=False)
        ]
    if filtered.empty:
        st.warning("No entities match those filters.")
        return

    st.caption(f"{len(filtered):,} entities match.")
    options = filtered["entity_key"].head(400).tolist()
    selected = st.selectbox(
        "Entity", options,
        format_func=lambda key: f"{key}  ({int(summary.loc[summary['entity_key'] == key, 'mentions'].iloc[0])} mentions)",
    )
    if not selected:
        return

    st.divider()
    relations = tables["relations"]
    if not relations.empty:
        involved = relations[
            ((relations["subj_key"] == selected) | (relations["obj_key"] == selected))
            & (relations.get("source", "rule") == "rule")
        ]
        st.subheader(f"Relations ({len(involved)})")
        if involved.empty:
            st.caption("No relations extracted for this entity.")
        else:
            st.dataframe(
                involved[["relation", "subj", "obj", "trigger", "pattern",
                          "confidence", "doc_id", "sentence"]],
                hide_index=True,
            )

    timeline = graph.entity_timeline(selected)
    st.subheader(f"Event timeline ({len(timeline)})")
    if not timeline:
        st.caption("This entity does not participate in any extracted event.")
    else:
        st.dataframe(
            pd.DataFrame([
                {"time": node.time_value or "(unresolved)",
                 "event": node.label, "type": node.category,
                 "document": node.doc_id, "negated": node.negated}
                for node in timeline
            ]),
            hide_index=True,
        )


def tab_relations(tables: dict[str, pd.DataFrame]) -> None:
    relations = tables["relations"]
    if relations.empty:
        st.info("relations.csv not found - run scripts/extract_relations.py.")
        return

    controls = st.columns([2, 2, 2, 2])
    types = controls[0].multiselect(
        "Relation", sorted(relations["relation"].unique()), default=[])
    sources = controls[1].multiselect(
        "Source", sorted(relations["source"].unique())
        if "source" in relations.columns else [], default=["rule"])
    patterns = controls[2].multiselect(
        "Pattern", sorted(relations["pattern"].unique()), default=[])
    threshold = controls[3].slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

    filtered = relations[relations["confidence"].astype(float) >= threshold]
    if types:
        filtered = filtered[filtered["relation"].isin(types)]
    if sources and "source" in filtered.columns:
        filtered = filtered[filtered["source"].isin(sources)]
    if patterns:
        filtered = filtered[filtered["pattern"].isin(patterns)]

    st.caption(f"{len(filtered):,} of {len(relations):,} rows.")
    st.dataframe(
        filtered[["doc_id", "relation", "subj", "obj", "trigger", "pattern",
                  "confidence", "sentence"]],
        hide_index=True, height=460,
    )
    st.download_button("Download filtered relations (CSV)",
                       filtered.to_csv(index=False).encode("utf-8"),
                       "relations_filtered.csv", "text/csv")


def tab_events(tables: dict[str, pd.DataFrame]) -> None:
    timed = tables["timed_events"]
    if timed.empty:
        st.info("timed_events.csv not found - run scripts/build_graph.py (Stage 5).")
        return

    controls = st.columns([2, 2, 2, 2])
    types = controls[0].multiselect(
        "Event type", sorted(timed["event_type"].unique()), default=[])
    only_resolved = controls[1].checkbox("Resolved times only", value=True)
    require_args = controls[2].checkbox("Has agent or patient", value=False)
    hide_negated = controls[3].checkbox("Hide negated / cancelled", value=False)

    filtered = timed.copy()
    if types:
        filtered = filtered[filtered["event_type"].isin(types)]
    if only_resolved:
        filtered = filtered[filtered["time_method"] != "dct_fallback"]
    if require_args:
        filtered = filtered[
            (filtered["agent_key"].astype(str) != "")
            | (filtered["patient_key"].astype(str) != "")
        ]
    if hide_negated:
        filtered = filtered[~filtered["negated"].astype(str).str.lower()
                            .isin({"true", "1"})]

    st.caption(f"{len(filtered):,} of {len(timed):,} events.")
    ordered = filtered.sort_values(["sort_key", "doc_id", "sent_id"])
    st.dataframe(
        ordered[["time_value", "event_type", "trigger_word", "agent", "patient",
                 "location", "time_expr", "time_method", "negated", "doc_id",
                 "sentence"]],
        hide_index=True, height=460,
    )

    st.subheader("How times were resolved")
    st.caption(
        "`dct_fallback` means the sentence carried no temporal expression, so "
        "the event was anchored to the document's publication date."
    )
    st.bar_chart(counted(timed, "time_method"), horizontal=True)


def tab_documents(tables: dict[str, pd.DataFrame]) -> None:
    timed, relations = tables["timed_events"], tables["relations"]
    entities = tables["entities"]
    if timed.empty:
        st.info("Run scripts/build_graph.py first.")
        return

    doc_ids = sorted(set(timed["doc_id"]) | set(relations.get("doc_id", [])))
    doc_id = st.selectbox("Document", doc_ids)
    if not doc_id:
        return

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Entities")
        if not entities.empty:
            local = entities[entities["doc_id"] == doc_id]
            st.dataframe(
                local.groupby(["label", "canonical"]).size()
                .reset_index(name="mentions").sort_values("mentions", ascending=False),
                hide_index=True, height=280,
            )
        st.subheader("Relations")
        local_relations = relations[relations["doc_id"] == doc_id]
        if "source" in local_relations.columns:
            local_relations = local_relations[local_relations["source"] == "rule"]
        st.dataframe(
            local_relations[["relation", "subj", "obj", "trigger", "confidence"]],
            hide_index=True, height=240,
        )
    with right:
        st.subheader("Events in narrative order")
        local_events = timed[timed["doc_id"] == doc_id].sort_values("sent_id")
        st.dataframe(
            local_events[["sent_id", "event_type", "trigger_word", "agent",
                          "patient", "time_expr", "time_value", "negated"]],
            hide_index=True, height=280,
        )
        st.subheader("Temporal ordering")
        links = tables["temporal_links"]
        local_links = links[links["doc_id"] == doc_id] if not links.empty else links
        if local_links.empty:
            st.caption("No temporal links for this document.")
        else:
            st.dataframe(
                local_links[["source_trigger", "relation", "target_trigger",
                             "basis", "confidence"]],
                hide_index=True, height=240,
            )

    st.subheader("Sentences with extractions")
    for _, row in timed[timed["doc_id"] == doc_id].sort_values("sent_id").iterrows():
        with st.expander(f"s{row['sent_id']} · {row['event_type']} · "
                         f"{row['trigger_word']}"):
            st.write(row["sentence"])


def tab_graph(tables: dict[str, pd.DataFrame], graph: EventGraph) -> None:
    st.caption(
        "Ellipses are entities, boxes are events. Dashed edges are temporal "
        "ordering; solid edges are relations and event participation."
    )
    entity_nodes = sorted(
        (node.node_id for node in graph.nodes.values() if node.node_type == "entity"),
        key=lambda key: -len(graph._adjacency.get(key, ())),
    )
    if not entity_nodes:
        st.info("The graph is empty - run scripts/build_graph.py.")
        return

    controls = st.columns([3, 1, 1, 2])
    focus = controls[0].selectbox("Focus entity", entity_nodes)
    hops = controls[1].slider("Hops", 1, 3, 1)
    max_nodes = controls[2].slider("Max nodes", 10, 120, 40, 10)
    edge_types = controls[3].multiselect(
        "Edge types",
        ["RELATION", "AGENT", "PATIENT", "LOCATION", "TEMPORAL"],
        default=["RELATION", "AGENT", "PATIENT"],
    )

    subgraph = graph.subgraph([focus], hops=hops, max_nodes=max_nodes)
    if edge_types:
        subgraph.edges = [e for e in subgraph.edges if e.edge_type in edge_types]
        connected = {e.source for e in subgraph.edges} | {
            e.target for e in subgraph.edges}
        connected.add(focus)
        subgraph.nodes = {k: v for k, v in subgraph.nodes.items() if k in connected}

    st.caption(f"{len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges.")
    if len(subgraph.nodes) <= 1:
        st.warning("Nothing connected at these settings - try more hops or "
                   "more edge types.")
        return

    dot = subgraph.to_dot(title=focus)
    st.graphviz_chart(dot, width="stretch")
    with st.expander("DOT source"):
        st.code(dot, language="dot")
    st.download_button("Download DOT", dot.encode("utf-8"),
                       f"{focus.replace(':', '_')}.dot", "text/vnd.graphviz")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    data_dir, source = resolve_data_dir()
    st.title("🛰️ Entropy")
    st.caption("Information extraction over 500 Wikinews space and astronomy "
               "articles · rule-based pipeline with a Hugging Face baseline")

    if source == "missing":
        st.error(
            "No extraction artefacts found. Run the pipeline first:\n\n"
            "```\npython scripts/preprocess.py\n"
            "python scripts/extract_entities.py\n"
            "python scripts/extract_relations.py\n"
            "python scripts/build_graph.py\n"
            "python scripts/export_artifacts.py\n```"
        )
        st.stop()

    tables = load_tables(str(data_dir))
    stats = load_stats(str(data_dir))

    with st.sidebar:
        st.header("Data")
        st.write(f"Loaded from **{source}**")
        st.code(str(data_dir), language=None)
        minimum = st.slider("Graph: min relation confidence", 0.0, 1.0, 0.0, 0.05)
        st.divider()
        st.caption(
            "Rule confidences are pattern-strength priors, not learned "
            "probabilities. A 0.4 score means the dependency-path fallback "
            "fired rather than a specific syntactic rule."
        )
        missing = [name for name, frame in tables.items() if frame.empty]
        if missing:
            st.warning("Missing: " + ", ".join(missing))

    graph = build_graph(str(data_dir), minimum)

    overview, entities, relations, events, documents, graph_tab = st.tabs(
        ["Overview", "Entities", "Relations", "Events", "Documents", "Graph"]
    )
    with overview:
        tab_overview(tables, stats)
    with entities:
        tab_entities(tables, graph)
    with relations:
        tab_relations(tables)
    with events:
        tab_events(tables)
    with documents:
        tab_documents(tables)
    with graph_tab:
        tab_graph(tables, graph)


if __name__ == "__main__":
    main()
