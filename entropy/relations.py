"""Stage 4 - relation extraction.

Primary extractor: dependency-arc rules run over the *cached* spaCy Docs
produced in Stage 2 and the mention rows produced in Stage 3.
Baseline extractor: a small Hugging Face zero-shot NLI model, used only for
comparison and only when explicitly requested (see ``HFRelationBaseline``).

HARD RULE: this module never calls ``nlp(text)``. Docs arrive already parsed
from ``entropy.nlp.iter_cached_docs``.

Public surface used by ``scripts/extract_relations.py`` and by ``events.py``:
    build_sentences(doc, mentions)  -> list[SentenceContext]
    extract_relations_from_sentence(ctx) -> list[Relation]
    summarise_relations(relations, hf_relations)
    write_relations_csv / write_relation_summary_csv
    subjects_of / objects_of / prep_objects / expand_conj / dep_path
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from spacy.tokens import Doc, Span, Token

from entropy import config
from entropy.entities import EntityMention

__all__ = [
    "Relation",
    "SentenceContext",
    "RELATION_TYPES",
    "RELATION_SIGNATURES",
    "build_sentences",
    "extract_relations_from_sentence",
    "summarise_relations",
    "write_relations_csv",
    "write_relation_summary_csv",
    "output_dir",
]

# Dependency helpers below are shared with entropy.events - keep them public.
__all__ += [
    "expand_conj",
    "is_passive",
    "subjects_of",
    "objects_of",
    "agents_of",
    "prep_objects",
    "possessors_of",
    "compounds_of",
    "appositions_of",
    "implicit_subjects",
    "governing_verb",
    "np_head",
    "dep_path",
    "resolve_mention",
    "trigger_text",
    "SPACECRAFT",
    "CELESTIAL",
    "PLACE_LABELS",
    "VERBISH",
    "RELATION_FIELDS",
    "HFRelationBaseline",
    "HF_DEFAULT_MODEL",
    "candidate_pairs",
]


# --------------------------------------------------------------------------
# config helpers
# --------------------------------------------------------------------------
# config.py currently defines both OUTPUT_DIR (line 26) and OUTPUTS_DIR
# (line 91) pointing at the same path. nlp.py papers over this with getattr;
# we do the same here so Stage 4 keeps working whichever name survives the
# consolidation flagged in OPEN ISSUES #3.
def output_dir() -> Path:
    """Resolve the outputs directory regardless of which config name exists."""
    value = getattr(config, "OUTPUT_DIR", None) or getattr(config, "OUTPUTS_DIR", None)
    path = Path(value) if value else Path(config.PROJECT_ROOT) / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


TEMPORAL_LABELS = frozenset(getattr(config, "TEMPORAL_LABELS", ("DATE", "TIME")))

SPACECRAFT = "SPACECRAFT"
CELESTIAL = "CELESTIAL"
PLACE_LABELS = frozenset({"FAC", "GPE", "LOC"})


# --------------------------------------------------------------------------
# relation inventory
# --------------------------------------------------------------------------
# (allowed subject labels, allowed object labels) - argument order matches the
# Stage 4 spec, e.g. LAUNCHED_BY(SPACECRAFT, ORG).
RELATION_SIGNATURES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "LAUNCHED_BY": (frozenset({SPACECRAFT}), frozenset({"ORG"})),
    "OPERATED_BY": (frozenset({SPACECRAFT}), frozenset({"ORG"})),
    "DESTINATION": (frozenset({SPACECRAFT}), frozenset({CELESTIAL})),
    "ORBITS": (frozenset({SPACECRAFT}), frozenset({CELESTIAL})),
    "LAUNCHED_FROM": (frozenset({SPACECRAFT}), PLACE_LABELS),
    "PART_OF": (frozenset({SPACECRAFT}), frozenset({SPACECRAFT, "ORG"})),
    "CREWED_BY": (frozenset({SPACECRAFT}), frozenset({"PERSON"})),
}
RELATION_TYPES: tuple[str, ...] = tuple(RELATION_SIGNATURES)


# --------------------------------------------------------------------------
# trigger lexicons (lemmas, lower-cased)
# --------------------------------------------------------------------------
LAUNCH_VERBS = frozenset({"launch", "lift", "blast", "loft", "boost", "orbit"})
LAUNCH_NOUNS = frozenset({"launch", "liftoff", "lift-off", "blastoff", "launching"})
OPERATE_VERBS = frozenset(
    {"operate", "run", "manage", "control", "command", "own", "oversee", "fly"}
)
OPERATE_NOUNS = frozenset({"operator", "control", "management"})
TRAVEL_VERBS = frozenset(
    {
        "travel", "head", "journey", "fly", "go", "send", "dispatch", "arrive",
        "reach", "approach", "return", "cruise", "voyage", "sail", "speed",
        "race", "land", "touch", "descend", "plunge", "crash", "slam", "impact",
    }
)
TRAVEL_NOUNS = frozenset(
    {"mission", "journey", "trip", "flight", "voyage", "route", "way", "course", "probe"}
)
DESTINATION_PREPS = frozenset({"to", "toward", "towards", "for", "at", "into", "onto", "on"})
ORBIT_VERBS = frozenset({"orbit", "circle", "encircle", "circumnavigate"})
ORBIT_NOUNS = frozenset({"orbit", "orbiter"})
ORBIT_PREPS = frozenset({"around", "round", "about", "of"})
SITE_PREPS = frozenset({"from", "at", "atop"})
PART_NOUNS = frozenset(
    {
        "part", "component", "module", "section", "segment", "element", "stage",
        "instrument", "payload", "portion", "half", "piece",
    }
)
CREW_VERBS = frozenset({"crew", "command", "pilot", "captain", "man", "helm"})
CREW_NOUNS = frozenset({"crew", "commander", "astronaut", "cosmonaut", "pilot", "member"})
ABOARD_PREPS = frozenset({"aboard", "onboard", "on", "in", "inside"})

# lemma sets used by the low-confidence fallback rule, keyed by relation
FALLBACK_TRIGGERS: dict[str, frozenset[str]] = {
    "LAUNCHED_BY": LAUNCH_VERBS | LAUNCH_NOUNS,
    "OPERATED_BY": OPERATE_VERBS | OPERATE_NOUNS,
    "DESTINATION": TRAVEL_VERBS | TRAVEL_NOUNS,
    "ORBITS": ORBIT_VERBS | ORBIT_NOUNS,
    "LAUNCHED_FROM": LAUNCH_VERBS | LAUNCH_NOUNS,
    "PART_OF": PART_NOUNS,
    "CREWED_BY": CREW_VERBS | CREW_NOUNS,
}

# Confidence is a rule-strength prior, not a learned probability. Tune here.
PATTERN_CONFIDENCE: dict[str, float] = {
    "passive_agent": 0.92,
    "active_svo": 0.88,
    "acl_agent": 0.85,
    "verb_prep": 0.82,
    "noun_of": 0.75,
    "noun_by": 0.80,
    "poss": 0.62,
    "np_prep": 0.60,
    "appos": 0.60,
    "path_trigger": 0.40,
}

SUBJ_DEPS = frozenset({"nsubj", "nsubjpass", "nsubj:pass", "csubj", "csubjpass"})
PASSIVE_SUBJ_DEPS = frozenset({"nsubjpass", "nsubj:pass"})
OBJ_DEPS = frozenset({"dobj", "obj", "attr", "dative", "oprd"})
# "dative" matters: en_core_web_sm parses "NASA sent Opportunity to Mars" with
# to/dative rather than to/prep, so a prep-only rule silently misses it.
PREP_DEPS = frozenset({"prep", "agent", "dative"})
NP_HEAD_DEPS = frozenset({"compound", "amod", "nmod"})
VERBISH = frozenset({"VERB", "AUX"})


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Relation:
    """One extracted relation instance (one row of outputs/relations.csv)."""

    doc_id: str
    sent_id: int
    relation: str
    subj: str            # surface form of the subject mention
    subj_canonical: str
    subj_label: str
    subj_key: str        # entity_key, e.g. "SPACECRAFT:Cassini"
    obj: str
    obj_canonical: str
    obj_label: str
    obj_key: str
    trigger: str         # word that licensed the arc
    pattern: str         # which rule fired
    confidence: float
    source: str          # "rule" | "hf"
    sentence: str

    @property
    def pair_key(self) -> tuple[str, str, str]:
        return (self.relation, self.subj_key, self.obj_key)

    @property
    def instance_key(self) -> tuple[str, int, str, str, str]:
        return (self.doc_id, self.sent_id, self.relation, self.subj_key, self.obj_key)


RELATION_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Relation))


# --------------------------------------------------------------------------
# sentence context
# --------------------------------------------------------------------------
class SentenceContext:
    """A sentence plus the Stage 3 mentions that fall inside it.

    ``mention_at(token)`` is the workhorse: rules collect candidate tokens by
    dependency shape, then this maps a token back to a typed mention. Slot
    assignment is therefore driven by the *entity label*, which keeps the rules
    short (e.g. "NASA's launch" vs "Discovery's launch" resolve themselves).
    """

    __slots__ = ("doc_id", "sent_id", "sent", "mentions", "_by_token", "_span")

    def __init__(
        self,
        doc_id: str,
        sent_id: int,
        sent: Span,
        mentions: Sequence[EntityMention],
        spans: dict[int, Span],
    ) -> None:
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.sent = sent
        self.mentions = list(mentions)
        self._span = spans
        self._by_token: dict[int, EntityMention] = {}
        for mention in self.mentions:
            span = spans.get(id(mention))
            if span is None:
                continue
            for token in span:
                # first mention wins; gazetteer patterns are inserted before
                # ner with overwrite_ents=False so overlaps are rare
                self._by_token.setdefault(token.i, mention)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<SentenceContext {self.doc_id}#{self.sent_id} {len(self.mentions)} mentions>"

    @property
    def tokens(self) -> Span:
        return self.sent

    @property
    def text(self) -> str:
        return self.sent.text.strip()

    def mention_at(self, token: Token | None) -> EntityMention | None:
        if token is None:
            return None
        return self._by_token.get(token.i)

    def span_of(self, mention: EntityMention) -> Span | None:
        return self._span.get(id(mention))

    def head_token(self, mention: EntityMention) -> Token | None:
        span = self._span.get(id(mention))
        return None if span is None else span.root

    def mentions_with_labels(self, labels: Iterable[str]) -> list[EntityMention]:
        wanted = set(labels)
        return [m for m in self.mentions if m.label in wanted]

    def temporal_mentions(self) -> list[EntityMention]:
        return [
            m
            for m in self.mentions
            if getattr(m, "is_temporal", False) or m.label in TEMPORAL_LABELS
        ]


def _span_for(doc: Doc, mention: EntityMention) -> Span | None:
    """Resolve a mention to a Span using character offsets (verified exact)."""
    span = doc.char_span(mention.start_char, mention.end_char, label=mention.label)
    if span is None:
        span = doc.char_span(
            mention.start_char, mention.end_char, label=mention.label,
            alignment_mode="expand",
        )
    if span is not None:
        return span
    # last-ditch fallback: token indices, tolerating either end convention
    start = int(mention.start_token)
    end = int(mention.end_token)
    if end <= start:
        end = start + 1
    if 0 <= start < len(doc) and end <= len(doc):
        return doc[start:end]
    return None


def build_sentences(
    doc: Doc,
    mentions: Sequence[EntityMention],
    doc_id: str | None = None,
) -> list[SentenceContext]:
    """Group a cached Doc into sentences carrying their Stage 3 mentions.

    ``sent_id`` is taken from the mentions themselves wherever possible so the
    ids stay byte-identical to entities.csv; sentences without mentions fall
    back to their positional index. The custom "entropy_paragraph_breaks"
    component means ``doc.sents`` order is the same order Stage 3 numbered.
    """
    if doc_id is None:
        doc_id = _doc_id_of(doc, mentions)

    spans: dict[int, Span] = {}
    for mention in mentions:
        span = _span_for(doc, mention)
        if span is not None:
            spans[id(mention)] = span

    sents = list(doc.sents)
    start_to_index = {sent.start: i for i, sent in enumerate(sents)}

    # sentence index -> sent_id, learned from the mentions
    index_to_sent_id: dict[int, int] = {}
    grouped: dict[int, list[EntityMention]] = defaultdict(list)
    for mention in mentions:
        span = spans.get(id(mention))
        if span is None:
            continue
        index = start_to_index.get(span.sent.start)
        if index is None:
            continue
        grouped[index].append(mention)
        index_to_sent_id.setdefault(index, int(mention.sent_id))

    contexts: list[SentenceContext] = []
    for index, sent in enumerate(sents):
        sent_id = index_to_sent_id.get(index, index)
        contexts.append(
            SentenceContext(doc_id, sent_id, sent, grouped.get(index, []), spans)
        )
    return contexts


def _doc_id_of(doc: Doc, mentions: Sequence[EntityMention]) -> str:
    """Best-effort doc_id: the Doc's own metadata, else the mentions."""
    for getter in (
        lambda: doc.user_data.get("doc_id"),
        lambda: getattr(doc._, "doc_id", None),
    ):
        try:
            value = getter()
        except (AttributeError, KeyError):
            value = None
        if value:
            return str(value)
    if mentions:
        return str(mentions[0].doc_id)
    return ""


# --------------------------------------------------------------------------
# dependency helpers (shared with events.py)
# --------------------------------------------------------------------------
def expand_conj(tokens: Iterable[Token], depth: int = 2) -> list[Token]:
    """Include coordinated siblings: "launched by NASA and ESA"."""
    out: list[Token] = []
    seen: set[int] = set()
    frontier = list(tokens)
    for _ in range(depth + 1):
        nxt: list[Token] = []
        for token in frontier:
            if token.i in seen:
                continue
            seen.add(token.i)
            out.append(token)
            nxt.extend(c for c in token.children if c.dep_ == "conj")
        if not nxt:
            break
        frontier = nxt
    return out


def is_passive(verb: Token) -> bool:
    """True for full passives *and* reduced relatives.

    "Cassini, launched by NASA in 1997, orbits Saturn" has no auxpass and no
    nsubjpass, so an auxpass-only test misses one of the most common shapes in
    news prose. A VBN under acl/advcl is a reduced passive relative unless it
    carries its own have/be auxiliary ("having launched from Florida").
    """
    node = verb
    for _ in range(3):  # follow conj chains: "was built and launched by NASA"
        if any(c.dep_ in PASSIVE_SUBJ_DEPS or c.dep_ == "auxpass" for c in node.children):
            return True
        if node.dep_ != "conj" or node.head is node:
            break
        node = node.head
    if verb.tag_ == "VBN" and verb.dep_ in {"acl", "advcl"}:
        return not any(
            c.dep_ == "aux" and c.lemma_.lower() in {"have", "be"} for c in verb.children
        )
    return False


def subjects_of(verb: Token, climb: int = 3) -> list[Token]:
    """Subjects of a verb, inherited through conj/xcomp chains if needed."""
    found = [c for c in verb.children if c.dep_ in SUBJ_DEPS]
    node = verb
    steps = 0
    while not found and steps < climb and node.dep_ in {"conj", "xcomp", "advcl", "ccomp"}:
        if node.head is node:
            break
        node = node.head
        found = [c for c in node.children if c.dep_ in SUBJ_DEPS]
        steps += 1
    return expand_conj(found)


def objects_of(verb: Token) -> list[Token]:
    return expand_conj([c for c in verb.children if c.dep_ in OBJ_DEPS])


def agents_of(verb: Token) -> list[Token]:
    """Objects of a by-phrase attached to a passive verb."""
    out: list[Token] = []
    for child in verb.children:
        if child.dep_ == "agent" or (child.dep_ == "prep" and child.lower_ == "by"):
            out.extend(c for c in child.children if c.dep_ == "pobj")
    return expand_conj(out)


def prep_objects(head: Token, preps: Iterable[str]) -> list[Token]:
    """Objects of the given prepositions hanging off ``head``."""
    wanted = {p.lower() for p in preps}
    out: list[Token] = []
    for child in head.children:
        if child.dep_ not in PREP_DEPS:
            continue
        if child.lower_ not in wanted and child.lemma_.lower() not in wanted:
            continue
        for grand in child.children:
            if grand.dep_ in {"pobj", "pcomp"}:
                out.append(grand)
            # "on board the ISS" -> board(pobj) -> ISS(det/compound-ish)
            elif grand.dep_ == "prep":
                out.extend(g for g in grand.children if g.dep_ == "pobj")
    return expand_conj(out)


def possessors_of(noun: Token) -> list[Token]:
    return [c for c in noun.children if c.dep_ == "poss"]


def compounds_of(noun: Token) -> list[Token]:
    return [c for c in noun.children if c.dep_ in {"compound", "nmod", "amod"}]


def appositions_of(noun: Token) -> list[Token]:
    return expand_conj([c for c in noun.children if c.dep_ == "appos"])


def implicit_subjects(verb: Token) -> list[Token]:
    """Reduced relatives: "Cassini, launched by NASA" -> Cassini."""
    if verb.dep_ in {"acl", "relcl", "advcl"} and verb.head is not verb:
        return [verb.head]
    return []


def np_head(token: Token, max_up: int = 3) -> Token:
    """Climb from a modifier to the head of its noun phrase.

    Needed because "NASA's Cassini spacecraft" hangs poss/NASA off *spacecraft*
    while the SPACECRAFT mention's head token is *Cassini* (a compound).
    """
    node = token
    for _ in range(max_up):
        if node.dep_ not in NP_HEAD_DEPS or node.head is node:
            break
        node = node.head
    return node


def governing_verb(token: Token, max_up: int = 4) -> Token | None:
    """Climb to the nearest governing VERB/AUX (for "in orbit around X")."""
    node = token
    for _ in range(max_up):
        if node.head is node:
            return None
        node = node.head
        if node.pos_ in VERBISH:
            return node
    return None


def dep_path(a: Token, b: Token, max_len: int = 6) -> list[Token] | None:
    """Shortest undirected dependency path between two tokens, or None."""
    up_a: dict[int, int] = {}
    node = a
    for depth in range(max_len + 1):
        up_a[node.i] = depth
        if node.head is node:
            break
        node = node.head
    chain_b: list[Token] = []
    node = b
    for _ in range(max_len + 1):
        chain_b.append(node)
        if node.i in up_a:
            if up_a[node.i] + len(chain_b) - 1 > max_len:
                return None
            path: list[Token] = []
            walker = a
            while walker.i != node.i:
                path.append(walker)
                walker = walker.head
            path.append(node)
            path.extend(reversed(chain_b[:-1]))
            return path
        if node.head is node:
            break
        node = node.head
    return None


def trigger_text(token: Token) -> str:
    """Trigger surface form, including a verb particle if present."""
    particles = [c.text for c in token.children if c.dep_ == "prt"]
    return " ".join([token.text] + particles)


# --------------------------------------------------------------------------
# rule engine
# --------------------------------------------------------------------------
def resolve_mention(
    ctx: SentenceContext, token: Token, labels: Iterable[str] | None = None
) -> EntityMention | None:
    """Map a token to a typed mention, looking inside its noun phrase.

    A dependency arc often lands on the NP head rather than on the entity:
    "part of the Cassini mission" gives pobj=*mission* with the SPACECRAFT
    mention sitting on the *Cassini* compound. Descending one level recovers it.
    """
    wanted = set(labels) if labels else None
    direct = ctx.mention_at(token)
    if direct is not None and (wanted is None or direct.label in wanted):
        return direct
    for child in token.children:
        if child.dep_ not in NP_HEAD_DEPS:
            continue
        nested = ctx.mention_at(child)
        if nested is not None and (wanted is None or nested.label in wanted):
            return nested
    return direct


def _emit(
    ctx: SentenceContext,
    out: list[Relation],
    relation: str,
    subj_tokens: Iterable[Token],
    obj_tokens: Iterable[Token],
    trigger: str,
    pattern: str,
) -> None:
    subj_labels, obj_labels = RELATION_SIGNATURES[relation]
    confidence = PATTERN_CONFIDENCE.get(pattern, 0.5)
    subj_list = list(subj_tokens)
    obj_list = list(obj_tokens)
    if not subj_list or not obj_list:
        return
    for stok in subj_list:
        smention = resolve_mention(ctx, stok, subj_labels)
        if smention is None or smention.label not in subj_labels:
            continue
        for otok in obj_list:
            omention = resolve_mention(ctx, otok, obj_labels)
            if omention is None or omention.label not in obj_labels:
                continue
            if smention is omention or smention.entity_key == omention.entity_key:
                continue
            out.append(
                Relation(
                    doc_id=ctx.doc_id,
                    sent_id=ctx.sent_id,
                    relation=relation,
                    subj=smention.mention,
                    subj_canonical=smention.canonical,
                    subj_label=smention.label,
                    subj_key=smention.entity_key,
                    obj=omention.mention,
                    obj_canonical=omention.canonical,
                    obj_label=omention.label,
                    obj_key=omention.entity_key,
                    trigger=trigger,
                    pattern=pattern,
                    confidence=confidence,
                    source="rule",
                    sentence=ctx.text,
                )
            )


def _rule_launch(ctx: SentenceContext, out: list[Relation]) -> None:
    """LAUNCHED_BY and LAUNCHED_FROM."""
    for token in ctx.tokens:
        lemma = token.lemma_.lower()
        particles = {c.lower_ for c in token.children if c.dep_ == "prt"}
        verbal = token.pos_ in VERBISH and (
            lemma == "launch"
            or (lemma in {"lift", "blast"} and particles & {"off"})
            or lemma in {"loft", "boost"}
        )
        nominal = token.pos_ in {"NOUN", "PROPN"} and lemma in LAUNCH_NOUNS
        if not (verbal or nominal):
            continue

        trigger = trigger_text(token)
        craft: list[Token] = []
        orgs: list[Token] = []
        sites: list[Token] = prep_objects(token, SITE_PREPS)

        if verbal:
            if is_passive(token):
                craft = subjects_of(token)
                pattern = "passive_agent"
                if not craft:
                    craft = implicit_subjects(token)
                    pattern = "acl_agent"
                orgs = agents_of(token)
            else:
                craft = objects_of(token)
                orgs = subjects_of(token)
                pattern = "active_svo"
                # "NASA launched Discovery" but also "Discovery launched
                # from Cape Canaveral" (intransitive) - then the subject is
                # the craft, not the launcher.
                if not craft:
                    subject_craft = [
                        t for t in orgs
                        if (m := ctx.mention_at(t)) is not None and m.label == SPACECRAFT
                    ]
                    if subject_craft:
                        craft, orgs = subject_craft, []
            if orgs:
                _emit(ctx, out, "LAUNCHED_BY", craft, orgs, trigger, pattern)
        else:
            of_objs = prep_objects(token, {"of"})
            by_objs = prep_objects(token, {"by"})
            owners = possessors_of(token) + compounds_of(token)
            craft = list(of_objs)
            orgs = list(by_objs)
            for tok in owners:
                mention = ctx.mention_at(tok)
                if mention is None:
                    continue
                if mention.label == SPACECRAFT:
                    craft.append(tok)
                elif mention.label == "ORG":
                    orgs.append(tok)
            if craft and by_objs:
                _emit(ctx, out, "LAUNCHED_BY", craft, by_objs, trigger, "noun_by")
            if craft and orgs and not by_objs:
                _emit(ctx, out, "LAUNCHED_BY", craft, orgs, trigger, "poss")
            # "the launch of Cassini" also supplies the craft for the site rule
            if not craft:
                verb = governing_verb(token)
                if verb is not None:
                    craft = subjects_of(verb)

        if craft and sites:
            _emit(ctx, out, "LAUNCHED_FROM", craft, sites, trigger, "verb_prep")


def _rule_operate(ctx: SentenceContext, out: list[Relation]) -> None:
    """OPERATED_BY - verbal triggers plus the high-yield possessive pattern."""
    for token in ctx.tokens:
        lemma = token.lemma_.lower()
        if token.pos_ in VERBISH and lemma in OPERATE_VERBS:
            trigger = trigger_text(token)
            if is_passive(token):
                craft = subjects_of(token)
                pattern = "passive_agent"
                if not craft:
                    craft = implicit_subjects(token)
                    pattern = "acl_agent"
                orgs = agents_of(token)
                _emit(ctx, out, "OPERATED_BY", craft, orgs, trigger, pattern)
            else:
                _emit(
                    ctx, out, "OPERATED_BY",
                    objects_of(token), subjects_of(token), trigger, "active_svo",
                )
        elif token.pos_ in {"NOUN", "PROPN"} and lemma in OPERATE_NOUNS:
            _emit(
                ctx, out, "OPERATED_BY",
                prep_objects(token, {"of", "for"}), possessors_of(token),
                trigger_text(token), "noun_of",
            )

    # "NASA's Cassini spacecraft" / "ESA's Rosetta". The poss arc hangs off the
    # NP head ("spacecraft"), not off the mention token, so climb first.
    for mention in ctx.mentions_with_labels({SPACECRAFT}):
        head = ctx.head_token(mention)
        if head is None:
            continue
        for owner in possessors_of(head) + possessors_of(np_head(head)):
            _emit(ctx, out, "OPERATED_BY", [head], [owner], owner.text, "poss")


def _rule_destination(ctx: SentenceContext, out: list[Relation]) -> None:
    """DESTINATION - motion verbs with a goal PP, plus bare "mission to X"."""
    for token in ctx.tokens:
        lemma = token.lemma_.lower()
        if token.pos_ in VERBISH and lemma in TRAVEL_VERBS:
            trigger = trigger_text(token)
            goals = prep_objects(token, DESTINATION_PREPS)
            if lemma in {"reach", "approach", "orbit"}:
                goals = goals + objects_of(token)
            if not goals:
                continue
            if lemma in {"send", "dispatch"}:
                craft = objects_of(token) or subjects_of(token)
            elif is_passive(token):
                craft = subjects_of(token) or implicit_subjects(token)
            else:
                craft = subjects_of(token) or implicit_subjects(token) or objects_of(token)
            _emit(ctx, out, "DESTINATION", craft, goals, trigger, "verb_prep")

        elif token.pos_ in {"NOUN", "PROPN"} and lemma in TRAVEL_NOUNS:
            goals = prep_objects(token, {"to", "toward", "towards"})
            if not goals:
                continue
            craft = possessors_of(token) + compounds_of(token) + appositions_of(token)
            if not craft:
                verb = governing_verb(token)
                if verb is not None:
                    craft = subjects_of(verb)
            _emit(ctx, out, "DESTINATION", craft, goals, trigger_text(token), "noun_of")

    # "Cassini to Saturn" - PP hanging straight off the spacecraft NP
    for mention in ctx.mentions_with_labels({SPACECRAFT}):
        head = ctx.head_token(mention)
        if head is None:
            continue
        goals = prep_objects(head, {"to", "toward", "towards"})
        if goals:
            _emit(ctx, out, "DESTINATION", [head], goals, "to", "np_prep")


def _rule_orbit(ctx: SentenceContext, out: list[Relation]) -> None:
    """ORBITS - verbal "orbits X" and nominal "in orbit around X"."""
    for token in ctx.tokens:
        lemma = token.lemma_.lower()
        if token.pos_ in VERBISH and lemma in ORBIT_VERBS:
            trigger = trigger_text(token)
            bodies = objects_of(token) + prep_objects(token, ORBIT_PREPS)
            if is_passive(token):
                craft = agents_of(token)
                bodies = bodies + subjects_of(token)
            else:
                craft = subjects_of(token) or implicit_subjects(token)
            _emit(ctx, out, "ORBITS", craft, bodies, trigger, "active_svo")

        elif token.pos_ in {"NOUN", "PROPN"} and lemma in ORBIT_NOUNS:
            verb = governing_verb(token)
            bodies = prep_objects(token, ORBIT_PREPS) + compounds_of(token)
            if not bodies and verb is not None:
                # "Mars Express is in orbit around Mars" parses around/Mars onto
                # the copula, not onto "orbit".
                bodies = prep_objects(verb, ORBIT_PREPS - {"of"})
            if not bodies:
                continue
            craft = possessors_of(token) + compounds_of(token)
            if not craft and verb is not None:
                craft = subjects_of(verb) or implicit_subjects(verb)
            _emit(ctx, out, "ORBITS", craft, bodies, trigger_text(token), "np_prep")


def _rule_part_of(ctx: SentenceContext, out: list[Relation]) -> None:
    """PART_OF - "part of X", "module of the ISS", "Huygens of Cassini"."""
    for token in ctx.tokens:
        lemma = token.lemma_.lower()
        if token.pos_ not in {"NOUN", "PROPN"} or lemma not in PART_NOUNS:
            continue
        wholes = prep_objects(token, {"of", "on", "in"})
        if not wholes:
            continue
        parts = possessors_of(token) + compounds_of(token) + appositions_of(token)
        if not parts:
            # "Huygens is part of Cassini" / "Huygens, part of Cassini"
            head = token.head
            if token.dep_ in {"attr", "appos", "acomp"} and head is not token:
                if token.dep_ == "attr":
                    parts = subjects_of(head)
                else:
                    parts = [head]
        _emit(ctx, out, "PART_OF", parts, wholes, trigger_text(token), "noun_of")

    # direct "the Huygens probe of the Cassini mission"
    for mention in ctx.mentions_with_labels({SPACECRAFT}):
        head = ctx.head_token(mention)
        if head is None:
            continue
        wholes = prep_objects(head, {"of"})
        if wholes:
            _emit(ctx, out, "PART_OF", [head], wholes, "of", "np_prep")


def _rule_crewed_by(ctx: SentenceContext, out: list[Relation]) -> None:
    """CREWED_BY - note the emitted argument order is (SPACECRAFT, PERSON)."""
    for token in ctx.tokens:
        lemma = token.lemma_.lower()
        if token.pos_ in VERBISH and lemma in CREW_VERBS:
            trigger = trigger_text(token)
            if is_passive(token):
                craft = subjects_of(token)
                pattern = "passive_agent"
                if not craft:
                    craft = implicit_subjects(token)
                    pattern = "acl_agent"
                people = agents_of(token)
                _emit(ctx, out, "CREWED_BY", craft, people, trigger, pattern)
            else:
                _emit(
                    ctx, out, "CREWED_BY",
                    objects_of(token), subjects_of(token), trigger, "active_svo",
                )
        elif token.pos_ in {"NOUN", "PROPN"} and lemma in CREW_NOUNS:
            craft = prep_objects(token, {"of", "aboard", "on", "onboard"})
            craft += possessors_of(token) + compounds_of(token)
            if not craft:
                continue
            people = appositions_of(token) + possessors_of(token)
            verb = governing_verb(token)
            if verb is not None and token.dep_ in SUBJ_DEPS:
                people = people + objects_of(verb)
            _emit(ctx, out, "CREWED_BY", craft, people, trigger_text(token), "noun_of")

    # "astronaut Steve Robinson aboard Discovery"
    for mention in ctx.mentions_with_labels({"PERSON"}):
        head = ctx.head_token(mention)
        if head is None:
            continue
        craft = prep_objects(head, ABOARD_PREPS)
        if craft:
            _emit(ctx, out, "CREWED_BY", craft, [head], "aboard", "np_prep")


def _rule_path_fallback(ctx: SentenceContext, out: list[Relation]) -> None:
    """Low-confidence catch-all: correct label pair + trigger on the dep path."""
    if len(ctx.mentions) < 2:
        return
    already = {r.instance_key for r in out}
    heads: list[tuple[EntityMention, Token]] = []
    for mention in ctx.mentions:
        head = ctx.head_token(mention)
        if head is not None:
            heads.append((mention, head))

    for relation, (subj_labels, obj_labels) in RELATION_SIGNATURES.items():
        triggers = FALLBACK_TRIGGERS[relation]
        for smention, stok in heads:
            if smention.label not in subj_labels:
                continue
            for omention, otok in heads:
                if omention.label not in obj_labels:
                    continue
                if smention is omention or smention.entity_key == omention.entity_key:
                    continue
                key = (ctx.doc_id, ctx.sent_id, relation, smention.entity_key,
                       omention.entity_key)
                if key in already:
                    continue
                path = dep_path(stok, otok)
                if not path:
                    continue
                hit = next(
                    (t for t in path if t.lemma_.lower() in triggers and t.i not in
                     {stok.i, otok.i}),
                    None,
                )
                if hit is None:
                    continue
                already.add(key)
                _emit(ctx, out, relation, [stok], [otok], trigger_text(hit),
                      "path_trigger")


RULES = (
    _rule_launch,
    _rule_operate,
    _rule_destination,
    _rule_orbit,
    _rule_part_of,
    _rule_crewed_by,
)


def extract_relations_from_sentence(
    ctx: SentenceContext, use_fallback: bool = True
) -> list[Relation]:
    """Run every rule over one sentence and de-duplicate by instance."""
    out: list[Relation] = []
    for rule in RULES:
        rule(ctx, out)
    if use_fallback:
        _rule_path_fallback(ctx, out)

    best: dict[tuple, Relation] = {}
    for relation in out:
        current = best.get(relation.instance_key)
        if current is None or relation.confidence > current.confidence:
            best[relation.instance_key] = relation
    return sorted(best.values(), key=lambda r: (r.relation, -r.confidence, r.subj))


def extract_relations(
    contexts: Iterable[SentenceContext], use_fallback: bool = True
) -> Iterator[Relation]:
    for ctx in contexts:
        if len(ctx.mentions) < 2:
            continue
        yield from extract_relations_from_sentence(ctx, use_fallback=use_fallback)


# --------------------------------------------------------------------------
# Hugging Face comparison baseline
# --------------------------------------------------------------------------
# Zero-shot NLI: each candidate pair becomes a set of natural-language
# hypotheses and the model picks the best one. This is the only formulation
# that fits the corpus (no labelled training data) *and* the RAM budget.
# distilbert-base-uncased-mnli is ~265 MB fp32, well inside the ~600 MB cap.
HF_DEFAULT_MODEL = "typeform/distilbert-base-uncased-mnli"
NO_RELATION = "NO_RELATION"

HYPOTHESIS_TEMPLATES: dict[str, str] = {
    "LAUNCHED_BY": "{subj} was launched by {obj}.",
    "OPERATED_BY": "{subj} is operated by {obj}.",
    "DESTINATION": "{subj} is travelling to {obj}.",
    "ORBITS": "{subj} orbits {obj}.",
    "LAUNCHED_FROM": "{subj} was launched from {obj}.",
    "PART_OF": "{subj} is part of {obj}.",
    "CREWED_BY": "{subj} is crewed by {obj}.",
    NO_RELATION: "{subj} and {obj} are not related.",
}


def candidate_pairs(
    ctx: SentenceContext, max_pairs: int = 12
) -> list[tuple[EntityMention, EntityMention, list[str]]]:
    """Type-compatible mention pairs in a sentence, with their candidate labels."""
    pairs: list[tuple[EntityMention, EntityMention, list[str]]] = []
    mentions = ctx.mentions
    for smention in mentions:
        for omention in mentions:
            if smention is omention or smention.entity_key == omention.entity_key:
                continue
            allowed = [
                rel
                for rel, (subj_labels, obj_labels) in RELATION_SIGNATURES.items()
                if smention.label in subj_labels and omention.label in obj_labels
            ]
            if allowed:
                pairs.append((smention, omention, allowed))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


class HFRelationBaseline:
    """Zero-shot NLI baseline used purely for comparison against the rules.

    Deliberately lazy: ``transformers`` and ``torch`` are imported inside
    ``_ensure_pipeline`` so neither the rule path nor the Stage 5 Streamlit app
    ever pays for them. The app must read outputs/relations.csv, never load
    this class - Community Cloud has ~1 GB RAM total.
    """

    def __init__(
        self,
        model_name: str = HF_DEFAULT_MODEL,
        threshold: float = 0.5,
        max_pairs_per_sentence: int = 12,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.max_pairs_per_sentence = max_pairs_per_sentence
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline  # local import, see docstring

            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1,  # CPU; Windows dev boxes and CI both lack CUDA here
            )
        return self._pipeline

    def predict_sentence(self, ctx: SentenceContext) -> list[Relation]:
        pairs = candidate_pairs(ctx, self.max_pairs_per_sentence)
        if not pairs:
            return []
        classifier = self._ensure_pipeline()
        premise = ctx.text
        out: list[Relation] = []
        for smention, omention, allowed in pairs:
            labels = allowed + [NO_RELATION]
            hypotheses = [
                HYPOTHESIS_TEMPLATES[label].format(
                    subj=smention.canonical, obj=omention.canonical
                )
                for label in labels
            ]
            result = classifier(
                premise,
                candidate_labels=hypotheses,
                hypothesis_template="{}",  # hypotheses are already complete
                multi_label=False,
            )
            top_hypothesis = result["labels"][0]
            score = float(result["scores"][0])
            label = labels[hypotheses.index(top_hypothesis)]
            if label == NO_RELATION or score < self.threshold:
                continue
            out.append(
                Relation(
                    doc_id=ctx.doc_id,
                    sent_id=ctx.sent_id,
                    relation=label,
                    subj=smention.mention,
                    subj_canonical=smention.canonical,
                    subj_label=smention.label,
                    subj_key=smention.entity_key,
                    obj=omention.mention,
                    obj_canonical=omention.canonical,
                    obj_label=omention.label,
                    obj_key=omention.entity_key,
                    trigger="",
                    pattern="zero_shot_nli",
                    confidence=round(score, 4),
                    source="hf",
                    sentence=premise,
                )
            )
        return out

    def predict(self, contexts: Iterable[SentenceContext]) -> Iterator[Relation]:
        for ctx in contexts:
            if len(ctx.mentions) < 2:
                continue
            yield from self.predict_sentence(ctx)


# --------------------------------------------------------------------------
# summary + IO
# --------------------------------------------------------------------------
def summarise_relations(
    rule_relations: Sequence[Relation],
    hf_relations: Sequence[Relation] = (),
    top_n: int = 5,
) -> list[dict]:
    """Per-relation-type counts plus rule/HF agreement (outputs/relation_summary.csv)."""
    rule_by_type: dict[str, list[Relation]] = defaultdict(list)
    for relation in rule_relations:
        rule_by_type[relation.relation].append(relation)
    hf_by_type: dict[str, list[Relation]] = defaultdict(list)
    for relation in hf_relations:
        hf_by_type[relation.relation].append(relation)

    rule_instances = {r.instance_key for r in rule_relations}
    hf_instances = {r.instance_key for r in hf_relations}

    rows: list[dict] = []
    for relation_type in RELATION_TYPES:
        rules = rule_by_type.get(relation_type, [])
        hfs = hf_by_type.get(relation_type, [])
        rule_keys = {r.instance_key for r in rules}
        hf_keys = {r.instance_key for r in hfs}
        agreed = rule_keys & hf_keys
        union = rule_keys | hf_keys
        pair_counts = Counter(r.pair_key for r in rules)
        top_pairs = "; ".join(
            f"{subj.split(':', 1)[-1]} -> {obj.split(':', 1)[-1]} ({n})"
            for (_, subj, obj), n in pair_counts.most_common(top_n)
        )
        rows.append(
            {
                "relation": relation_type,
                "rule_count": len(rules),
                "rule_distinct_pairs": len(pair_counts),
                "rule_docs": len({r.doc_id for r in rules}),
                "rule_mean_confidence": round(
                    sum(r.confidence for r in rules) / len(rules), 3
                ) if rules else 0.0,
                "rule_high_conf_count": sum(1 for r in rules if r.confidence >= 0.8),
                "hf_count": len(hfs),
                "hf_distinct_pairs": len({r.pair_key for r in hfs}),
                "agreed_count": len(agreed),
                "rule_only_count": len(rule_keys - hf_keys),
                "hf_only_count": len(hf_keys - rule_keys),
                "jaccard": round(len(agreed) / len(union), 3) if union else 0.0,
                "top_pairs": top_pairs,
            }
        )

    total_union = rule_instances | hf_instances
    rows.append(
        {
            "relation": "ALL",
            "rule_count": len(rule_relations),
            "rule_distinct_pairs": len({r.pair_key for r in rule_relations}),
            "rule_docs": len({r.doc_id for r in rule_relations}),
            "rule_mean_confidence": round(
                sum(r.confidence for r in rule_relations) / len(rule_relations), 3
            ) if rule_relations else 0.0,
            "rule_high_conf_count": sum(1 for r in rule_relations if r.confidence >= 0.8),
            "hf_count": len(hf_relations),
            "hf_distinct_pairs": len({r.pair_key for r in hf_relations}),
            "agreed_count": len(rule_instances & hf_instances),
            "rule_only_count": len(rule_instances - hf_instances),
            "hf_only_count": len(hf_instances - rule_instances),
            "jaccard": round(
                len(rule_instances & hf_instances) / len(total_union), 3
            ) if total_union else 0.0,
            "top_pairs": "",
        }
    )
    return rows


def write_relations_csv(relations: Sequence[Relation], path: Path | None = None) -> Path:
    path = path or (output_dir() / "relations.csv")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RELATION_FIELDS)
        writer.writeheader()
        for relation in relations:
            writer.writerow(asdict(relation))
    return path


def write_relation_summary_csv(rows: Sequence[dict], path: Path | None = None) -> Path:
    path = path or (output_dir() / "relation_summary.csv")
    if not rows:
        return path
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
