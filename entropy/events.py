"""Stage 4 - event extraction.

Trigger-based detection over the cached Stage 2 Docs, with dependency slot
filling for agent / patient / time / location. Shares the dependency helpers
with ``entropy.relations`` so both extractors treat coordination, passives and
reduced relatives the same way.

HARD RULE: never calls ``nlp(text)``. Docs arrive already parsed.

Event types: launch, landing, docking, flyby, discovery, failure, delay.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from spacy.tokens import Token

from entropy.entities import EntityMention
from entropy.relations import (
    SPACECRAFT,
    CELESTIAL,
    PLACE_LABELS,
    VERBISH,
    is_passive,
    SentenceContext,
    agents_of,
    appositions_of,
    compounds_of,
    expand_conj,
    implicit_subjects,
    objects_of,
    output_dir,
    possessors_of,
    prep_objects,
    subjects_of,
    trigger_text,
)

__all__ = [
    "Event",
    "Lexicon",
    "EVENT_TYPES",
    "EVENT_LEXICON",
    "extract_events_from_sentence",
    "extract_events",
    "summarise_events",
    "write_events_csv",
    "write_event_summary_csv",
]


# --------------------------------------------------------------------------
# trigger lexicon
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Lexicon:
    """Trigger words for one event type.

    ``particle_verbs`` maps a verb lemma to the particles that must accompany
    it, so "lift off" fires but "lift the payload" does not.
    """

    verbs: frozenset[str] = frozenset()
    nouns: frozenset[str] = frozenset()
    adjectives: frozenset[str] = frozenset()
    particle_verbs: tuple[tuple[str, frozenset[str]], ...] = ()
    # prepositions whose object is the event's location, if any
    location_preps: frozenset[str] = frozenset({"from", "at", "in", "on", "near"})
    # prepositions whose object is the patient of an otherwise intransitive verb
    patient_preps: frozenset[str] = frozenset({"with"})
    # entity labels to prefer when several fillers compete for the patient slot
    patient_prefer: frozenset[str] = frozenset()
    # allow the particle to attach as prep rather than prt ("fly by Titan")
    prep_particles: bool = False

    def particles_for(self, lemma: str) -> frozenset[str] | None:
        for verb_lemma, particles in self.particle_verbs:
            if verb_lemma == lemma:
                return particles
        return None


# Order matters: the first type whose lexicon matches a token wins, so the
# more specific types are listed first ("crash" -> failure, not landing).
EVENT_LEXICON: dict[str, Lexicon] = {
    "docking": Lexicon(
        verbs=frozenset({"dock", "undock", "berth", "rendezvous"}),
        nouns=frozenset({"docking", "undocking", "berthing", "rendezvous", "linkup"}),
        particle_verbs=(("link", frozenset({"up"})),),
        location_preps=frozenset({"at"}),
        patient_preps=frozenset({"with", "to"}),
        patient_prefer=frozenset({SPACECRAFT}),
    ),
    "delay": Lexicon(
        verbs=frozenset({"delay", "postpone", "scrub", "reschedule", "defer", "slip"}),
        nouns=frozenset({"delay", "postponement", "scrub", "slippage"}),
        adjectives=frozenset({"delayed", "postponed"}),
        particle_verbs=(("push", frozenset({"back"})), ("put", frozenset({"off"}))),
        patient_prefer=frozenset({SPACECRAFT, "ORG"}),
    ),
    "failure": Lexicon(
        verbs=frozenset(
            {
                "fail", "malfunction", "abort", "explode", "crash", "disintegrate",
                "destroy", "cancel", "lose", "misfire",
            }
        ),
        nouns=frozenset(
            {
                "failure", "malfunction", "anomaly", "explosion", "crash", "glitch",
                "breakdown", "accident", "disaster", "loss", "abort",
            }
        ),
        adjectives=frozenset({"failed", "aborted", "faulty"}),
        particle_verbs=(("break", frozenset({"down", "up", "apart"})),),
        patient_prefer=frozenset({SPACECRAFT, "ORG"}),
    ),
    "flyby": Lexicon(
        verbs=frozenset({"flyby"}),
        nouns=frozenset({"flyby", "fly-by", "flypast", "encounter"}),
        particle_verbs=(
            ("fly", frozenset({"by", "past"})),
            ("swing", frozenset({"by", "past"})),
            ("sweep", frozenset({"by", "past"})),
        ),
        location_preps=frozenset(),
        patient_preps=frozenset({"by", "past", "near", "of"}),
        patient_prefer=frozenset({CELESTIAL, SPACECRAFT}),
        prep_particles=True,
    ),
    "landing": Lexicon(
        verbs=frozenset({"land", "alight"}),
        nouns=frozenset(
            {"landing", "touchdown", "splashdown", "reentry", "re-entry", "descent"}
        ),
        particle_verbs=(
            ("touch", frozenset({"down"})),
            ("splash", frozenset({"down"})),
            ("set", frozenset({"down"})),
        ),
        patient_prefer=frozenset({SPACECRAFT}),
    ),
    "launch": Lexicon(
        verbs=frozenset({"launch", "loft"}),
        nouns=frozenset({"launch", "liftoff", "lift-off", "blastoff", "launching"}),
        particle_verbs=(
            ("lift", frozenset({"off"})),
            ("blast", frozenset({"off"})),
            ("take", frozenset({"off"})),
        ),
        patient_prefer=frozenset({SPACECRAFT}),
    ),
    "discovery": Lexicon(
        verbs=frozenset(
            {"discover", "detect", "identify", "spot", "uncover", "observe", "find"}
        ),
        nouns=frozenset({"discovery", "detection", "finding", "sighting", "observation"}),
        location_preps=frozenset({"in", "on", "near", "around", "at"}),
        patient_prefer=frozenset({CELESTIAL, SPACECRAFT, "ORG"}),
    ),
}
EVENT_TYPES: tuple[str, ...] = tuple(EVENT_LEXICON)

# "failed to launch", "aborted the docking" - the embedded event did not happen
NEGATING_HEADS = frozenset({"fail", "abort", "cancel", "scrub", "postpone", "delay"})

TIME_PREPS = frozenset({"on", "in", "at", "by", "during", "after", "before", "since",
                        "until", "from", "for", "within"})

# Event types whose grammatical subject is the *theme*, not the actor:
# "Discovery lifted off" - Discovery is the thing launched. Docking and flyby
# are excluded on purpose ("Soyuz docked with the ISS" has a real agent).
THEME_SUBJECT_EVENTS = frozenset({"launch", "landing"})
# Arguments are never temporal expressions - those belong in time_expr.
ACTOR_LABELS = frozenset({"ORG", "PERSON", "GPE", "NORP"})
NP_MODIFIER_DEPS = frozenset({"det", "compound", "amod", "poss", "nummod", "nmod"})


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Event:
    """One extracted event mention (one row of outputs/events.csv).

    The first eight fields are the Stage 4 spec columns, in spec order. The
    remainder are Stage 5 hooks: ``event_id`` gives the event graph a stable
    node id, ``*_key`` line the arguments up with entity_summary.csv, and
    ``time_key`` / ``time_prep`` feed TIMEX resolution against ``get_dct``.
    """

    doc_id: str
    sent_id: int
    event_type: str
    trigger_word: str
    agent: str
    patient: str
    time_expr: str
    sentence: str
    event_id: str
    trigger_lemma: str
    trigger_token: int
    agent_key: str
    patient_key: str
    time_key: str
    time_prep: str
    location: str
    location_key: str
    negated: bool
    confidence: float


EVENT_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Event))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _np_text(token: Token) -> str:
    """Compact noun phrase around a token: determiners/compounds/adjectives."""
    left = token.i
    for child in token.children:
        if child.dep_ in NP_MODIFIER_DEPS and child.i < token.i:
            left = min(left, child.left_edge.i)
    return token.doc[left : token.i + 1].text.strip()


def _matches(token: Token, lexicon: Lexicon) -> bool:
    lemma = token.lemma_.lower()
    particles = {c.lower_ for c in token.children if c.dep_ == "prt"}
    if lexicon.prep_particles and not is_passive(token):
        # en_core_web_sm parses "fly by Titan" with by/prep, not by/prt. The
        # passive guard keeps "flown by NASA" out.
        particles |= {c.lower_ for c in token.children if c.dep_ == "prep"}
    required = lexicon.particles_for(lemma)
    if required is not None and particles & required:
        return True
    if token.pos_ in VERBISH and lemma in lexicon.verbs:
        return True
    if token.pos_ in {"NOUN", "PROPN"} and lemma in lexicon.nouns:
        return True
    if token.pos_ == "ADJ" and token.lower_ in lexicon.adjectives:
        return True
    if token.pos_ == "VERB" and token.dep_ == "amod" and token.lower_ in lexicon.adjectives:
        return True
    return False


def _event_type_for(token: Token) -> str | None:
    for event_type, lexicon in EVENT_LEXICON.items():
        if _matches(token, lexicon):
            return event_type
    return None


def _is_negated(token: Token) -> bool:
    if any(child.dep_ == "neg" for child in token.children):
        return True
    head = token.head
    if head is not token and token.dep_ in {"xcomp", "ccomp", "dobj", "nsubjpass"}:
        if head.lemma_.lower() in NEGATING_HEADS:
            return True
    return False


def _pick(
    ctx: SentenceContext, tokens: Iterable[Token], prefer: Iterable[str] = ()
) -> tuple[str, str, Token | None]:
    """Choose the best argument filler: entity mentions first, then a bare NP."""
    preferred = set(prefer)
    entity_hits: list[tuple[EntityMention, Token]] = []
    plain: list[Token] = []
    for token in tokens:
        mention = ctx.mention_at(token)
        if mention is not None and (
            mention.is_temporal or mention.label in ("DATE", "TIME")
        ):
            continue  # "landed on Monday" - Monday is time_expr, not patient
        if mention is not None:
            entity_hits.append((mention, token))
        else:
            plain.append(token)
    if entity_hits:
        if preferred:
            for mention, token in entity_hits:
                if mention.label in preferred:
                    return mention.mention, mention.entity_key, token
        mention, token = entity_hits[0]
        return mention.mention, mention.entity_key, token
    if plain:
        token = plain[0]
        if token.pos_ in {"NOUN", "PROPN", "PRON"}:
            return _np_text(token), "", token
    return "", "", None


def _time_for(ctx: SentenceContext, trigger: Token) -> tuple[str, str, str]:
    """Temporal expression for an event: attached first, then nearest in sentence."""
    temporals = ctx.temporal_mentions()
    if not temporals:
        return "", "", ""

    # (a) a DATE/TIME hanging directly off the trigger
    for child in trigger.children:
        if child.dep_ in {"prep", "npadvmod", "advmod", "tmod"}:
            candidates = (
                [g for g in child.children if g.dep_ == "pobj"]
                if child.dep_ == "prep"
                else [child]
            )
            for candidate in expand_conj(candidates):
                mention = ctx.mention_at(candidate)
                if mention is not None and mention in temporals:
                    prep = child.lower_ if child.dep_ == "prep" else ""
                    if prep and prep not in TIME_PREPS:
                        continue
                    return mention.mention, mention.entity_key, prep

    # (b) nearest temporal mention by token distance
    best: tuple[int, EntityMention] | None = None
    for mention in temporals:
        head = ctx.head_token(mention)
        if head is None:
            continue
        distance = abs(head.i - trigger.i)
        if best is None or distance < best[0]:
            best = (distance, mention)
    if best is not None:
        return best[1].mention, best[1].entity_key, ""
    return "", "", ""


def _location_for(
    ctx: SentenceContext, trigger: Token, lexicon: Lexicon
) -> tuple[str, str]:
    for token in prep_objects(trigger, lexicon.location_preps):
        mention = ctx.mention_at(token)
        if mention is not None and mention.label in (PLACE_LABELS | {CELESTIAL, "ORG"}):
            return mention.mention, mention.entity_key
    return "", ""


# --------------------------------------------------------------------------
# slot filling
# --------------------------------------------------------------------------
def _fill_slots(
    ctx: SentenceContext, trigger: Token, event_type: str, lexicon: Lexicon
) -> tuple[list[Token], list[Token]]:
    """Return (agent candidates, patient candidates) for a trigger token."""
    agents: list[Token] = []
    patients: list[Token] = []

    if trigger.pos_ in VERBISH or (trigger.pos_ == "VERB" and trigger.dep_ == "amod"):
        if trigger.dep_ == "amod":
            # "the delayed launch", "the failed mission"
            patients = [trigger.head]
        elif is_passive(trigger):
            patients = subjects_of(trigger) or implicit_subjects(trigger)
            agents = agents_of(trigger)
        else:
            agents = subjects_of(trigger) or implicit_subjects(trigger)
            patients = objects_of(trigger)
            if not patients:
                patients = prep_objects(trigger, lexicon.patient_preps)
    elif trigger.pos_ == "ADJ":
        if trigger.dep_ == "amod":
            patients = [trigger.head]
        elif trigger.dep_ in {"acomp", "attr"}:
            patients = subjects_of(trigger.head)
        agents = agents_of(trigger) + prep_objects(trigger, {"by"})
    else:  # nominal trigger: "the launch of Discovery by NASA"
        patients = prep_objects(trigger, {"of"}) + compounds_of(trigger)
        agents = prep_objects(trigger, {"by"})
        for owner in possessors_of(trigger):
            mention = ctx.mention_at(owner)
            if mention is not None and mention.label in {"ORG", "PERSON", "GPE", "NORP"}:
                agents.append(owner)
            else:
                patients.append(owner)
        patients += appositions_of(trigger)

    # Theme-subject events ("Discovery lifted off") and unaccusative failures
    # ("the main engine failed"): nothing acted, so the subject is the theme.
    if event_type in THEME_SUBJECT_EVENTS or event_type == "failure":
        craft = [
            t for t in agents
            if (m := ctx.mention_at(t)) is not None and m.label == SPACECRAFT
        ]
        already = any(
            (m := ctx.mention_at(t)) is not None and m.label == SPACECRAFT
            for t in patients
        )
        if craft and not already:
            patients = craft + patients
            agents = [t for t in agents if t not in craft]
        elif agents and not patients:
            actors = [
                t for t in agents
                if (m := ctx.mention_at(t)) is not None and m.label in ACTOR_LABELS
            ]
            if not actors:
                agents, patients = [], agents

    return agents, patients


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
def extract_events_from_sentence(
    ctx: SentenceContext, resolve_noun_args: bool = True
) -> list[Event]:
    events: dict[tuple[str, int], Event] = {}

    for token in ctx.tokens:
        # A token inside an entity name is a name, not a trigger. Without this
        # the spacecraft "Discovery" fires a discovery event on every mention
        # (89 of them in this corpus), and "Launch Complex 39A" a launch.
        if ctx.mention_at(token) is not None:
            continue
        event_type = _event_type_for(token)
        if event_type is None:
            continue
        lexicon = EVENT_LEXICON[event_type]
        agent_tokens, patient_tokens = _fill_slots(ctx, token, event_type, lexicon)

        agent, agent_key, _ = _pick(ctx, agent_tokens, ACTOR_LABELS)
        patient, patient_key, _ = _pick(ctx, patient_tokens, lexicon.patient_prefer)
        if not resolve_noun_args:
            if not agent_key:
                agent = ""
            if not patient_key:
                patient = ""

        time_expr, time_key, time_prep = _time_for(ctx, token)
        location, location_key = _location_for(ctx, token, lexicon)

        confidence = 0.55
        if patient_key:
            confidence += 0.20
        elif patient:
            confidence += 0.05
        if agent_key:
            confidence += 0.10
        if time_expr:
            confidence += 0.05
        if token.pos_ not in VERBISH:
            confidence -= 0.05
        confidence = round(min(confidence, 0.95), 2)

        events[(event_type, token.i)] = Event(
            doc_id=ctx.doc_id,
            sent_id=ctx.sent_id,
            event_type=event_type,
            trigger_word=trigger_text(token),
            agent=agent,
            patient=patient,
            time_expr=time_expr,
            sentence=ctx.text,
            event_id=f"{ctx.doc_id}:{ctx.sent_id}:{token.i}",
            trigger_lemma=token.lemma_.lower(),
            trigger_token=token.i,
            agent_key=agent_key,
            patient_key=patient_key,
            time_key=time_key,
            time_prep=time_prep,
            location=location,
            location_key=location_key,
            negated=_is_negated(token),
            confidence=confidence,
        )

    return sorted(events.values(), key=lambda e: e.trigger_token)


def extract_events(
    contexts: Iterable[SentenceContext], resolve_noun_args: bool = True
) -> Iterator[Event]:
    for ctx in contexts:
        yield from extract_events_from_sentence(ctx, resolve_noun_args=resolve_noun_args)


# --------------------------------------------------------------------------
# summary + IO
# --------------------------------------------------------------------------
def summarise_events(events: Sequence[Event]) -> list[dict]:
    by_type: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        by_type[event.event_type].append(event)

    rows: list[dict] = []
    for event_type in EVENT_TYPES:
        group = by_type.get(event_type, [])
        triggers = Counter(e.trigger_lemma for e in group)
        rows.append(
            {
                "event_type": event_type,
                "count": len(group),
                "docs": len({e.doc_id for e in group}),
                "with_agent": sum(1 for e in group if e.agent_key),
                "with_patient": sum(1 for e in group if e.patient_key),
                "with_time": sum(1 for e in group if e.time_expr),
                "negated": sum(1 for e in group if e.negated),
                "mean_confidence": round(
                    sum(e.confidence for e in group) / len(group), 3
                ) if group else 0.0,
                "top_triggers": "; ".join(f"{t} ({n})" for t, n in triggers.most_common(5)),
            }
        )
    rows.append(
        {
            "event_type": "ALL",
            "count": len(events),
            "docs": len({e.doc_id for e in events}),
            "with_agent": sum(1 for e in events if e.agent_key),
            "with_patient": sum(1 for e in events if e.patient_key),
            "with_time": sum(1 for e in events if e.time_expr),
            "negated": sum(1 for e in events if e.negated),
            "mean_confidence": round(
                sum(e.confidence for e in events) / len(events), 3
            ) if events else 0.0,
            "top_triggers": "",
        }
    )
    return rows


def write_events_csv(events: Sequence[Event], path: Path | None = None) -> Path:
    path = path or (output_dir() / "events.csv")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))
    return path


def write_event_summary_csv(rows: Sequence[dict], path: Path | None = None) -> Path:
    path = path or (output_dir() / "event_summary.csv")
    if not rows:
        return path
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
