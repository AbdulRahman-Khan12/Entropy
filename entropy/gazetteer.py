"""
entropy/gazetteer.py
====================
Domain gazetteer for the space / astronomy corpus.

Why this exists
---------------
``en_core_web_sm`` is trained on OntoNotes, which is news-general.  On a space
corpus it fails in specific, predictable ways:

  * mission names get read as people           - "Cassini" -> PERSON
  * hyphenated craft are missed entirely       - "Chandrayaan-3" -> (nothing)
  * planets drift between labels               - "Mars" -> LOC / ORG / PERSON
  * rocket designations fragment               - "Falcon 9" -> "Falcon" + CARDINAL

Relation extraction in the next pass keys off entity spans and labels, so this
noise propagates directly into the relation table.  A gazetteer fixes the
domain vocabulary cheaply and deterministically, and the statistical model
still handles everything outside the lists below.

The ruler is added *before* the ``ner`` component.  spaCy's ``EntityRecognizer``
does not overwrite entity spans that already exist on the Doc, so anything
matched here takes priority and the model fills in the rest.

Labels
------
Agencies, companies and institutes are emitted as ``ORG``, and launch sites as
``FAC``, so they merge with what spaCy already produces rather than fragmenting
the label set.  Two custom labels are introduced because spaCy has no adequate
equivalent:

  ``SPACECRAFT``  craft, missions, rockets, launch vehicles, telescopes
  ``CELESTIAL``   planets, moons, stars, comets, regions of space

``CELESTIAL`` matters more than it looks: the destination relations in the next
pass ("bound for Mars", "orbiting Saturn") need a reliable celestial-body class
to anchor on.

The ambiguity problem
---------------------
Many mission names are ordinary English words: Discovery, Opportunity, Spirit,
Curiosity, Columbia, Genesis, Dawn, Juno, Mercury, Enterprise.  Blind phrase
matching would tag every capitalised "Opportunity" at the start of a sentence.

Those names therefore live in :data:`AMBIGUOUS_SPACECRAFT` and are matched only
when a cue word sits next to them ("the Discovery shuttle", "rover
Opportunity").  Bare mentions are left to the statistical model.  This trades a
little recall for a lot of precision, which is the right trade when the output
feeds a relation extractor.
"""

from __future__ import annotations

from typing import Iterable

# --------------------------------------------------------------------------
# Organisations -> ORG
# --------------------------------------------------------------------------

AGENCIES = [
    "NASA", "National Aeronautics and Space Administration",
    "ESA", "European Space Agency",
    "ISRO", "Indian Space Research Organisation", "Indian Space Research Organization",
    "JAXA", "Japan Aerospace Exploration Agency",
    "Roscosmos", "Russian Federal Space Agency", "Russian Space Agency",
    "CNSA", "China National Space Administration",
    "CNES", "DLR", "UK Space Agency", "Canadian Space Agency", "CSA",
    "NOAA", "NRO", "National Reconnaissance Office",
    "ESO", "European Southern Observatory",
    "SETI Institute", "SETI",
    "Jet Propulsion Laboratory", "JPL",
    "Goddard Space Flight Center", "Johnson Space Center",
    "Marshall Space Flight Center", "Ames Research Center",
    "Applied Physics Laboratory",
    "Space Telescope Science Institute",
    "International Astronomical Union", "IAU",
]

COMPANIES = [
    "SpaceX", "Space Exploration Technologies",
    "Blue Origin", "Virgin Galactic", "Virgin Orbit",
    "Rocket Lab", "Firefly Aerospace", "Relativity Space",
    "United Launch Alliance", "ULA",
    "Arianespace", "Boeing", "Lockheed Martin", "Northrop Grumman",
    "Orbital Sciences", "Orbital ATK", "Sierra Nevada Corporation",
    "Scaled Composites", "Energia", "Khrunichev",
    "Antrix Corporation", "Airbus Defence and Space",
]

# --------------------------------------------------------------------------
# Launch sites and facilities -> FAC
# --------------------------------------------------------------------------

FACILITIES = [
    "Cape Canaveral", "Cape Canaveral Air Force Station",
    "Kennedy Space Center", "Baikonur Cosmodrome", "Baikonur",
    "Satish Dhawan Space Centre", "Sriharikota",
    "Vandenberg Air Force Base", "Vandenberg Space Force Base", "Vandenberg",
    "Wallops Flight Facility", "Wallops Island",
    "Guiana Space Centre", "Kourou",
    "Tanegashima Space Center", "Uchinoura Space Center",
    "Jiuquan Satellite Launch Center", "Xichang Satellite Launch Center",
    "Wenchang Satellite Launch Center", "Taiyuan Satellite Launch Center",
    "Plesetsk Cosmodrome", "Plesetsk", "Vostochny Cosmodrome",
    "Boca Chica", "Starbase", "Mojave Air and Space Port",
    "Mauna Kea Observatory", "Very Large Telescope", "Arecibo Observatory",
    "Palomar Observatory", "Keck Observatory",
]

# --------------------------------------------------------------------------
# Spacecraft, missions, launch vehicles -> SPACECRAFT
# Unambiguous names only.  Anything that is also a common English word belongs
# in AMBIGUOUS_SPACECRAFT below.
# --------------------------------------------------------------------------

SPACECRAFT = [
    # planetary and deep space
    "Cassini", "Cassini-Huygens", "Huygens",
    "Voyager 1", "Voyager 2", "Pioneer 10", "Pioneer 11",
    "New Horizons", "Rosetta", "Philae", "Deep Impact", "Stardust",
    "MESSENGER", "BepiColombo", "Parker Solar Probe", "Ulysses", "SOHO",
    "Mars Express", "Beagle 2", "Mars Odyssey", "Mars Global Surveyor",
    "Mars Reconnaissance Orbiter", "MAVEN", "Phoenix", "InSight",
    "Mars Science Laboratory", "Perseverance", "Ingenuity", "Sojourner",
    "Venus Express", "Akatsuki", "Magellan", "Hayabusa", "Hayabusa2",
    "OSIRIS-REx", "Lucy", "Psyche", "Europa Clipper", "JUICE",
    # telescopes and observatories in orbit
    "Hubble Space Telescope", "Hubble", "James Webb Space Telescope",
    "Spitzer Space Telescope", "Spitzer", "Chandra X-ray Observatory",
    "Kepler Space Telescope", "TESS", "Gaia", "Planck", "Herschel",
    "XMM-Newton", "Swift", "Fermi", "WMAP", "COBE", "IRAS",
    # crewed programmes and stations
    "International Space Station", "ISS", "Skylab", "Mir", "Tiangong",
    "Soyuz", "Progress", "Shenzhou", "Gaganyaan", "Orion", "Artemis",
    "SpaceShipOne", "SpaceShipTwo", "Crew Dragon", "Cargo Dragon",
    "Starliner", "Dragon", "Starship",
    # launch vehicles
    "Falcon 1", "Falcon 9", "Falcon Heavy", "Super Heavy",
    "Ariane 4", "Ariane 5", "Ariane 6", "Vega",
    "Atlas V", "Delta II", "Delta IV", "Delta IV Heavy",
    "Titan IV", "Titan II", "Saturn V", "Saturn IB",
    "PSLV", "GSLV", "LVM3", "SSLV",
    "Long March", "Proton", "Zenit", "Dnepr", "Rokot",
    "Antares", "Cygnus", "Electron", "Pegasus", "Minotaur", "H-IIA", "H-IIB",
    # Indian and other national missions
    "Chandrayaan-1", "Chandrayaan-2", "Chandrayaan-3",
    "Mangalyaan", "Mars Orbiter Mission", "Aditya-L1",
    "Cartosat", "RISAT", "INSAT", "GSAT", "IRNSS", "NavIC",
    "Chang'e", "Yutu", "Tianwen-1", "Zhurong",
    # shuttle programme, disambiguated by full name
    "Space Shuttle", "Space Shuttle Discovery", "Space Shuttle Atlantis",
    "Space Shuttle Endeavour", "Space Shuttle Columbia",
    "Space Shuttle Challenger", "Space Shuttle Enterprise",
    "Project Mercury", "Project Gemini", "Project Apollo",
    "Apollo 11", "Apollo 13", "Apollo 17",
]

#: Mission names that are also ordinary English words.  Matched only when a cue
#: word is adjacent - see :func:`build_patterns`.
AMBIGUOUS_SPACECRAFT = [
    "Discovery", "Atlantis", "Endeavour", "Columbia", "Challenger",
    "Enterprise", "Opportunity", "Spirit", "Curiosity", "Genesis",
    "Dawn", "Juno", "Galileo", "Apollo", "Gemini", "Voyager", "Pioneer",
    "Sentinel", "Odyssey", "Explorer", "Discoverer",
]

#: Words that, next to an ambiguous name, make the spacecraft reading safe.
CUE_WORDS = [
    "shuttle", "orbiter", "rover", "probe", "spacecraft", "craft",
    "capsule", "lander", "mission", "satellite", "telescope", "module",
    "rocket", "launcher", "vehicle", "flight",
]

# --------------------------------------------------------------------------
# Celestial bodies and regions -> CELESTIAL
# Single-token planet names that clash with mission names (Mercury) are handled
# by the ambiguity rules above; the bare token stays CELESTIAL, which is the
# far more common reading in this corpus.
# --------------------------------------------------------------------------

CELESTIAL = [
    # solar system
    "Sun", "Moon", "Earth", "Mercury", "Venus", "Mars", "Jupiter",
    "Saturn", "Uranus", "Neptune", "Pluto",
    # moons
    "Titan", "Europa", "Ganymede", "Callisto", "Io", "Enceladus",
    "Mimas", "Iapetus", "Rhea", "Dione", "Tethys", "Triton", "Charon",
    "Phobos", "Deimos", "Miranda", "Ariel", "Umbriel", "Oberon", "Titania",
    # small bodies
    "Ceres", "Vesta", "Pallas", "Eros", "Itokawa", "Ryugu", "Bennu",
    "Halley's Comet", "Comet Halley", "Hale-Bopp", "Shoemaker-Levy 9",
    "Tempel 1", "Wild 2", "Churyumov-Gerasimenko", "Eris", "Makemake",
    "Haumea", "Sedna", "Quaoar",
    # regions
    "Kuiper Belt", "Oort Cloud", "asteroid belt", "Solar System",
    "heliosphere", "Lagrange point",
    # stars, galaxies, deep sky
    "Alpha Centauri", "Proxima Centauri", "Betelgeuse", "Sirius",
    "Polaris", "Vega", "Rigel", "Antares",
    "Milky Way", "Andromeda", "Andromeda Galaxy", "Large Magellanic Cloud",
    "Small Magellanic Cloud", "Orion Nebula", "Crab Nebula", "Eagle Nebula",
    "Sagittarius A*", "Sagittarius A",
]

# Names appearing in more than one list.  "Vega" is a star and a launch vehicle;
# "Titan" is a moon and a rocket family; "Orion" is a nebula, a constellation
# and a crew capsule.  Multi-token forms ("Titan IV", "Orion Nebula") are
# matched first because the EntityRuler prefers the longest match, so the
# single-token entry only fires when no qualifier is present.
KNOWN_COLLISIONS = ("Vega", "Titan", "Orion", "Mercury", "Apollo", "Gemini")


# --------------------------------------------------------------------------
# Pattern construction
# --------------------------------------------------------------------------

def _phrase_patterns(names: Iterable[str], label: str) -> list[dict]:
    """Plain phrase patterns, matched case-sensitively on surface form."""
    return [{"label": label, "pattern": name} for name in names]


def _ambiguous_patterns(names: Iterable[str], label: str,
                        cues: Iterable[str] = CUE_WORDS) -> list[dict]:
    """Token patterns requiring a cue word immediately before or after the name.

    Covers "the Discovery shuttle" / "shuttle Discovery" / "Discovery orbiter",
    and leaves a bare "Discovery" for the statistical model to judge.
    """
    cue_list = list(cues)
    patterns: list[dict] = []
    for name in names:
        patterns.append({
            "label": label,
            "pattern": [{"ORTH": name}, {"LOWER": {"IN": cue_list}}],
        })
        patterns.append({
            "label": label,
            "pattern": [{"LOWER": {"IN": cue_list}}, {"ORTH": name}],
        })
    return patterns


def build_patterns() -> list[dict]:
    """Assemble the full EntityRuler pattern list."""
    patterns: list[dict] = []
    patterns += _phrase_patterns(AGENCIES, "ORG")
    patterns += _phrase_patterns(COMPANIES, "ORG")
    patterns += _phrase_patterns(FACILITIES, "FAC")
    patterns += _phrase_patterns(SPACECRAFT, "SPACECRAFT")
    patterns += _phrase_patterns(CELESTIAL, "CELESTIAL")
    patterns += _ambiguous_patterns(AMBIGUOUS_SPACECRAFT, "SPACECRAFT")
    return patterns


def add_gazetteer(nlp, name: str = "entropy_gazetteer"):
    """Insert the gazetteer EntityRuler into a loaded pipeline.

    Placed before ``ner`` so domain terms take priority; spaCy's entity
    recogniser preserves spans that already exist and annotates the remainder.
    """
    if name in nlp.pipe_names:
        return nlp.get_pipe(name)
    anchor = {"before": "ner"} if "ner" in nlp.pipe_names else {"last": True}
    ruler = nlp.add_pipe(
        "entity_ruler",
        name=name,
        config={"overwrite_ents": False, "validate": True},
        **anchor,
    )
    # Phrase patterns are built by running each pattern string through the
    # pipeline. The ruler sits late, so tagger/parser/lemmatiser would run over
    # ~700 short strings on every load (spaCy W012). Disabling everything else
    # leaves just the tokenizer, which is all ORTH/LOWER matching needs.
    others = [p for p in nlp.pipe_names if p != name]
    with nlp.select_pipes(disable=others):
        ruler.add_patterns(build_patterns())
    return ruler


#: Custom labels introduced here, for displacy colouring and downstream filters.
CUSTOM_LABELS = ("SPACECRAFT", "CELESTIAL")


def pattern_count() -> int:
    return len(build_patterns())
