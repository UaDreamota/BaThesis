from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
NS = {"tei": TEI_NS}
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from path_config import get_parlam_data_dir


@dataclass
class Affiliation:
    ref: str
    role: str | None
    start: dt.date | None
    end: dt.date | None


@dataclass
class PersonRecord:
    person_id: str
    name: str
    affiliations: list[Affiliation] = field(default_factory=list)


@dataclass
class OrgRecord:
    org_id: str
    name: str
    role: str


@dataclass
class CorpusContext:
    tei_root: Path
    corpus_id: str
    country: str
    people: dict[str, PersonRecord]
    orgs: dict[str, OrgRecord]
    topic_labels: dict[str, str]
    speaker_type_labels: dict[str, str]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def join_unique(items: list[str], sep: str = " | ") -> str:
    return sep.join(dict.fromkeys(item for item in items if item))


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_iso_date(raw: str | None) -> dt.date | None:
    if not raw:
        return None

    value = raw.strip()
    if not value:
        return None

    value = value.replace("Z", "+00:00")

    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        pass

    try:
        return dt.datetime.fromisoformat(value).date()
    except ValueError:
        pass

    match = re.match(r"^(\d{4}-\d{2}-\d{2})", value)
    if match:
        try:
            return dt.date.fromisoformat(match.group(1))
        except ValueError:
            return None
    return None


def in_range(day: dt.date, start: dt.date | None, end: dt.date | None) -> bool:
    if start and day < start:
        return False
    if end and day > end:
        return False
    return True


def org_name_from_node(org: ET.Element) -> str:
    candidates: list[tuple[int, str]] = []
    for org_name in org.findall("./tei:orgName", NS):
        text = clean_text("".join(org_name.itertext()))
        if not text:
            continue

        full = org_name.attrib.get("full", "")
        lang = org_name.attrib.get(XML_LANG, "")

        if full == "abb":
            rank = 0
        elif full == "yes" and lang == "en":
            rank = 1
        elif full == "yes":
            rank = 2
        else:
            rank = 3
        candidates.append((rank, text))

    if not candidates:
        return org.attrib.get(XML_ID, "")
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def load_orgs(list_org_path: Path) -> dict[str, OrgRecord]:
    tree = ET.parse(list_org_path)
    root = tree.getroot()
    orgs: dict[str, OrgRecord] = {}
    for org in root.findall(".//tei:org", NS):
        org_id = org.attrib.get(XML_ID)
        if not org_id:
            continue
        orgs[org_id] = OrgRecord(
            org_id=org_id,
            name=org_name_from_node(org),
            role=org.attrib.get("role", ""),
        )
    return orgs


def load_people(list_person_path: Path) -> dict[str, PersonRecord]:
    people: dict[str, PersonRecord] = {}
    person_tag = f"{{{TEI_NS}}}person"
    affiliation_tag = f"{{{TEI_NS}}}affiliation"
    pers_name_tag = f"{{{TEI_NS}}}persName"

    for _, elem in ET.iterparse(list_person_path, events=("end",)):
        if elem.tag != person_tag:
            continue

        person_id = elem.attrib.get(XML_ID)
        if not person_id:
            elem.clear()
            continue

        name = person_id
        pers_name = elem.find(f"./{pers_name_tag}")
        if pers_name is not None:
            parsed = clean_text("".join(pers_name.itertext()))
            if parsed:
                name = parsed

        affiliations: list[Affiliation] = []
        for aff in elem.findall(f"./{affiliation_tag}"):
            ref = (aff.attrib.get("ref") or "").lstrip("#")
            if not ref:
                continue
            affiliations.append(
                Affiliation(
                    ref=ref,
                    role=aff.attrib.get("role"),
                    start=parse_iso_date(aff.attrib.get("from")),
                    end=parse_iso_date(aff.attrib.get("to")),
                )
            )

        people[person_id] = PersonRecord(
            person_id=person_id,
            name=name,
            affiliations=affiliations,
        )
        elem.clear()

    return people


def detect_metadata_file(tei_root: Path, suffix: str) -> Path:
    matches = sorted(tei_root.glob(f"*-{suffix}.xml"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find '*-{suffix}.xml' under {tei_root}"
        )
    return matches[0]


def detect_optional_taxonomy_file(tei_root: Path, taxonomy_name: str) -> Path | None:
    matches = sorted(tei_root.glob(f"*taxonomy-{taxonomy_name}.xml"))
    return matches[0] if matches else None


def category_label(category: ET.Element) -> str:
    labels: list[tuple[int, str]] = []
    for cat_desc in category.findall("./tei:catDesc", NS):
        lang = cat_desc.attrib.get(XML_LANG, "")
        term = cat_desc.find("./tei:term", NS)
        text = clean_text("".join(term.itertext() if term is not None else cat_desc.itertext()))
        if not text:
            continue

        if lang == "en":
            rank = 0
        elif lang:
            rank = 1
        else:
            rank = 2
        labels.append((rank, text))

    if not labels:
        return category.attrib.get(XML_ID, "")
    labels.sort(key=lambda item: item[0])
    return labels[0][1]


def load_taxonomy_labels(taxonomy_path: Path | None) -> dict[str, str]:
    if taxonomy_path is None:
        return {}

    tree = ET.parse(taxonomy_path)
    root = tree.getroot()
    labels: dict[str, str] = {}
    for cat in root.findall(".//tei:category", NS):
        cat_id = cat.attrib.get(XML_ID)
        if not cat_id:
            continue
        labels[cat_id] = category_label(cat)
    return labels


def party_priority(ref: str, orgs: dict[str, OrgRecord]) -> int | None:
    record = orgs.get(ref)
    role = (record.role if record else "").strip()

    if role == "politicalParty" or ref.startswith(("politicalParty.", "pp.")):
        return 0
    if role in {"parliamentaryGroup", "parliamentaryFaction"} or ref.startswith(
        ("parliamentaryGroup.", "fr.", "gr.")
    ):
        return 1
    return None


def resolve_party(
    person: PersonRecord | None,
    day: dt.date | None,
    orgs: dict[str, OrgRecord],
) -> str:
    if person is None:
        return ""

    candidates: list[tuple[int, int, Affiliation]] = []
    fallback: list[tuple[int, int, Affiliation]] = []

    for aff in person.affiliations:
        priority = party_priority(aff.ref, orgs)
        if priority is None:
            continue

        start_ord = aff.start.toordinal() if aff.start else dt.date.min.toordinal()
        fallback.append((priority, -start_ord, aff))
        if day is None or in_range(day, aff.start, aff.end):
            candidates.append((priority, -start_ord, aff))

    pool = candidates or fallback
    if not pool:
        return ""

    pool.sort(key=lambda item: (item[0], item[1]))
    chosen = pool[0][2]
    record = orgs.get(chosen.ref)
    return record.name if record else chosen.ref


def pick_speech_date(root: ET.Element, file_path: Path) -> dt.date | None:
    for date_node in root.findall(".//tei:profileDesc//tei:setting//tei:date", NS):
        ana = date_node.attrib.get("ana", "")
        when = date_node.attrib.get("when")
        if "#parla.sitting" in ana and when:
            parsed = parse_iso_date(when)
            if parsed:
                return parsed

    for xpath in (".//tei:sourceDesc//tei:bibl//tei:date", ".//tei:sourceDesc//tei:date"):
        for date_node in root.findall(xpath, NS):
            parsed = parse_iso_date(date_node.attrib.get("when"))
            if parsed:
                return parsed

    match = re.search(r"(19|20)\d{2}-\d{2}-\d{2}", file_path.name)
    if match:
        return parse_iso_date(match.group(0))
    return None


def extract_meeting_fields(root: ET.Element) -> tuple[str, str, str]:
    term = ""
    session = ""
    sitting = ""
    for meeting in root.findall(".//tei:titleStmt/tei:meeting", NS):
        ana = meeting.attrib.get("ana", "")
        value = meeting.attrib.get("n", "") or clean_text("".join(meeting.itertext()))
        if "#parla.term" in ana and not term:
            term = value
        if ("#parla.session" in ana or "#parla.meeting" in ana) and not session:
            session = value
        if "#parla.sitting" in ana and not sitting:
            sitting = value
    return term, session, sitting


def extract_document_url(root: ET.Element) -> str:
    for idno in root.findall(".//tei:sourceDesc//tei:idno", NS):
        if idno.attrib.get("subtype") == "parliament":
            text = clean_text("".join(idno.itertext()))
            if text:
                return text
    return ""


def parse_u_ana(
    ana_raw: str,
    speaker_type_labels: dict[str, str],
    topic_labels: dict[str, str],
) -> tuple[str, str, str, str, str]:
    tokens = ana_raw.split()

    speaker_type = ""
    topic_code = ""
    used: set[str] = set()

    for tok in tokens:
        if tok.startswith("#"):
            maybe = tok[1:]
            if maybe in speaker_type_labels or maybe in {"chair", "regular", "guest"}:
                speaker_type = maybe
                used.add(tok)
                break

    for tok in tokens:
        if tok.startswith("topic:"):
            topic_code = tok.split(":", 1)[1]
            used.add(tok)
            break

    extra = [tok for tok in tokens if tok not in used]
    speaker_type_label = speaker_type_labels.get(speaker_type, "")
    topic_label = topic_labels.get(topic_code, "")

    return speaker_type, speaker_type_label, topic_code, topic_label, join_unique(extra)


def classify_note_content_kind(div_type: str, note_type: str) -> str:
    if note_type == "speaker":
        return "speaker_marker"
    if div_type == "commentSection" or note_type in {"comment", "narrative", "date", "time"}:
        return "procedural_commentary"
    return "note"


def base_row(
    context: CorpusContext,
    file_path: Path,
    doc_id: str,
    doc_lang: str,
    doc_ana: str,
    text_ana: str,
    date_str: str,
    meeting_term: str,
    meeting_session: str,
    meeting_sitting: str,
    source_doc_url: str,
    div_type: str,
    section_head: str,
) -> dict[str, str]:
    return {
        "corpus_id": context.corpus_id,
        "country": context.country,
        "date": date_str,
        "doc_id": doc_id,
        "doc_lang": doc_lang,
        "doc_ana": doc_ana,
        "text_ana": text_ana,
        "meeting_term": meeting_term,
        "meeting_session": meeting_session,
        "meeting_sitting": meeting_sitting,
        "source_doc_url": source_doc_url,
        "div_type": div_type,
        "section_head": section_head,
        "pre_u_note_types": "",
        "pre_u_notes": "",
        "speaker": "",
        "speaker_id": "",
        "party": "",
        "speaker_type": "",
        "speaker_type_label": "",
        "topic_code": "",
        "topic_label": "",
        "ana_raw": "",
        "ana_extra": "",
        "unit_type": "",
        "content_kind": "",
        "speech_id": "",
        "unit_id": "",
        "note_type": "",
        "note_text": "",
        "inline_note_types": "",
        "inline_note_count": "",
        "kinesic_types": "",
        "kinesic_count": "",
        "text": "",
        "source": "",
        "source_file": str(file_path),
    }


def extract_rows_from_file(
    file_path: Path,
    context: CorpusContext,
    unit: str,
    include_notes: bool,
) -> list[dict[str, str]]:
    tree = ET.parse(file_path)
    root = tree.getroot()

    speech_date = pick_speech_date(root, file_path)
    date_str = speech_date.isoformat() if speech_date else ""

    doc_id = root.attrib.get(XML_ID, file_path.stem)
    doc_lang = root.attrib.get(XML_LANG, "")
    doc_ana = clean_text(root.attrib.get("ana", ""))
    text_node = root.find(".//tei:text", NS)
    text_ana = clean_text(text_node.attrib.get("ana", "") if text_node is not None else "")
    meeting_term, meeting_session, meeting_sitting = extract_meeting_fields(root)
    source_doc_url = extract_document_url(root)

    rows: list[dict[str, str]] = []

    for div in root.findall(".//tei:div", NS):
        div_type = div.attrib.get("type", "")
        section_head = ""
        pending_note_types: list[str] = []
        pending_note_texts: list[str] = []

        for child in list(div):
            child_name = local_name(child.tag)

            if child_name == "head":
                heading = clean_text("".join(child.itertext()))
                if heading:
                    section_head = heading
                continue

            if child_name == "note":
                note_type = child.attrib.get("type", "untyped")
                note_text = clean_text("".join(child.itertext()))
                if note_text:
                    pending_note_types.append(note_type)
                    pending_note_texts.append(note_text)

                if include_notes and note_text:
                    note_row = base_row(
                        context=context,
                        file_path=file_path,
                        doc_id=doc_id,
                        doc_lang=doc_lang,
                        doc_ana=doc_ana,
                        text_ana=text_ana,
                        date_str=date_str,
                        meeting_term=meeting_term,
                        meeting_session=meeting_session,
                        meeting_sitting=meeting_sitting,
                        source_doc_url=source_doc_url,
                        div_type=div_type,
                        section_head=section_head,
                    )
                    note_row["unit_type"] = "note"
                    note_row["content_kind"] = classify_note_content_kind(div_type, note_type)
                    note_row["unit_id"] = child.attrib.get(XML_ID, "")
                    note_row["note_type"] = note_type
                    note_row["note_text"] = note_text
                    note_row["text"] = note_text
                    rows.append(note_row)
                continue

            if child_name != "u":
                continue

            u_node = child
            speech_id = u_node.attrib.get(XML_ID, "")
            who_raw = clean_text(u_node.attrib.get("who", ""))

            speaker_ids: list[str] = []
            for token in who_raw.split():
                if token.startswith("#"):
                    speaker_ids.append(token[1:])
                elif token:
                    speaker_ids.append(token)

            speaker_names: list[str] = []
            parties: list[str] = []
            for speaker_id in speaker_ids:
                person = context.people.get(speaker_id)
                speaker_names.append(person.name if person else speaker_id)
                party = resolve_party(person, speech_date, context.orgs)
                if party:
                    parties.append(party)

            speaker = join_unique(speaker_names)
            speaker_id_str = join_unique(speaker_ids)
            party = join_unique(parties)

            ana_raw = clean_text(u_node.attrib.get("ana", ""))
            speaker_type, speaker_type_label, topic_code, topic_label, ana_extra = parse_u_ana(
                ana_raw=ana_raw,
                speaker_type_labels=context.speaker_type_labels,
                topic_labels=context.topic_labels,
            )

            pre_u_note_types = join_unique(pending_note_types)
            pre_u_notes = " || ".join(dict.fromkeys(note for note in pending_note_texts if note))
            pending_note_types = []
            pending_note_texts = []

            inline_note_types = [
                note.attrib.get("type", "untyped")
                for note in u_node.findall(".//tei:note", NS)
            ]
            kinesic_types = [
                kin.attrib.get("type", "unspecified")
                for kin in u_node.findall(".//tei:kinesic", NS)
            ]

            seg_nodes = u_node.findall("./tei:seg", NS)
            use_segments = unit == "segment" or (unit == "auto" and bool(seg_nodes))
            speech_kind = "procedural_commentary" if div_type == "commentSection" else "speech"

            if use_segments and seg_nodes:
                for idx, seg_node in enumerate(seg_nodes, start=1):
                    seg_text = clean_text("".join(seg_node.itertext()))
                    if not seg_text:
                        continue

                    row = base_row(
                        context=context,
                        file_path=file_path,
                        doc_id=doc_id,
                        doc_lang=doc_lang,
                        doc_ana=doc_ana,
                        text_ana=text_ana,
                        date_str=date_str,
                        meeting_term=meeting_term,
                        meeting_session=meeting_session,
                        meeting_sitting=meeting_sitting,
                        source_doc_url=source_doc_url,
                        div_type=div_type,
                        section_head=section_head,
                    )
                    row["pre_u_note_types"] = pre_u_note_types
                    row["pre_u_notes"] = pre_u_notes
                    row["speaker"] = speaker
                    row["speaker_id"] = speaker_id_str
                    row["party"] = party
                    row["speaker_type"] = speaker_type
                    row["speaker_type_label"] = speaker_type_label
                    row["topic_code"] = topic_code
                    row["topic_label"] = topic_label
                    row["ana_raw"] = ana_raw
                    row["ana_extra"] = ana_extra
                    row["unit_type"] = "segment"
                    row["content_kind"] = speech_kind
                    row["speech_id"] = speech_id
                    row["unit_id"] = seg_node.attrib.get(XML_ID, f"{speech_id}.seg{idx}")
                    row["inline_note_types"] = join_unique(inline_note_types)
                    row["inline_note_count"] = str(len(inline_note_types))
                    row["kinesic_types"] = join_unique(kinesic_types)
                    row["kinesic_count"] = str(len(kinesic_types))
                    row["text"] = seg_text
                    row["source"] = u_node.attrib.get("source", "")
                    rows.append(row)
                continue

            if seg_nodes:
                text_parts = [clean_text("".join(seg_node.itertext())) for seg_node in seg_nodes]
                speech_text = clean_text(" ".join(part for part in text_parts if part))
            else:
                speech_text = clean_text("".join(u_node.itertext()))

            if not speech_text:
                continue

            row = base_row(
                context=context,
                file_path=file_path,
                doc_id=doc_id,
                doc_lang=doc_lang,
                doc_ana=doc_ana,
                text_ana=text_ana,
                date_str=date_str,
                meeting_term=meeting_term,
                meeting_session=meeting_session,
                meeting_sitting=meeting_sitting,
                source_doc_url=source_doc_url,
                div_type=div_type,
                section_head=section_head,
            )
            row["pre_u_note_types"] = pre_u_note_types
            row["pre_u_notes"] = pre_u_notes
            row["speaker"] = speaker
            row["speaker_id"] = speaker_id_str
            row["party"] = party
            row["speaker_type"] = speaker_type
            row["speaker_type_label"] = speaker_type_label
            row["topic_code"] = topic_code
            row["topic_label"] = topic_label
            row["ana_raw"] = ana_raw
            row["ana_extra"] = ana_extra
            row["unit_type"] = "speech"
            row["content_kind"] = speech_kind
            row["speech_id"] = speech_id
            row["unit_id"] = speech_id
            row["inline_note_types"] = join_unique(inline_note_types)
            row["inline_note_count"] = str(len(inline_note_types))
            row["kinesic_types"] = join_unique(kinesic_types)
            row["kinesic_count"] = str(len(kinesic_types))
            row["text"] = speech_text
            row["source"] = u_node.attrib.get("source", "")
            rows.append(row)

    return rows


def is_tei_speech_file(path: Path) -> bool:
    if path.suffix.lower() != ".xml":
        return False
    name = path.name
    if "listPerson" in name or "listOrg" in name:
        return False
    if "taxonomy" in name:
        return False
    if name == "00README.txt":
        return False
    return any(re.fullmatch(r"\d{4}", part) for part in path.parts)


def collect_tei_files(tei_root: Path) -> list[Path]:
    return sorted(path for path in tei_root.rglob("*.xml") if is_tei_speech_file(path))


def parse_country_and_corpus(tei_root: Path) -> tuple[str, str]:
    corpus_id = tei_root.parent.name if tei_root.parent.name.startswith("ParlaMint-") else tei_root.name
    match = re.search(r"ParlaMint-([A-Za-z]+)", corpus_id)
    country = match.group(1) if match else ""
    return country, corpus_id


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return REPO_ROOT / path


def discover_tei_roots(path: Path) -> list[Path]:
    if any(path.glob("*-listPerson.xml")) and any(path.glob("*-listOrg.xml")):
        return [path]

    direct_tei_roots = sorted(path.glob("ParlaMint-*.TEI"))
    if direct_tei_roots:
        return direct_tei_roots

    nested_tei_roots = sorted(path.glob("ParlaMint-*/ParlaMint-*.TEI"))
    if nested_tei_roots:
        return nested_tei_roots

    if path.name.endswith(".TEI"):
        return [path]

    raise FileNotFoundError(
        f"Could not discover any ParlaMint .TEI directories under: {path}"
    )


def load_corpus_context(tei_root: Path) -> CorpusContext:
    list_person_path = detect_metadata_file(tei_root, "listPerson")
    list_org_path = detect_metadata_file(tei_root, "listOrg")

    topic_taxonomy = detect_optional_taxonomy_file(tei_root, "topic")
    speaker_type_taxonomy = detect_optional_taxonomy_file(tei_root, "speaker_types")

    country, corpus_id = parse_country_and_corpus(tei_root)

    print(f"Loading metadata for {corpus_id} from: {tei_root}")
    orgs = load_orgs(list_org_path)
    people = load_people(list_person_path)
    topic_labels = load_taxonomy_labels(topic_taxonomy)
    speaker_type_labels = load_taxonomy_labels(speaker_type_taxonomy)

    print(
        f"Loaded {corpus_id}: people={len(people)} orgs={len(orgs)} "
        f"topic_labels={len(topic_labels)} speaker_type_labels={len(speaker_type_labels)}"
    )

    return CorpusContext(
        tei_root=tei_root,
        corpus_id=corpus_id,
        country=country,
        people=people,
        orgs=orgs,
        topic_labels=topic_labels,
        speaker_type_labels=speaker_type_labels,
    )


def parse_args() -> argparse.Namespace:
    parlam_data_dir = get_parlam_data_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Extract ParlaMint TEI speeches/segments and procedural notes into CSV, "
            "with topic/speaker metadata."
        )
    )
    parser.add_argument(
        "--tei-root",
        type=Path,
        default=parlam_data_dir,
        help=(
            "Path to a single ParlaMint .TEI directory, "
            "or to a parent folder that contains multiple ParlaMint corpora."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=parlam_data_dir / "parlamint_extracted.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=parlam_data_dir,
        help=(
            "Output directory when using --per-corpus. "
            "One CSV per corpus will be written here."
        ),
    )
    parser.add_argument(
        "--unit",
        choices=["auto", "speech", "segment"],
        default="auto",
        help="Analysis unit for speeches: full speech, segment, or auto (segment if present).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help=(
            "Optional limit for number of TEI files. "
            "In --per-corpus mode, this is applied per corpus."
        ),
    )
    output_mode_group = parser.add_mutually_exclusive_group()
    output_mode_group.add_argument(
        "--per-corpus",
        dest="per_corpus",
        action="store_true",
        help=(
            "Write one CSV per corpus (country) under --output-dir."
        ),
    )
    output_mode_group.add_argument(
        "--single-output",
        dest="per_corpus",
        action="store_false",
        help="Write one merged CSV to --output.",
    )
    parser.set_defaults(per_corpus=True)

    notes_group = parser.add_mutually_exclusive_group()
    notes_group.add_argument(
        "--include-notes",
        dest="include_notes",
        action="store_true",
        help="Include standalone TEI note rows (procedural/commentary metadata).",
    )
    notes_group.add_argument(
        "--no-notes",
        dest="include_notes",
        action="store_false",
        help="Exclude standalone TEI note rows.",
    )
    parser.set_defaults(include_notes=True)

    return parser.parse_args()


FIELDNAMES = [
    "corpus_id",
    "country",
    "date",
    "doc_id",
    "doc_lang",
    "doc_ana",
    "text_ana",
    "meeting_term",
    "meeting_session",
    "meeting_sitting",
    "source_doc_url",
    "div_type",
    "section_head",
    "pre_u_note_types",
    "pre_u_notes",
    "speaker",
    "speaker_id",
    "party",
    "speaker_type",
    "speaker_type_label",
    "topic_code",
    "topic_label",
    "ana_raw",
    "ana_extra",
    "unit_type",
    "content_kind",
    "speech_id",
    "unit_id",
    "note_type",
    "note_text",
    "inline_note_types",
    "inline_note_count",
    "kinesic_types",
    "kinesic_count",
    "text",
    "source",
    "source_file",
]


def write_rows_to_csv(
    output_path: Path,
    jobs: list[tuple[CorpusContext, Path]],
    unit: str,
    include_notes: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for idx, (context, tei_file) in enumerate(jobs, start=1):
            rows = extract_rows_from_file(
                file_path=tei_file,
                context=context,
                unit=unit,
                include_notes=include_notes,
            )
            for row in rows:
                writer.writerow(row)
                rows_written += 1

            if idx % 200 == 0:
                print(f"Processed {idx}/{len(jobs)} files, rows={rows_written}")

    return rows_written


def main() -> None:
    args = parse_args()
    tei_input = resolve_input_path(args.tei_root)
    output_path = resolve_input_path(args.output)
    output_dir = resolve_input_path(args.output_dir)

    tei_roots = discover_tei_roots(tei_input)
    print(f"Discovered TEI roots: {len(tei_roots)}")
    for tei_root in tei_roots:
        print(f" - {tei_root}")

    contexts = [load_corpus_context(root) for root in tei_roots]

    if args.per_corpus:
        total_rows = 0
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Per-corpus mode enabled. Output directory: {output_dir}")

        for context in contexts:
            corpus_files = collect_tei_files(context.tei_root)
            if args.max_files is not None:
                corpus_files = corpus_files[: args.max_files]

            jobs = [(context, file_path) for file_path in corpus_files]
            corpus_output = output_dir / f"{context.corpus_id}_extracted.csv"
            print(
                f"{context.corpus_id}: writing {len(jobs)} files to {corpus_output}"
            )

            rows_written = write_rows_to_csv(
                output_path=corpus_output,
                jobs=jobs,
                unit=args.unit,
                include_notes=args.include_notes,
            )
            print(f"{context.corpus_id}: wrote {rows_written} rows")
            total_rows += rows_written

        print(
            f"Done. Wrote {total_rows} rows across {len(contexts)} corpus CSV files in {output_dir}"
        )
        return

    file_jobs: list[tuple[CorpusContext, Path]] = []
    for context in contexts:
        corpus_files = collect_tei_files(context.tei_root)
        print(f"{context.corpus_id}: speech files={len(corpus_files)}")
        file_jobs.extend((context, file_path) for file_path in corpus_files)

    file_jobs.sort(key=lambda item: str(item[1]))
    if args.max_files is not None:
        file_jobs = file_jobs[: args.max_files]

    print(f"Total TEI speech files to process: {len(file_jobs)}")
    rows_written = write_rows_to_csv(
        output_path=output_path,
        jobs=file_jobs,
        unit=args.unit,
        include_notes=args.include_notes,
    )

    print(f"Done. Wrote {rows_written} rows to {output_path}")


if __name__ == "__main__":
    main()
