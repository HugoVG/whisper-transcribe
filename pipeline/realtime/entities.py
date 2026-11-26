"""Live entity extraction from transcript text (Dutch + English).

Pulls structured context out of a segment so a support agent UI can react
live: ticket numbers spoken as digits ("TI 0 1 ... 5674"), device numbers
("NL012345"), dates ("donderdag 5 december"), emails, phone numbers.

Pure stdlib — no models required.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta


@dataclass
class Entity:
    type: str        # ticket_number | device_number | date | email | phone_number | number
    value: str       # normalized value, e.g. "TI01015674", "NL012345", "2026-12-05"
    raw: str         # original text span
    start: int       # char offset in the segment text
    end: int
    confidence: float

    def as_dict(self) -> dict:
        return asdict(self)


NUMBER_WORDS = {
    # Dutch
    "nul": "0", "een": "1", "één": "1", "eén": "1", "twee": "2", "drie": "3",
    "vier": "4", "vijf": "5", "zes": "6", "zeven": "7", "acht": "8",
    "negen": "9", "tien": "10",
    # English
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}

NL_MONTHS = {
    "januari": 1, "februari": 2, "maart": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "augustus": 8, "september": 9, "oktober": 10, "november": 11,
    "december": 12,
}
EN_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
NL_WEEKDAYS = ["maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag"]
EN_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

_WORD_RE = re.compile(r"[A-Za-zà-ÿÀ-ÿéÉèÈ]+")


def normalize_number_words(text: str) -> str:
    """Replace standalone number words (nl/en) with digits, but only in
    digit-like contexts: runs of 2+ consecutive number words, or directly
    after a keyword like TI / ticket / nummer / device."""
    tokens = list(_WORD_RE.finditer(text))
    if not tokens:
        return text
    # find runs of consecutive number-word tokens (allowing single spaces)
    i = 0
    spans: list[tuple[int, int, str]] = []
    while i < len(tokens):
        if tokens[i].group(0).lower() in NUMBER_WORDS:
            j = i
            while j + 1 < len(tokens) and tokens[j + 1].group(0).lower() in NUMBER_WORDS:
                gap = text[tokens[j].end():tokens[j + 1].start()]
                if gap.strip() == "":
                    j += 1
                else:
                    break
            if j > i:  # run of 2+: replace the whole run with contiguous digits
                digits = "".join(NUMBER_WORDS[tokens[k].group(0).lower()] for k in range(i, j + 1))
                spans.append((tokens[i].start(), tokens[j].end(), digits))
                i = j + 1
            else:
                # single number word: only convert when preceded by a keyword
                prev = text[:tokens[i].start()].rstrip().lower()
                if re.search(r"\b(ti|ticket|nummer|number|nr|no|device|id|serial)\s*$", prev):
                    spans.append((tokens[i].start(), tokens[i].end(), NUMBER_WORDS[tokens[i].group(0).lower()]))
                i += 1
        else:
            i += 1
    out = text
    for s, e, digit in reversed(spans):
        out = out[:s] + digit + out[e:]
    return out


def collapse_repeated_digits(digits: str) -> str:
    """Collapse immediate repetition used for confirmation: '56745674' -> '5674'."""
    n = len(digits)
    if n >= 4 and n % 2 == 0 and digits[: n // 2] == digits[n // 2:]:
        return digits[: n // 2]
    return digits


_collapse_repeated_digits = collapse_repeated_digits


_TICKET_PREFIX_RE = re.compile(
    r"\b(?P<prefix>TI|TK|T)\s*(?P<digits>(?:\d[\s-]*){2,})", re.IGNORECASE
)
_TICKET_KEYWORD_RE = re.compile(
    r"\b(?:ticket|case)\s*(?:nummer|number|nr\.?|no\.?|id)?\s*(?:is\s*)?[:#]?\s*"
    r"(?P<code>[A-Z]{0,3}\s?\d[\d\s-]{2,})",
    re.IGNORECASE,
)
_DEVICE_RE = re.compile(r"\b(?P<letters>[A-Z]{2})\s?(?P<digits>\d{6,})\b")
_DEVICE_KEYWORD_RE = re.compile(
    r"\b(?:device|toestel|apparaat)\s*(?:nummer|number|id|serial)?\s*[:#]?\s*"
    r"(?P<code>[A-Z0-9][A-Z0-9-]{4,})",
    re.IGNORECASE,
)
_DIGIT_RUN_RE = re.compile(r"\b(?P<digits>\d{4,})\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE_RE = re.compile(r"(?P<phone>(?:\+31|0031|0)\s?\d[\d\s-]{7,}\d)")

_DATE_NL_RE = re.compile(
    r"\b(?:(?P<wd>" + "|".join(NL_WEEKDAYS) + r")\s+)?(?P<day>\d{1,2})(?:e|ste|de)?\s+"
    r"(?P<month>" + "|".join(NL_MONTHS) + r")\b",
    re.IGNORECASE,
)
_DATE_EN_RE = re.compile(
    r"\b(?:(?P<wd>" + "|".join(EN_WEEKDAYS) + r"),?\s+)?(?:(?P<day>\d{1,2})(?:st|nd|rd|th)\s+)?"
    r"(?P<month>" + "|".join(EN_MONTHS) + r")\s+(?P<day2>\d{1,2})?\b"
    r"|\b(?P<day3>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<month2>" + "|".join(EN_MONTHS) + r")\b",
    re.IGNORECASE,
)
_RELATIVE_RE = re.compile(
    r"\b(?P<rel>vandaag|today|morgen|tomorrow|overmorgen|day after tomorrow)\b",
    re.IGNORECASE,
)


def _next_date(day: int, month: int, today: date) -> date | None:
    try:
        candidate = date(today.year, month, day)
    except ValueError:
        return None
    if candidate < today:
        try:
            candidate = date(today.year + 1, month, day)
        except ValueError:
            return None
    return candidate


def extract_entities(text: str, *, today: date | None = None) -> list[Entity]:
    """Extract structured entities from one transcript segment.

    `today` is injectable for deterministic tests.
    """
    today = today or date.today()
    norm = normalize_number_words(text)
    entities: list[Entity] = []

    def add(type_: str, value: str, raw: str, start: int, end: int, conf: float) -> None:
        # skip overlapping duplicates of the same type (nl + en patterns can both match)
        if any(e.type == type_ and not (end <= e.start or start >= e.end) for e in entities):
            return
        entities.append(Entity(type_, value, raw.strip(), start, end, conf))

    for m in _TICKET_PREFIX_RE.finditer(norm):
        digits = re.sub(r"\D", "", m.group("digits"))
        digits = _collapse_repeated_digits(digits)
        add("ticket_number", m.group("prefix").upper() + digits, m.group(0), m.start(), m.end(), 0.9)

    for m in _TICKET_KEYWORD_RE.finditer(norm):
        code = m.group("code").upper().replace(" ", "").replace("-", "")
        if re.search(r"\d", code):
            add("ticket_number", code, m.group(0), m.start(), m.end(), 0.85)

    for m in _DEVICE_RE.finditer(norm):
        add("device_number", m.group("letters").upper() + m.group("digits"), m.group(0), m.start(), m.end(), 0.9)

    for m in _DEVICE_KEYWORD_RE.finditer(norm):
        code = m.group("code").upper().replace(" ", "")
        if code not in {e.value for e in entities}:
            add("device_number", code, m.group(0), m.start(), m.end(), 0.85)

    for m in _DIGIT_RUN_RE.finditer(norm):
        digits = _collapse_repeated_digits(m.group("digits"))
        add("number", digits, m.group(0), m.start(), m.end(), 0.55)

    for m in _EMAIL_RE.finditer(norm):
        add("email", m.group(0).lower(), m.group(0), m.start(), m.end(), 0.95)

    for m in _PHONE_RE.finditer(norm):
        digits = re.sub(r"\D", "", m.group("phone"))
        add("phone_number", digits, m.group(0), m.start(), m.end(), 0.8)

    def add_date(day: int, month: int, raw: str, start: int, end: int) -> None:
        d = _next_date(day, month, today)
        if d:
            add("date", d.isoformat(), raw, start, end, 0.85)

    for m in _DATE_NL_RE.finditer(norm):
        add_date(int(m.group("day")), NL_MONTHS[m.group("month").lower()], m.group(0), m.start(), m.end())

    for m in _DATE_EN_RE.finditer(norm):
        month_name = (m.group("month") or m.group("month2") or "").lower()
        day = m.group("day") or m.group("day2") or m.group("day3")
        if month_name in EN_MONTHS and day:
            add_date(int(day), EN_MONTHS[month_name], m.group(0), m.start(), m.end())

    for m in _RELATIVE_RE.finditer(norm):
        rel = m.group("rel").lower()
        if rel in ("vandaag", "today"):
            d = today
        elif rel in ("morgen", "tomorrow"):
            d = today + timedelta(days=1)
        else:
            d = today + timedelta(days=2)
        add("date", d.isoformat(), m.group(0), m.start(), m.end(), 0.9)

    entities.sort(key=lambda e: e.start)
    return entities
