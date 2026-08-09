"""Conservative deterministic ITN for local Qwen3-ASR service.

V4 scope:
- Chinese cardinal integers and decimals -> Arabic digits
- English cardinal integers and decimals -> Arabic digits
- Optional negative sign (负 / minus / negative)
- Percent / per-mille / per-myriad expressions
- Common currency names -> ISO 4217 currency code + normalized amount
- Common physical / engineering / chemistry / geometry units -> canonical symbols
- Explicit dates, clock times, and common durations -> canonical notation

Intentionally NOT handled in V4:
- phone numbers and general-purpose ordinals (date ordinals are supported)
- relative-date resolution/calendar arithmetic/time-zone conversion
- currency subunits (cents/pence/fen), exchange-rate reasoning, arithmetic
- punctuation insertion or semantic rewriting

The goal is to keep ITN narrowly scoped, deterministic, and auditable.
"""
from __future__ import annotations

import re
from typing import Iterable


# ------------------------- Chinese -------------------------
_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "幺": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_BIG_UNITS = {"万": 10_000, "亿": 100_000_000}
_CN_NUM_CHARS = "零〇一幺二两三四五六七八九十百千万亿点负"
_CN_RE = re.compile(rf"[{_CN_NUM_CHARS}]+")


def _cn_digit_sequence(token: str) -> str | None:
    """Return a spoken digit sequence without discarding leading zeroes."""
    if not token or not all(ch in _CN_DIGITS for ch in token):
        return None
    return "".join(str(_CN_DIGITS[ch]) for ch in token)


def _cn_positional_sequence(token: str) -> str | None:
    digits = _cn_digit_sequence(token)
    if digits is not None:
        return digits

    # ASR may collapse a final “一零” in a long digit sequence into “十”.
    # Retain the compatibility rule while keeping any leading zeroes.
    serial_ten = re.fullmatch(r"([零〇一幺二两三四五六七八九]{2,})十", token)
    if serial_ten is None:
        return None
    prefix = "".join(str(_CN_DIGITS[ch]) for ch in serial_ten.group(1))
    return f"{prefix}10"


def _cn_integer(token: str) -> int | None:
    if not token:
        return None

    # Reject unit-leading forms except 十/百/千, which are valid colloquial numerals.
    # This avoids turning idioms such as “万一” into a number.
    if token[0] in _CN_BIG_UNITS:
        return None

    has_unit = any(ch in _CN_SMALL_UNITS or ch in _CN_BIG_UNITS for ch in token)

    if not has_unit:
        # Digit-by-digit form: 二零二六 -> 2026
        digits = _cn_digit_sequence(token)
        if digits is not None:
            return int(digits)
        return None

    # ASR output may contain a digit-by-digit sequence that happens to end in
    # 十, for example “一二三四五六七八九十”.  This is not the cardinal
    # expression 九十: treating it as one would repeatedly overwrite the
    # previous digits and produce 90.  Preserve the spoken sequence instead.
    # Require at least two leading digits so ordinary numerals such as 二十 and
    # 一百二十 continue through the cardinal-number parser below.
    serial_ten = _cn_positional_sequence(token)
    if serial_ten is not None:
        return int(serial_ten)

    total = 0
    section = 0
    number = 0
    seen_any = False

    for ch in token:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
            seen_any = True
            continue

        if ch in _CN_SMALL_UNITS:
            unit = _CN_SMALL_UNITS[ch]
            # 十 / 百 / 千 may imply leading one.
            if number == 0:
                number = 1
            section += number * unit
            number = 0
            seen_any = True
            continue

        if ch in _CN_BIG_UNITS:
            big = _CN_BIG_UNITS[ch]
            section += number
            number = 0
            if section == 0:
                return None
            total += section * big
            section = 0
            seen_any = True
            continue

        return None

    if not seen_any:
        return None
    return total + section + number


def _cn_number(token: str) -> str | None:
    if not token:
        return None

    negative = token.startswith("负")
    if negative:
        token = token[1:]
    if not token:
        return None

    # Reject a dangling decimal point, e.g. “一点问题” should not become “1.问题”.
    if "点" in token:
        if token.count("点") != 1:
            return None
        left, right = token.split("点", 1)
        if not right or not all(ch in _CN_DIGITS for ch in right):
            return None
        left_value: int | str | None
        if not left:
            left_value = 0
        else:
            # A unitless sequence is a positional reading. Preserve explicitly
            # spoken leading zeroes instead of round-tripping it through int.
            left_value = _cn_positional_sequence(left)
            if left_value is None:
                left_value = _cn_integer(left)
        if left_value is None:
            return None
        right_digits = "".join(str(_CN_DIGITS[ch]) for ch in right)
        value = f"{left_value}.{right_digits}"
    else:
        # Unitless Chinese numerals are already interpreted digit by digit.
        # Keep the string form here so 零幺二 becomes 012 rather than 12.
        value = _cn_positional_sequence(token)
        if value is None:
            value_int = _cn_integer(token)
            if value_int is None:
                return None
            value = str(value_int)

    return f"-{value}" if negative else value


def normalize_chinese_numbers(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        start, end = match.span()
        # Keep ratio/percentile/percentage-point grammar intact here. Dedicated
        # ratio ITN handles 百分之/千分之/万分之 before this generic pass, while
        # words such as 百分位、千分位、百分点 must not become 100分位/1000分位.
        if any(text[end:].startswith(suffix) for suffix in ("分之", "分位", "分点", "分比")):
            return token
        converted = _cn_number(token)
        if converted is None:
            return token

        # Conservative guard for a single Chinese digit embedded directly inside
        # an ordinary Han word (e.g. 一旦 / 一方面). Leave it alone.
        core = token[1:] if token.startswith("负") else token
        core_no_point = core.replace("点", "")
        is_single_digit = len(core_no_point) == 1 and core_no_point in _CN_DIGITS and "点" not in core
        if is_single_digit:
            prev = text[start - 1] if start > 0 else ""
            nxt = text[end] if end < len(text) else ""
            # Convert isolated single digits or those adjacent to ASCII/digits/punctuation,
            # but not those glued to a normal CJK word.
            if (prev and "\u4e00" <= prev <= "\u9fff") or (nxt and "\u4e00" <= nxt <= "\u9fff"):
                return token
        return converted

    return _CN_RE.sub(repl, text)


# ------------------------- English -------------------------
_EN_ONES = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_EN_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_EN_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_EN_NUM_WORDS = set(_EN_ONES) | set(_EN_TENS) | set(_EN_SCALES) | {"and", "point", "minus", "negative"}
_EN_TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?|\s+|[^A-Za-z\s]+")


def _split_en_word(word: str) -> list[str]:
    return [p for p in word.lower().split("-") if p]


def _en_integer(words: Iterable[str]) -> int | None:
    total = 0
    current = 0
    seen = False

    for w in words:
        if w == "and":
            continue
        if w in _EN_ONES:
            current += _EN_ONES[w]
            seen = True
        elif w in _EN_TENS:
            current += _EN_TENS[w]
            seen = True
        elif w == "hundred":
            current = max(1, current) * 100
            seen = True
        elif w in ("thousand", "million", "billion"):
            scale = _EN_SCALES[w]
            total += max(1, current) * scale
            current = 0
            seen = True
        else:
            return None

    return total + current if seen else None


def _en_number(words: list[str]) -> str | None:
    if not words:
        return None

    negative = words[0] in ("minus", "negative")
    if negative:
        words = words[1:]
    if not words:
        return None

    if "point" in words:
        if words.count("point") != 1:
            return None
        idx = words.index("point")
        left_words = words[:idx]
        right_words = words[idx + 1 :]
        if not right_words:
            return None
        left = _en_integer(left_words) if left_words else 0
        if left is None:
            return None
        digits: list[str] = []
        for w in right_words:
            if w not in _EN_ONES or _EN_ONES[w] > 9:
                return None
            digits.append(str(_EN_ONES[w]))
        value = f"{left}.{''.join(digits)}"
    else:
        value_int = _en_integer(words)
        if value_int is None:
            return None
        value = str(value_int)

    return f"-{value}" if negative else value


def normalize_english_numbers(text: str) -> str:
    # Tokenize while preserving spaces/punctuation. We consume contiguous runs of
    # number words separated only by whitespace.
    tokens = _EN_TOKEN_RE.findall(text)
    out: list[str] = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]
        if not re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)?", tok):
            out.append(tok)
            i += 1
            continue

        parts = _split_en_word(tok)
        if not parts or not all(p in _EN_NUM_WORDS for p in parts):
            out.append(tok)
            i += 1
            continue

        words: list[str] = []
        consumed_tokens: list[str] = []
        j = i
        while j < len(tokens):
            word_tok = tokens[j]
            if not re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)?", word_tok):
                break
            word_parts = _split_en_word(word_tok)
            if not word_parts or not all(p in _EN_NUM_WORDS for p in word_parts):
                break
            words.extend(word_parts)
            consumed_tokens.append(word_tok)
            j += 1
            if j < len(tokens) and tokens[j].isspace():
                consumed_tokens.append(tokens[j])
                j += 1
            else:
                break

        # Drop trailing whitespace from the candidate and preserve it after conversion.
        trailing_ws = ""
        if consumed_tokens and consumed_tokens[-1].isspace():
            trailing_ws = consumed_tokens.pop()
            j -= 1

        converted = _en_number(words)
        if converted is None:
            out.extend(consumed_tokens)
            if trailing_ws:
                out.append(trailing_ws)
            i = j + (1 if trailing_ws else 0)
            continue

        out.append(converted)
        if trailing_ws:
            out.append(trailing_ws)
            j += 1
        i = j

    return "".join(out)



# ------------------------- Date / time -------------------------
# V4 adds deterministic calendar/time ITN.  The rules intentionally do NOT
# resolve relative dates (今天/明天/yesterday/tomorrow) against the system clock;
# they only normalize explicit calendar fields and clock expressions.
#
# Canonical forms:
# - full dates: YYYY-MM-DD
# - Chinese month/day without a year: M月D日 (do not invent a year)
# - clock time: HH:MM[:SS] in 24-hour notation when a daypart/AM/PM is explicit
# - durations: compact SI-like h/min/s/ms/μs/ns/d/wk notation

_CN_CAL_NUM_CHARS = "零〇一幺二两三四五六七八九十百千万"
_CN_CAL_NUM = rf"(?:\d{{1,4}}|[{_CN_CAL_NUM_CHARS}]+)"
_CN_TIME_NUM = rf"(?:\d{{1,2}}|[零〇一幺二两三四五六七八九十百]+)"


def _parse_cn_int(token: str) -> int | None:
    if not token:
        return None
    if token.isdigit():
        return int(token)
    return _cn_integer(token)


def _valid_ymd(year: int, month: int, day: int) -> bool:
    if not (1 <= year <= 9999 and 1 <= month <= 12 and 1 <= day <= 31):
        return False
    # Avoid importing calendar/datetime in the hot path: the month/day table is
    # sufficient, with the ordinary Gregorian leap-year rule for February.
    mdays = [31, 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28,
             31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return day <= mdays[month - 1]


def _valid_month_day(month: int, day: int) -> bool:
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return False
    # Without a year, permit Feb 29 but reject impossible dates such as Feb 30.
    mdays = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return day <= mdays[month - 1]


_CN_FULL_DATE_RE = re.compile(
    rf"(?P<year>{_CN_CAL_NUM})年\s*(?P<month>{_CN_CAL_NUM})月\s*(?P<day>{_CN_CAL_NUM})(?:日|号)"
)
_CN_YEAR_MONTH_RE = re.compile(rf"(?P<year>{_CN_CAL_NUM})年\s*(?P<month>{_CN_CAL_NUM})月")
_CN_MONTH_DAY_RE = re.compile(rf"(?P<month>{_CN_CAL_NUM})月\s*(?P<day>{_CN_CAL_NUM})(?:日|号)")


def normalize_chinese_dates(text: str) -> str:
    def full_date(match: re.Match[str]) -> str:
        year = _parse_cn_int(match.group("year"))
        month = _parse_cn_int(match.group("month"))
        day = _parse_cn_int(match.group("day"))
        if year is None or month is None or day is None or not _valid_ymd(year, month, day):
            return match.group(0)
        return f"{year:04d}-{month:02d}-{day:02d}"

    text = _CN_FULL_DATE_RE.sub(full_date, text)

    def year_month(match: re.Match[str]) -> str:
        year = _parse_cn_int(match.group("year"))
        month = _parse_cn_int(match.group("month"))
        if year is None or month is None or not (1 <= year <= 9999 and 1 <= month <= 12):
            return match.group(0)
        return f"{year:04d}-{month:02d}"

    text = _CN_YEAR_MONTH_RE.sub(year_month, text)

    def month_day(match: re.Match[str]) -> str:
        month = _parse_cn_int(match.group("month"))
        day = _parse_cn_int(match.group("day"))
        if month is None or day is None or not _valid_month_day(month, day):
            return match.group(0)
        return f"{month}月{day}日"

    return _CN_MONTH_DAY_RE.sub(month_day, text)


_CN_DAYPARTS = "凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|夜里|夜间"
_CN_DAYPART_RE = rf"(?P<period>{_CN_DAYPARTS})?"


def _apply_cn_daypart(hour: int, period: str | None) -> int | None:
    if period is None:
        return hour if 0 <= hour <= 23 else None
    if not (1 <= hour <= 12):
        return None
    if period in ("凌晨", "早上", "早晨", "上午"):
        return 0 if hour == 12 else hour
    if period == "中午":
        if hour == 12:
            return 12
        if 1 <= hour <= 10:
            return hour + 12
        return hour  # 11点/11时 stays 11:xx
    if period in ("下午", "傍晚"):
        return hour if hour == 12 else hour + 12
    if period in ("晚上", "夜里", "夜间"):
        return 0 if hour == 12 else hour + 12
    return None


def _format_clock(hour: int, minute: int = 0, second: int | None = None) -> str | None:
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    if second is not None and not (0 <= second <= 59):
        return None
    if second is None:
        return f"{hour:02d}:{minute:02d}"
    return f"{hour:02d}:{minute:02d}:{second:02d}"


# Most-specific patterns first.
_CN_TIME_HMS_RE = re.compile(
    rf"{_CN_DAYPART_RE}\s*(?P<hour>{_CN_TIME_NUM})(?:点|时)\s*"
    rf"(?P<minute>{_CN_TIME_NUM})分\s*(?P<second>{_CN_TIME_NUM})秒(?:钟)?"
)
_CN_TIME_HM_RE = re.compile(
    rf"{_CN_DAYPART_RE}\s*(?P<hour>{_CN_TIME_NUM})(?:点|时)\s*(?P<minute>{_CN_TIME_NUM})分(?:钟)?"
)
_CN_TIME_HALF_RE = re.compile(
    rf"{_CN_DAYPART_RE}\s*(?P<hour>{_CN_TIME_NUM})(?:点|时)\s*(?P<fraction>半|一刻|三刻)"
)
_CN_TIME_EXPLICIT_HOUR_RE = re.compile(
    rf"{_CN_DAYPART_RE}\s*(?P<hour>{_CN_TIME_NUM})(?:点|时)(?P<suffix>钟|整)"
)
# Without 分, minute digits are accepted only when an explicit daypart is
# present (下午三点零五) to avoid treating decimal speech 三点零五 as a clock.
_CN_TIME_DAYPART_COMPACT_RE = re.compile(
    rf"(?P<period>{_CN_DAYPARTS})\s*(?P<hour>{_CN_TIME_NUM})(?:点|时)\s*(?P<minute>{_CN_TIME_NUM})(?![分秒摄氏华氏])"
)
_CN_TIME_DAYPART_HOUR_RE = re.compile(
    rf"(?P<period>{_CN_DAYPARTS})\s*(?P<hour>{_CN_TIME_NUM})(?:点|时)(?![零〇一幺二两三四五六七八九十百点])"
)
_CN_TIME_DAYPART_COLON_RE = re.compile(
    rf"(?P<period>{_CN_DAYPARTS})\s*(?P<hour>\d{{1,2}}):(?P<minute>\d{{1,2}})(?::(?P<second>\d{{1,2}}))?"
)


def _replace_cn_time(match: re.Match[str], *, minute: int | None = None,
                     second: int | None = None) -> str:
    raw_hour = _parse_cn_int(match.group("hour"))
    if raw_hour is None:
        return match.group(0)
    period = match.groupdict().get("period")
    hour = _apply_cn_daypart(raw_hour, period)
    if hour is None:
        return match.group(0)
    if minute is None:
        minute_token = match.groupdict().get("minute")
        minute = _parse_cn_int(minute_token) if minute_token is not None else 0
    if minute is None:
        return match.group(0)
    if second is None and "second" in match.groupdict() and match.group("second") is not None:
        second = _parse_cn_int(match.group("second"))
        if second is None:
            return match.group(0)
    formatted = _format_clock(hour, minute, second)
    return formatted if formatted is not None else match.group(0)


def normalize_chinese_times(text: str) -> str:
    text = _CN_TIME_DAYPART_COLON_RE.sub(lambda m: _replace_cn_time(m), text)
    text = _CN_TIME_HMS_RE.sub(lambda m: _replace_cn_time(m), text)
    text = _CN_TIME_HM_RE.sub(lambda m: _replace_cn_time(m), text)

    def fraction(match: re.Match[str]) -> str:
        minutes = {"半": 30, "一刻": 15, "三刻": 45}[match.group("fraction")]
        return _replace_cn_time(match, minute=minutes)

    text = _CN_TIME_HALF_RE.sub(fraction, text)
    text = _CN_TIME_EXPLICIT_HOUR_RE.sub(lambda m: _replace_cn_time(m, minute=0), text)
    text = _CN_TIME_DAYPART_COMPACT_RE.sub(lambda m: _replace_cn_time(m), text)
    text = _CN_TIME_DAYPART_HOUR_RE.sub(lambda m: _replace_cn_time(m, minute=0), text)
    return text


# English calendar names and ordinals.  Date-specific ordinal parsing is kept
# separate from the generic number normalizer because "August eighth" is a
# calendar expression, while arbitrary ordinal rewriting is intentionally out of
# scope for this ITN layer.
_EN_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_EN_MONTH_CANON = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
_EN_MONTH_ALT = "|".join(sorted(map(re.escape, _EN_MONTHS), key=len, reverse=True))

_EN_ORDINAL_DAY_ALIASES: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}
for _base_word, _base_value in (("twenty", 20), ("thirty", 30)):
    for _word, _value in list(_EN_ORDINAL_DAY_ALIASES.items()):
        if 1 <= _value <= 9 and _base_value + _value <= 31:
            _EN_ORDINAL_DAY_ALIASES[f"{_base_word} {_word}"] = _base_value + _value
            _EN_ORDINAL_DAY_ALIASES[f"{_base_word}-{_word}"] = _base_value + _value
_EN_ORDINAL_DAY_ALT = "|".join(sorted(map(re.escape, _EN_ORDINAL_DAY_ALIASES), key=len, reverse=True))
_EN_DAY_TOKEN = rf"(?:\d{{1,2}}(?:st|nd|rd|th)?|{_EN_ORDINAL_DAY_ALT})"
_EN_YEAR_WORD = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|and)"
_EN_YEAR_START = r"(?:nineteen|twenty|one|two)"
_EN_YEAR_TOKEN = rf"(?:\d{{4}}|{_EN_YEAR_START}(?:[ -]+{_EN_YEAR_WORD}){{0,5}})"


def _parse_en_day(token: str) -> int | None:
    token = token.lower().strip()
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?", token)
    if m:
        day = int(m.group(1))
        return day if 1 <= day <= 31 else None
    return _EN_ORDINAL_DAY_ALIASES.get(token)


def _parse_en_year(token: str) -> int | None:
    token = re.sub(r"\s+", " ", token.lower().replace("-", " ")).strip()
    if token.isdigit():
        value = int(token)
        return value if 1000 <= value <= 9999 else None
    words = token.split()
    if not words:
        return None
    # Common spoken-year form: nineteen ninety nine / twenty twenty six /
    # twenty oh five.
    if words[0] in ("nineteen", "twenty") and len(words) >= 2:
        century = 19 if words[0] == "nineteen" else 20
        tail_words = words[1:]
        if tail_words and tail_words[0] in ("oh", "zero"):
            tail_words = tail_words[1:]
            tail = _en_integer(tail_words) if tail_words else 0
        else:
            tail = _en_integer(tail_words)
        if tail is not None and 0 <= tail <= 99:
            return century * 100 + tail
    value = _en_integer(words)
    if value is not None and 1000 <= value <= 9999:
        return value
    return None


_EN_DATE_MONTH_FIRST_RE = re.compile(
    rf"\b(?P<month>{_EN_MONTH_ALT})\s+(?:the\s+)?(?P<day>{_EN_DAY_TOKEN})"
    rf"(?:\s*,?\s*(?P<year>{_EN_YEAR_TOKEN}))?\b",
    re.I,
)
_EN_DATE_DAY_FIRST_RE = re.compile(
    rf"\b(?:the\s+)?(?P<day>{_EN_DAY_TOKEN})(?:\s+of)?\s+(?P<month>{_EN_MONTH_ALT})"
    rf"(?:\s*,?\s*(?P<year>{_EN_YEAR_TOKEN}))?\b",
    re.I,
)


def normalize_english_dates(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        month = _EN_MONTHS[match.group("month").lower()]
        day = _parse_en_day(match.group("day"))
        if day is None:
            return match.group(0)
        year_token = match.groupdict().get("year")
        if year_token:
            year = _parse_en_year(year_token)
            if year is None or not _valid_ymd(year, month, day):
                return match.group(0)
            return f"{year:04d}-{month:02d}-{day:02d}"
        if not _valid_month_day(month, day):
            return match.group(0)
        return f"{_EN_MONTH_CANON[month]} {day}"

    text = _EN_DATE_MONTH_FIRST_RE.sub(repl, text)
    text = _EN_DATE_DAY_FIRST_RE.sub(repl, text)
    return text


# Generate English 0..59 number phrases used by clock rules.
def _en_small_word(n: int) -> str:
    inv_ones = {v: k for k, v in _EN_ONES.items() if k != "oh" and 0 <= v <= 19}
    inv_tens = {v: k for k, v in _EN_TENS.items()}
    if n <= 19:
        return inv_ones[n]
    tens, ones = divmod(n, 10)
    return inv_tens[tens * 10] if ones == 0 else f"{inv_tens[tens * 10]} {inv_ones[ones]}"


_EN_SMALL_ALIASES: dict[str, int] = {}
for _n in range(60):
    _phrase = _en_small_word(_n)
    _EN_SMALL_ALIASES[_phrase] = _n
    _EN_SMALL_ALIASES[_phrase.replace(" ", "-")] = _n
    if 0 <= _n <= 9:
        _EN_SMALL_ALIASES[f"oh {_phrase}"] = _n
_EN_SMALL_ALT = "|".join(sorted(map(re.escape, _EN_SMALL_ALIASES), key=len, reverse=True))
_EN_CLOCK_NUM = rf"(?:\d{{1,2}}|{_EN_SMALL_ALT})"
_EN_AMPM = r"(?:a\.?m\.?|p\.?m\.?)"


def _parse_en_small(token: str) -> int | None:
    token = re.sub(r"\s+", " ", token.lower().replace("-", " ")).strip()
    if token.isdigit():
        value = int(token)
        return value if 0 <= value <= 59 else None
    if token.startswith("oh "):
        token = token[3:]
    # normalize the generated space form
    return _EN_SMALL_ALIASES.get(token)


def _apply_ampm(hour: int, ampm: str | None) -> int | None:
    if ampm is None:
        return hour if 0 <= hour <= 23 else None
    if not 1 <= hour <= 12:
        return None
    marker = re.sub(r"[^apm]", "", ampm.lower())
    if marker == "am":
        return 0 if hour == 12 else hour
    if marker == "pm":
        return 12 if hour == 12 else hour + 12
    return None


_EN_NUMERIC_COLON_TIME_RE = re.compile(
    rf"\b(?P<hour>\d{{1,2}}):(?P<minute>\d{{1,2}})(?::(?P<second>\d{{1,2}}))?\s*(?P<ampm>{_EN_AMPM})?(?![A-Za-z])",
    re.I,
)
_EN_TIME_WITH_AMPM_RE = re.compile(
    rf"\b(?P<hour>{_EN_CLOCK_NUM})\s+(?P<minute>{_EN_CLOCK_NUM})\s*(?P<ampm>{_EN_AMPM})(?![A-Za-z])",
    re.I,
)
_EN_HOUR_AMPM_RE = re.compile(rf"\b(?P<hour>{_EN_CLOCK_NUM})\s*(?P<ampm>{_EN_AMPM})(?![A-Za-z])", re.I)
_EN_OCLOCK_RE = re.compile(rf"\b(?P<hour>{_EN_CLOCK_NUM})\s+o['’]?clock(?:\s*(?P<ampm>{_EN_AMPM}))?(?![A-Za-z])", re.I)
_EN_HALF_PAST_RE = re.compile(rf"\bhalf\s+past\s+(?P<hour>{_EN_CLOCK_NUM})\b", re.I)
_EN_QUARTER_PAST_RE = re.compile(rf"\b(?:a\s+)?quarter\s+past\s+(?P<hour>{_EN_CLOCK_NUM})\b", re.I)
_EN_QUARTER_TO_RE = re.compile(rf"\b(?:a\s+)?quarter\s+to\s+(?P<hour>{_EN_CLOCK_NUM})\b", re.I)
_EN_DAYPART_TIME_RE = re.compile(
    rf"\b(?P<hour>{_EN_CLOCK_NUM})(?:\s+(?P<minute>{_EN_CLOCK_NUM}))?\s+in\s+the\s+"
    rf"(?P<period>morning|afternoon|evening|night)\b",
    re.I,
)
_EN_AT_NIGHT_TIME_RE = re.compile(rf"\b(?P<hour>{_EN_CLOCK_NUM})(?:\s+(?P<minute>{_EN_CLOCK_NUM}))?\s+at\s+night\b", re.I)
_EN_CONTEXT_TIME_RE = re.compile(
    rf"\b(?P<prefix>at|around|about|by|from|until)\s+(?P<hour>{_EN_CLOCK_NUM})\s+(?P<minute>{_EN_CLOCK_NUM})\b",
    re.I,
)
_EN_NOON_RE = re.compile(r"\bnoon\b", re.I)
_EN_MIDNIGHT_RE = re.compile(r"\bmidnight\b", re.I)


def _replace_en_clock(match: re.Match[str], *, minute: int | None = None,
                      second: int | None = None, ampm: str | None = None) -> str:
    hour = _parse_en_small(match.group("hour"))
    if hour is None:
        return match.group(0)
    if minute is None:
        minute_token = match.groupdict().get("minute")
        minute = _parse_en_small(minute_token) if minute_token is not None else 0
    if minute is None:
        return match.group(0)
    if second is None and match.groupdict().get("second") is not None:
        second = _parse_en_small(match.group("second"))
        if second is None:
            return match.group(0)
    ampm = ampm if ampm is not None else match.groupdict().get("ampm")
    hour24 = _apply_ampm(hour, ampm)
    if hour24 is None:
        return match.group(0)
    formatted = _format_clock(hour24, minute, second)
    return formatted if formatted is not None else match.group(0)


def normalize_english_times(text: str) -> str:
    text = _EN_NUMERIC_COLON_TIME_RE.sub(lambda m: _replace_en_clock(m), text)
    text = _EN_OCLOCK_RE.sub(lambda m: _replace_en_clock(m, minute=0), text)
    text = _EN_TIME_WITH_AMPM_RE.sub(lambda m: _replace_en_clock(m), text)
    text = _EN_HOUR_AMPM_RE.sub(lambda m: _replace_en_clock(m, minute=0), text)
    text = _EN_HALF_PAST_RE.sub(lambda m: _replace_en_clock(m, minute=30), text)
    text = _EN_QUARTER_PAST_RE.sub(lambda m: _replace_en_clock(m, minute=15), text)

    def quarter_to(match: re.Match[str]) -> str:
        hour = _parse_en_small(match.group("hour"))
        if hour is None or not (1 <= hour <= 23):
            return match.group(0)
        hour = (hour - 1) % 24
        formatted = _format_clock(hour, 45)
        return formatted if formatted is not None else match.group(0)

    text = _EN_QUARTER_TO_RE.sub(quarter_to, text)

    def daypart(match: re.Match[str]) -> str:
        period = match.group("period").lower()
        hour = _parse_en_small(match.group("hour"))
        minute = _parse_en_small(match.group("minute")) if match.group("minute") else 0
        if hour is None or minute is None or not (1 <= hour <= 12):
            return match.group(0)
        if period == "morning":
            hour24 = 0 if hour == 12 else hour
        elif period == "afternoon":
            hour24 = hour if hour == 12 else hour + 12
        elif period == "evening":
            hour24 = 12 if hour == 12 else hour + 12
        elif period == "night":
            hour24 = 0 if hour == 12 else (hour if hour <= 5 else hour + 12)
        else:
            return match.group(0)
        formatted = _format_clock(hour24, minute)
        return formatted if formatted is not None else match.group(0)

    text = _EN_DAYPART_TIME_RE.sub(daypart, text)

    def at_night(match: re.Match[str]) -> str:
        hour = _parse_en_small(match.group("hour"))
        minute = _parse_en_small(match.group("minute")) if match.group("minute") else 0
        if hour is None or minute is None or not (1 <= hour <= 12):
            return match.group(0)
        hour24 = 0 if hour == 12 else (hour if hour <= 5 else hour + 12)
        formatted = _format_clock(hour24, minute)
        return formatted if formatted is not None else match.group(0)

    text = _EN_AT_NIGHT_TIME_RE.sub(at_night, text)

    def context_time(match: re.Match[str]) -> str:
        converted = _replace_en_clock(match)
        if converted == match.group(0):
            return match.group(0)
        return f"{match.group('prefix')} {converted}"

    text = _EN_CONTEXT_TIME_RE.sub(context_time, text)
    text = _EN_NOON_RE.sub("12:00", text)
    text = _EN_MIDNIGHT_RE.sub("00:00", text)
    return text


# Duration expressions are part of time ITN but do not attempt calendar
# arithmetic.  Months/years are left in natural-language calendar notation.
_CN_DURATION_ALIASES = {
    "纳秒": "ns", "微秒": "μs", "毫秒": "ms", "秒钟": "s", "秒": "s",
    "分钟": "min", "分种": "min", "小时": "h", "钟头": "h",
    "天": "d", "星期": "wk", "周": "wk",
}
_CN_DURATION_ALT = "|".join(sorted(map(re.escape, _CN_DURATION_ALIASES), key=len, reverse=True))
_CN_DURATION_RAW_RE = re.compile(rf"(?P<num>负?[{_CN_NUM_CHARS}]+|[+-]?\d+(?:\.\d+)?)\s*(?:个)?\s*(?P<unit>{_CN_DURATION_ALT})")

_EN_DURATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bnanoseconds?\b", re.I), "ns"),
    (re.compile(r"\bmicroseconds?\b", re.I), "μs"),
    (re.compile(r"\bmilliseconds?\b", re.I), "ms"),
    (re.compile(r"\bseconds?\b", re.I), "s"),
    (re.compile(r"\bminutes?\b", re.I), "min"),
    (re.compile(r"\bhours?\b", re.I), "h"),
    (re.compile(r"\bdays?\b", re.I), "d"),
    (re.compile(r"\bweeks?\b", re.I), "wk"),
)


def normalize_chinese_durations(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        raw = match.group("num")
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", raw):
            num = raw
        else:
            num = _cn_number(raw)
        if num is None:
            return match.group(0)
        symbol = _CN_DURATION_ALIASES[match.group('unit')]
        tail = match.string[match.end():]
        needs_sep = bool(tail and (tail[0].isdigit() or tail[0] == "负" or tail[0] in _CN_DIGITS))
        return f"{num} {symbol}{' ' if needs_sep else ''}"
    return _CN_DURATION_RAW_RE.sub(repl, text)


def normalize_english_durations(text: str) -> str:
    amount = r"(?P<amount>[+-]?\d+(?:\.\d+)?)"
    for pattern, symbol in _EN_DURATION_PATTERNS:
        combined = re.compile(amount + r"\s+" + pattern.pattern, pattern.flags)
        text = combined.sub(lambda m, s=symbol: f"{m.group('amount')} {s}", text)
    return text


def normalize_datetime_spacing(text: str) -> str:
    # When Chinese source text has no whitespace between date and time, the two
    # independent normalizers would otherwise produce 2026-08-0815:30.
    text = re.sub(r"(?<=\d{4}-\d{2}-\d{2})(?=\d{2}:\d{2})", " ", text)
    text = re.sub(r"(?<=日)(?=\d{2}:\d{2})", " ", text)
    text = re.sub(r"(?<=(?:今天|明天|后天|昨天|前天))(?=\d{2}:\d{2})", " ", text)
    return text


def normalize_dates_times_raw(text: str) -> str:
    text = normalize_chinese_dates(text)
    text = normalize_chinese_times(text)
    text = normalize_english_dates(text)
    text = normalize_english_times(text)
    text = normalize_chinese_durations(text)
    return normalize_datetime_spacing(text)


def normalize_dates_times(text: str) -> str:
    # Second pass after generic number normalization catches mixed forms such as
    # "2026年八月8日" and English durations whose number words are now digits.
    text = normalize_chinese_dates(text)
    text = normalize_chinese_times(text)
    text = normalize_english_dates(text)
    text = normalize_english_times(text)
    text = normalize_chinese_durations(text)
    text = normalize_english_durations(text)
    return normalize_datetime_spacing(text)

# ------------------------- Ratio-related measure words -------------------------
# Preserve semantics for percentage points / percentiles while still normalizing
# the spoken number in front.  These are NOT converted to %/‰/‱ because e.g.
# “3 percentage points” is not the same thing as “3 percent”.
_CN_RATIO_MEASURE_RE = re.compile(
    rf"(?P<num>负?[{_CN_NUM_CHARS}]+)(?P<suffix>个百分点|千分点|万分点|百分位|千分位|万分位)"
)


def normalize_chinese_ratio_measure_words(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        num = _cn_number(match.group("num"))
        if num is None:
            return match.group(0)
        return f"{num}{match.group('suffix')}"

    return _CN_RATIO_MEASURE_RE.sub(repl, text)


# ------------------------- Percent / per-mille / per-myriad -------------------------
# Keep the three scales distinct instead of converting everything to percent.
#  百分之  -> %
#  千分之  -> ‰  (U+2030 PER MILLE SIGN)
#  万分之  -> ‱  (U+2031 PER TEN THOUSAND SIGN)
_CN_RATIO_SCALE = {"百分之": "%", "千分之": "‰", "万分之": "‱"}
_CN_RATIO_RAW_RE = re.compile(
    rf"(?P<prefix>负)?(?P<scale>百分之|千分之|万分之)(?P<num>负?(?:[{_CN_NUM_CHARS}]+|[+-]?\d+(?:\.\d+)?))"
)
_CN_RATIO_DIGIT_PATTERNS = (
    (re.compile(r"(?P<prefix>负)?百分之(?P<num>[+-]?\d+(?:\.\d+)?)"), "%"),
    (re.compile(r"(?P<prefix>负)?千分之(?P<num>[+-]?\d+(?:\.\d+)?)"), "‰"),
    (re.compile(r"(?P<prefix>负)?万分之(?P<num>[+-]?\d+(?:\.\d+)?)"), "‱"),
)

_EN_RATIO_PATTERNS = (
    (re.compile(r"(?P<num>[+-]?\d+(?:\.\d+)?)\s*(?:percent|per\s+cent)\b", re.IGNORECASE), "%"),
    (re.compile(r"(?P<num>[+-]?\d+(?:\.\d+)?)\s*(?:per\s*mille|permille|per\s*1000)\b", re.IGNORECASE), "‰"),
    (re.compile(r"(?P<num>[+-]?\d+(?:\.\d+)?)\s*(?:per\s*myriad|permyriad|per\s*ten\s*thousand|per\s*10000)\b", re.IGNORECASE), "‱"),
)


def _normalize_cn_ratio_raw(match: re.Match[str]) -> str:
    raw_num = match.group("num")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", raw_num):
        num = raw_num
    else:
        num = _cn_number(raw_num)
        if num is None:
            return match.group(0)

    if match.group("prefix") == "负" and not num.startswith("-"):
        num = "-" + num.lstrip("+")
    return f"{num}{_CN_RATIO_SCALE[match.group('scale')]}"


def normalize_chinese_ratios_raw(text: str) -> str:
    """Normalize spoken Chinese ratio expressions before generic number ITN.

    This pre-pass is important for single-digit numerators such as 百分之一:
    the generic Chinese normalizer intentionally leaves an isolated 一 alone
    when it is attached to Han text, while here the 分之 grammar makes it
    unambiguously numeric.
    """
    return _CN_RATIO_RAW_RE.sub(_normalize_cn_ratio_raw, text)


def _apply_signed_ratio(match: re.Match[str], symbol: str) -> str:
    num = match.group("num")
    prefix = match.groupdict().get("prefix")
    if prefix == "负" and not num.startswith("-"):
        num = "-" + num.lstrip("+")
    return f"{num}{symbol}"


def normalize_ratios(text: str) -> str:
    # Second pass catches already-numeric Chinese and English forms.
    for pattern, symbol in _CN_RATIO_DIGIT_PATTERNS:
        text = pattern.sub(lambda m, s=symbol: _apply_signed_ratio(m, s), text)
    for pattern, symbol in _EN_RATIO_PATTERNS:
        text = pattern.sub(lambda m, s=symbol: f"{m.group('num')}{s}", text)
    return text


# ------------------------- Currency -------------------------
# Canonical output uses ISO 4217 codes rather than ambiguous symbols.  This is
# more useful to downstream software and avoids collisions such as CNY/JPY (¥)
# and the many currencies that use "$".
_CN_CURRENCY_ALIASES = {
    "人民币元": "CNY",
    "人民币": "CNY",
    "美元": "USD",
    "美金": "USD",
    "欧元": "EUR",
    "英镑": "GBP",
    "日元": "JPY",
    "日币": "JPY",
    "新加坡元": "SGD",
    "新币": "SGD",
    "港元": "HKD",
    "港币": "HKD",
    "澳大利亚元": "AUD",
    "澳元": "AUD",
    "澳币": "AUD",
    "加拿大元": "CAD",
    "加元": "CAD",
    "加币": "CAD",
    "瑞士法郎": "CHF",
    "韩国元": "KRW",
    "韩元": "KRW",
    "新台币": "TWD",
    "台湾元": "TWD",
    "台币": "TWD",
    "新西兰元": "NZD",
    "纽元": "NZD",
    "印度卢比": "INR",
    "俄罗斯卢布": "RUB",
    "卢布": "RUB",
    "泰铢": "THB",
    "马来西亚林吉特": "MYR",
    "林吉特": "MYR",
    "令吉": "MYR",
    "印度尼西亚卢比": "IDR",
    "印尼盾": "IDR",
    "菲律宾比索": "PHP",
    "越南盾": "VND",
    "阿联酋迪拉姆": "AED",
    "沙特里亚尔": "SAR",
    "瑞典克朗": "SEK",
    "挪威克朗": "NOK",
    "丹麦克朗": "DKK",
}

# Longest aliases first so e.g. 人民币元 is not partially consumed as 人民币.
_CN_CURRENCY_ALT = "|".join(sorted(map(re.escape, _CN_CURRENCY_ALIASES), key=len, reverse=True))
_CN_AMOUNT_BEFORE_CURRENCY_RE = re.compile(
    rf"(?P<amount>[+-]?\d+(?:\.\d+)?)\s*(?:块|元)?\s*(?P<currency>{_CN_CURRENCY_ALT})"
)
# Common explicit prefix form such as “人民币38元”.  A trailing 元/块 is
# required for RMB to avoid changing unrelated phrases such as “人民币38指数”.
_CN_CURRENCY_BEFORE_AMOUNT_RE = re.compile(
    rf"(?P<currency>{_CN_CURRENCY_ALT})\s*(?P<amount>[+-]?\d+(?:\.\d+)?)(?P<trailer>\s*(?:元|块))?"
)

_EN_CURRENCY_ALIASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:U\.?S\.?|US|United\s+States)\s+dollars?\b", re.I), "USD"),
    (re.compile(r"\beuros?\b", re.I), "EUR"),
    (re.compile(r"\b(?:British\s+pounds?|pounds?\s+sterling|sterling)\b", re.I), "GBP"),
    (re.compile(r"\b(?:Japanese\s+)?yen\b", re.I), "JPY"),
    (re.compile(r"\b(?:Chinese\s+)?(?:yuan|renminbi)\b", re.I), "CNY"),
    (re.compile(r"\bSingapore\s+dollars?\b", re.I), "SGD"),
    (re.compile(r"\bHong\s+Kong\s+dollars?\b", re.I), "HKD"),
    (re.compile(r"\bAustralian\s+dollars?\b", re.I), "AUD"),
    (re.compile(r"\bCanadian\s+dollars?\b", re.I), "CAD"),
    (re.compile(r"\bSwiss\s+francs?\b", re.I), "CHF"),
    (re.compile(r"\b(?:South\s+)?Korean\s+won\b", re.I), "KRW"),
    (re.compile(r"\b(?:New\s+Taiwan|Taiwan)\s+dollars?\b", re.I), "TWD"),
    (re.compile(r"\bNew\s+Zealand\s+dollars?\b", re.I), "NZD"),
    (re.compile(r"\bIndian\s+rupees?\b", re.I), "INR"),
    (re.compile(r"\bRussian\s+rubles?\b", re.I), "RUB"),
    (re.compile(r"\bThai\s+baht\b", re.I), "THB"),
    (re.compile(r"\bMalaysian\s+ringgit\b", re.I), "MYR"),
    (re.compile(r"\bIndonesian\s+rupiah\b", re.I), "IDR"),
    (re.compile(r"\bPhilippine\s+pesos?\b", re.I), "PHP"),
    (re.compile(r"\bVietnamese\s+dong\b", re.I), "VND"),
    (re.compile(r"\b(?:UAE|Emirati)\s+dirhams?\b", re.I), "AED"),
    (re.compile(r"\bSaudi\s+riyals?\b", re.I), "SAR"),
    (re.compile(r"\bSwedish\s+krona\b", re.I), "SEK"),
    (re.compile(r"\bNorwegian\s+krone\b", re.I), "NOK"),
    (re.compile(r"\bDanish\s+krone\b", re.I), "DKK"),
]


_CN_RAW_AMOUNT_BEFORE_CURRENCY_RE = re.compile(
    rf"(?P<amount>负?[{_CN_NUM_CHARS}]+)\s*(?:块|元)?\s*(?P<currency>{_CN_CURRENCY_ALT})"
)
_CN_RAW_CURRENCY_BEFORE_AMOUNT_RE = re.compile(
    rf"(?P<currency>{_CN_CURRENCY_ALT})\s*(?P<amount>负?[{_CN_NUM_CHARS}]+)(?P<trailer>\s*(?:元|块))?"
)


def normalize_chinese_currencies_raw(text: str) -> str:
    """Normalize Chinese-spoken currency amounts before generic number ITN.

    Needed for single-digit forms such as 三美元 / 三英镑, which the generic
    Chinese number pass intentionally leaves alone when glued to Han text.
    """
    def amount_before(match: re.Match[str]) -> str:
        amount = _cn_number(match.group("amount"))
        if amount is None:
            return match.group(0)
        code = _CN_CURRENCY_ALIASES[match.group("currency")]
        return f"{code} {amount}"

    text = _CN_RAW_AMOUNT_BEFORE_CURRENCY_RE.sub(amount_before, text)

    def currency_before(match: re.Match[str]) -> str:
        amount = _cn_number(match.group("amount"))
        if amount is None:
            return match.group(0)
        currency = match.group("currency")
        trailer = match.group("trailer") or ""
        if _CN_CURRENCY_ALIASES[currency] == "CNY" and not trailer.strip():
            return match.group(0)
        return f"{_CN_CURRENCY_ALIASES[currency]} {amount}"

    return _CN_RAW_CURRENCY_BEFORE_AMOUNT_RE.sub(currency_before, text)


def normalize_chinese_currencies(text: str) -> str:
    def amount_before(match: re.Match[str]) -> str:
        code = _CN_CURRENCY_ALIASES[match.group("currency")]
        return f"{code} {match.group('amount')}"

    text = _CN_AMOUNT_BEFORE_CURRENCY_RE.sub(amount_before, text)

    def currency_before(match: re.Match[str]) -> str:
        currency = match.group("currency")
        trailer = match.group("trailer") or ""
        # RMB prefix forms should normally include 元/块.  For other explicit
        # currency names (“美元38”, “欧元20”) the name itself is unambiguous.
        if _CN_CURRENCY_ALIASES[currency] == "CNY" and not trailer.strip():
            return match.group(0)
        code = _CN_CURRENCY_ALIASES[currency]
        return f"{code} {match.group('amount')}"

    return _CN_CURRENCY_BEFORE_AMOUNT_RE.sub(currency_before, text)


def normalize_english_currencies(text: str) -> str:
    # Number-first English speech is the common ASR form: “38 US dollars”.
    # Bare “dollars” and bare “pounds” are intentionally not normalized because
    # they are ambiguous (currency jurisdiction / unit of weight).
    amount = r"(?P<amount>[+-]?\d+(?:\.\d+)?)"
    for currency_pattern, code in _EN_CURRENCY_ALIASES:
        combined = re.compile(amount + r"\s+" + currency_pattern.pattern, currency_pattern.flags)
        text = combined.sub(lambda m, c=code: f"{c} {m.group('amount')}", text)
    return text


def normalize_currencies(text: str) -> str:
    text = normalize_chinese_currencies(text)
    text = normalize_english_currencies(text)
    return text



# ------------------------- Measurement / scientific units -------------------------
# This layer is notation normalization, NOT physical unit conversion.  For example,
# “三十千帕” becomes “30 kPa”; it is not converted to 30000 Pa.  Canonical symbols
# are chosen to be compact, unambiguous, and convenient for downstream processing.
#
# Deliberately ambiguous bare forms are omitted where practical:
# - Chinese bare “度” is not forced to °C/° because it can mean angle,
#   electricity (kWh), alcohol strength, etc.  A temperature cue is handled
#   separately below and defaults to Celsius, which is the ordinary Chinese
#   convention when no scale is spoken.
# - English bare “pounds” is not forced to lb because it can also mean GBP.
# - English bare “gallons” is not normalized unless US/imperial is explicit.

# value: (canonical symbol, tight_join).  tight_join=True is used for temperature
# symbols such as 30°C; ordinary SI-style output uses a space, e.g. 5 kg / 10 N.
_CN_UNIT_ALIASES: dict[str, tuple[str, bool]] = {
    # Temperature
    "摄氏度": ("°C", True),
    "摄氏": ("°C", True),
    "℃": ("°C", True),
    "华氏度": ("°F", True),
    "华氏": ("°F", True),
    "℉": ("°F", True),
    "开尔文": ("K", False),
    "开氏度": ("K", False),

    # Mass / weight in everyday speech
    "公吨": ("t", False),
    "吨": ("t", False),
    "千克": ("kg", False),
    "公斤": ("kg", False),
    "克": ("g", False),
    "毫克": ("mg", False),
    "微克": ("μg", False),
    "纳克": ("ng", False),
    "皮克": ("pg", False),
    "磅": ("lb", False),
    "盎司": ("oz", False),
    "克拉": ("ct", False),

    # Length
    "公里": ("km", False),
    "千米": ("km", False),
    "分米": ("dm", False),
    "厘米": ("cm", False),
    "公分": ("cm", False),
    "毫米": ("mm", False),
    "公厘": ("mm", False),
    "微米": ("μm", False),
    "纳米": ("nm", False),
    "皮米": ("pm", False),
    "米": ("m", False),
    "英里": ("mi", False),
    "海里": ("nmi", False),
    "浬": ("nmi", False),
    "英尺": ("ft", False),
    "英寸": ("in", False),
    "英码": ("yd", False),

    # Area
    "平方公里": ("km²", False),
    "平方千米": ("km²", False),
    "平方分米": ("dm²", False),
    "平方米": ("m²", False),
    "平方厘米": ("cm²", False),
    "平方公分": ("cm²", False),
    "平方毫米": ("mm²", False),
    "平方微米": ("μm²", False),
    "平方英里": ("mi²", False),
    "平方英尺": ("ft²", False),
    "平方英寸": ("in²", False),
    "公顷": ("ha", False),
    "英亩": ("acre", False),
    "亩": ("亩", False),

    # Volume / capacity
    "立方公里": ("km³", False),
    "立方千米": ("km³", False),
    "立方米": ("m³", False),
    "立方分米": ("dm³", False),
    "立方厘米": ("cm³", False),
    "立方公分": ("cm³", False),
    "立方毫米": ("mm³", False),
    "公升": ("L", False),
    "升": ("L", False),
    "分升": ("dL", False),
    "厘升": ("cL", False),
    "毫升": ("mL", False),
    "微升": ("μL", False),
    "纳升": ("nL", False),
    "美制加仑": ("US gal", False),
    "英制加仑": ("imp gal", False),

    # Speed / acceleration / flow
    "千米每小时": ("km/h", False),
    "公里每小时": ("km/h", False),
    "米每秒": ("m/s", False),
    "厘米每秒": ("cm/s", False),
    "毫米每秒": ("mm/s", False),
    "英里每小时": ("mph", False),
    "海里每小时": ("kn", False),
    "节": ("kn", False),
    "米每秒平方": ("m/s²", False),
    "米每平方秒": ("m/s²", False),
    "米每二次方秒": ("m/s²", False),
    "英尺每秒平方": ("ft/s²", False),
    "升每分钟": ("L/min", False),
    "毫升每分钟": ("mL/min", False),
    "立方米每秒": ("m³/s", False),
    "立方米每小时": ("m³/h", False),

    # Force / torque
    "兆牛顿": ("MN", False),
    "千牛顿": ("kN", False),
    "千牛": ("kN", False),
    "毫牛顿": ("mN", False),
    "微牛顿": ("μN", False),
    "牛顿": ("N", False),
    "磅力": ("lbf", False),
    "千牛顿米": ("kN·m", False),
    "千牛米": ("kN·m", False),
    "牛顿米": ("N·m", False),
    "牛米": ("N·m", False),

    # Pressure
    "吉帕斯卡": ("GPa", False),
    "吉帕": ("GPa", False),
    "兆帕斯卡": ("MPa", False),
    "兆帕": ("MPa", False),
    "千帕斯卡": ("kPa", False),
    "千帕": ("kPa", False),
    "百帕": ("hPa", False),
    "帕斯卡": ("Pa", False),
    "帕": ("Pa", False),
    "毫巴": ("mbar", False),
    "巴": ("bar", False),
    "标准大气压": ("atm", False),
    "大气压": ("atm", False),
    "托尔": ("Torr", False),
    "毫米汞柱": ("mmHg", False),
    "厘米水柱": ("cmH₂O", False),
    "磅每平方英寸": ("psi", False),

    # Energy / heat / power
    "吉焦耳": ("GJ", False),
    "吉焦": ("GJ", False),
    "兆焦耳": ("MJ", False),
    "兆焦": ("MJ", False),
    "千焦耳": ("kJ", False),
    "千焦": ("kJ", False),
    "焦耳": ("J", False),
    "千卡路里": ("kcal", False),
    "千卡": ("kcal", False),
    "大卡": ("kcal", False),
    "卡路里": ("cal", False),
    "吉瓦时": ("GWh", False),
    "兆瓦时": ("MWh", False),
    "千瓦时": ("kWh", False),
    "度电": ("kWh", False),
    "瓦时": ("Wh", False),
    "吉电子伏特": ("GeV", False),
    "兆电子伏特": ("MeV", False),
    "千电子伏特": ("keV", False),
    "电子伏特": ("eV", False),
    "吉瓦特": ("GW", False),
    "吉瓦": ("GW", False),
    "兆瓦特": ("MW", False),
    "兆瓦": ("MW", False),
    "千瓦特": ("kW", False),
    "千瓦": ("kW", False),
    "毫瓦": ("mW", False),
    "瓦特": ("W", False),
    "瓦": ("W", False),
    "马力": ("hp", False),

    # Frequency
    "太赫兹": ("THz", False),
    "太赫": ("THz", False),
    "吉赫兹": ("GHz", False),
    "吉赫": ("GHz", False),
    "兆赫兹": ("MHz", False),
    "兆赫": ("MHz", False),
    "千赫兹": ("kHz", False),
    "千赫": ("kHz", False),
    "赫兹": ("Hz", False),

    # Electrical / magnetic / photometric / radiation
    "兆伏特": ("MV", False),
    "兆伏": ("MV", False),
    "千伏特": ("kV", False),
    "千伏": ("kV", False),
    "毫伏特": ("mV", False),
    "毫伏": ("mV", False),
    "伏特": ("V", False),
    "千安培": ("kA", False),
    "千安": ("kA", False),
    "毫安培": ("mA", False),
    "毫安": ("mA", False),
    "微安培": ("μA", False),
    "微安": ("μA", False),
    "安培": ("A", False),
    "兆欧姆": ("MΩ", False),
    "兆欧": ("MΩ", False),
    "千欧姆": ("kΩ", False),
    "千欧": ("kΩ", False),
    "欧姆": ("Ω", False),
    "库仑": ("C", False),
    "微法拉": ("μF", False),
    "微法": ("μF", False),
    "纳法拉": ("nF", False),
    "纳法": ("nF", False),
    "皮法拉": ("pF", False),
    "皮法": ("pF", False),
    "毫法拉": ("mF", False),
    "毫法": ("mF", False),
    "法拉": ("F", False),
    "毫亨利": ("mH", False),
    "毫亨": ("mH", False),
    "微亨利": ("μH", False),
    "微亨": ("μH", False),
    "亨利": ("H", False),
    "毫特斯拉": ("mT", False),
    "微特斯拉": ("μT", False),
    "特斯拉": ("T", False),
    "韦伯": ("Wb", False),
    "西门子": ("S", False),
    "流明": ("lm", False),
    "勒克斯": ("lx", False),
    "坎德拉": ("cd", False),
    "贝可勒尔": ("Bq", False),
    "戈瑞": ("Gy", False),
    "希沃特": ("Sv", False),

    # Chemistry / concentration / molar quantities
    "毫摩尔每升": ("mmol/L", False),
    "微摩尔每升": ("μmol/L", False),
    "纳摩尔每升": ("nmol/L", False),
    "摩尔每升": ("mol/L", False),
    "摩尔每千克": ("mol/kg", False),
    "千克每摩尔": ("kg/mol", False),
    "克每摩尔": ("g/mol", False),
    "微克每升": ("μg/L", False),
    "纳克每升": ("ng/L", False),
    "毫克每升": ("mg/L", False),
    "克每升": ("g/L", False),
    "微克每毫升": ("μg/mL", False),
    "毫克每毫升": ("mg/mL", False),
    "毫摩尔": ("mmol", False),
    "微摩尔": ("μmol", False),
    "纳摩尔": ("nmol", False),
    "摩尔": ("mol", False),
    "毫当量": ("mEq", False),
    "当量": ("Eq", False),

    # Density / common compound quantities
    "千克每立方米": ("kg/m³", False),
    "克每立方厘米": ("g/cm³", False),
    "克每毫升": ("g/mL", False),

    # Angle / solid angle (bare 度 intentionally omitted)
    "毫弧度": ("mrad", False),
    "弧度": ("rad", False),
    "球面度": ("sr", False),
}

# Longest unit names first.  The numeric part is intentionally lazy: this lets
# “三千克” be interpreted as 3 kg rather than consuming 千 as part of 三千 and
# then treating 克 as grams.  Both are physically equivalent, but preserving the
# spoken unit is the cleaner ITN behavior.  The same applies to 三千米 -> 3 km.
_CN_UNIT_ALT = "|".join(sorted(map(re.escape, _CN_UNIT_ALIASES), key=len, reverse=True))
_CN_UNIT_AMOUNT_RE = re.compile(
    rf"(?P<amount>负?(?:[{_CN_NUM_CHARS}]+?|[+-]?\d+(?:\.\d+)?))\s*(?P<unit>{_CN_UNIT_ALT})"
)

# Explicit temperature-prefix speech: 摄氏三十度 / 华氏零下四十度.
_CN_TEMP_PREFIX_RE = re.compile(
    rf"(?P<scale>摄氏|华氏)\s*(?P<below>零下)?\s*(?P<amount>负?(?:[{_CN_NUM_CHARS}]+|[+-]?\d+(?:\.\d+)?))\s*度"
)
# Number-first colloquial sub-zero form: 零下五摄氏度 / 零下四十华氏度.
_CN_TEMP_BELOW_RE = re.compile(
    rf"零下\s*(?P<amount>(?:[{_CN_NUM_CHARS}]+|\d+(?:\.\d+)?))\s*(?P<scale>摄氏度|摄氏|华氏度|华氏)"
)
# “温度三十度” and similar phrasing conventionally imply Celsius in Chinese.
# Keep the cue in the output so surrounding sentence structure is unchanged.
_CN_TEMPERATURE_CONTEXT_RE = re.compile(
    rf"(?P<cue>温度|气温|室温|体温|水温|油温)\s*(?P<link>是|为|在|约|大约|达到|有)?\s*"
    rf"(?P<below>零下)?\s*(?P<amount>负?(?:[{_CN_NUM_CHARS}]+|[+-]?\d+(?:\.\d+)?))\s*度"
)


def _parse_cn_or_digit_amount(raw: str) -> str | None:
    raw = raw.strip()
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", raw):
        return raw
    return _cn_number(raw)


def _format_unit(amount: str, symbol: str, tight: bool) -> str:
    return f"{amount}{symbol}" if tight else f"{amount} {symbol}"


def normalize_chinese_units(text: str) -> str:
    def temp_prefix(match: re.Match[str]) -> str:
        amount = _parse_cn_or_digit_amount(match.group("amount"))
        if amount is None:
            return match.group(0)
        if match.group("below") and not amount.startswith("-"):
            amount = "-" + amount.lstrip("+")
        symbol = "°C" if match.group("scale") == "摄氏" else "°F"
        return f"{amount}{symbol}"

    text = _CN_TEMP_PREFIX_RE.sub(temp_prefix, text)

    def temp_below(match: re.Match[str]) -> str:
        amount = _parse_cn_or_digit_amount(match.group("amount"))
        if amount is None:
            return match.group(0)
        amount = "-" + amount.lstrip("+-")
        symbol = "°C" if match.group("scale").startswith("摄氏") else "°F"
        return f"{amount}{symbol}"

    text = _CN_TEMP_BELOW_RE.sub(temp_below, text)

    def temperature_context(match: re.Match[str]) -> str:
        amount = _parse_cn_or_digit_amount(match.group("amount"))
        if amount is None:
            return match.group(0)
        if match.group("below") and not amount.startswith("-"):
            amount = "-" + amount.lstrip("+")
        return f"{match.group('cue')}{match.group('link') or ''}{amount}°C"

    text = _CN_TEMPERATURE_CONTEXT_RE.sub(temperature_context, text)

    def repl(match: re.Match[str]) -> str:
        amount = _parse_cn_or_digit_amount(match.group("amount"))
        if amount is None:
            return match.group(0)
        symbol, tight = _CN_UNIT_ALIASES[match.group("unit")]
        return _format_unit(amount, symbol, tight)

    return _CN_UNIT_AMOUNT_RE.sub(repl, text)


# English unit normalization runs after English number-word normalization, so
# “thirty eight kilograms” has already become “38 kilograms”.  More specific
# compound units are listed before simple base units.
def _en_unit(pattern: str, symbol: str, tight: bool = False) -> tuple[re.Pattern[str], str, bool]:
    return re.compile(pattern, re.IGNORECASE), symbol, tight


_EN_UNIT_SPECS: list[tuple[re.Pattern[str], str, bool]] = [
    # Temperature
    _en_unit(r"(?:degrees?\s+)?(?:celsius|centigrade)\b", "°C", True),
    _en_unit(r"(?:degrees?\s+)?fahrenheit\b", "°F", True),
    _en_unit(r"(?:degrees?\s+)?kelvins?\b", "K"),

    # Chemistry / density / compound quantities first
    _en_unit(r"millimoles?\s+per\s+(?:liter|litre)\b", "mmol/L"),
    _en_unit(r"micromoles?\s+per\s+(?:liter|litre)\b", "μmol/L"),
    _en_unit(r"nanomoles?\s+per\s+(?:liter|litre)\b", "nmol/L"),
    _en_unit(r"moles?\s+per\s+(?:liter|litre)\b", "mol/L"),
    _en_unit(r"moles?\s+per\s+kilogram\b", "mol/kg"),
    _en_unit(r"kilograms?\s+per\s+mole\b", "kg/mol"),
    _en_unit(r"grams?\s+per\s+mole\b", "g/mol"),
    _en_unit(r"micrograms?\s+per\s+(?:liter|litre)\b", "μg/L"),
    _en_unit(r"nanograms?\s+per\s+(?:liter|litre)\b", "ng/L"),
    _en_unit(r"milligrams?\s+per\s+(?:liter|litre)\b", "mg/L"),
    _en_unit(r"grams?\s+per\s+(?:liter|litre)\b", "g/L"),
    _en_unit(r"micrograms?\s+per\s+milliliter\b|micrograms?\s+per\s+millilitre\b", "μg/mL"),
    _en_unit(r"milligrams?\s+per\s+milliliter\b|milligrams?\s+per\s+millilitre\b", "mg/mL"),
    _en_unit(r"kilograms?\s+per\s+cubic\s+(?:meter|metre)\b", "kg/m³"),
    _en_unit(r"grams?\s+per\s+cubic\s+(?:centimeter|centimetre)\b", "g/cm³"),
    _en_unit(r"grams?\s+per\s+milliliter\b|grams?\s+per\s+millilitre\b", "g/mL"),

    # Area / volume
    _en_unit(r"square\s+kilometers?\b|square\s+kilometres?\b", "km²"),
    _en_unit(r"square\s+meters?\b|square\s+metres?\b", "m²"),
    _en_unit(r"square\s+centimeters?\b|square\s+centimetres?\b", "cm²"),
    _en_unit(r"square\s+millimeters?\b|square\s+millimetres?\b", "mm²"),
    _en_unit(r"square\s+miles?\b", "mi²"),
    _en_unit(r"square\s+feet\b|square\s+foot\b", "ft²"),
    _en_unit(r"square\s+inches?\b", "in²"),
    _en_unit(r"cubic\s+kilometers?\b|cubic\s+kilometres?\b", "km³"),
    _en_unit(r"cubic\s+meters?\b|cubic\s+metres?\b", "m³"),
    _en_unit(r"cubic\s+centimeters?\b|cubic\s+centimetres?\b", "cm³"),
    _en_unit(r"cubic\s+millimeters?\b|cubic\s+millimetres?\b", "mm³"),

    # Speed / acceleration / flow
    _en_unit(r"kilometers?\s+per\s+hour\b|kilometres?\s+per\s+hour\b", "km/h"),
    _en_unit(r"miles?\s+per\s+hour\b", "mph"),
    _en_unit(r"meters?\s+per\s+second\s+squared\b|metres?\s+per\s+second\s+squared\b", "m/s²"),
    _en_unit(r"meters?\s+per\s+square\s+second\b|metres?\s+per\s+square\s+second\b", "m/s²"),
    _en_unit(r"feet\s+per\s+second\s+squared\b|foot\s+per\s+second\s+squared\b", "ft/s²"),
    _en_unit(r"meters?\s+per\s+second\b|metres?\s+per\s+second\b", "m/s"),
    _en_unit(r"centimeters?\s+per\s+second\b|centimetres?\s+per\s+second\b", "cm/s"),
    _en_unit(r"liters?\s+per\s+minute\b|litres?\s+per\s+minute\b", "L/min"),
    _en_unit(r"milliliters?\s+per\s+minute\b|millilitres?\s+per\s+minute\b", "mL/min"),
    _en_unit(r"cubic\s+(?:meters?|metres?)\s+per\s+second\b", "m³/s"),
    _en_unit(r"cubic\s+(?:meters?|metres?)\s+per\s+hour\b", "m³/h"),
    _en_unit(r"knots?\b", "kn"),

    # Force / torque
    _en_unit(r"kilonewton[ -]?meters?\b|kilonewton[ -]?metres?\b", "kN·m"),
    _en_unit(r"newton[ -]?meters?\b|newton[ -]?metres?\b", "N·m"),
    _en_unit(r"meganewtons?\b", "MN"),
    _en_unit(r"kilonewtons?\b", "kN"),
    _en_unit(r"millinewtons?\b", "mN"),
    _en_unit(r"micronewtons?\b", "μN"),
    _en_unit(r"newtons?\b", "N"),
    _en_unit(r"pounds?[- ]force\b", "lbf"),

    # Pressure
    _en_unit(r"gigapascals?\b", "GPa"),
    _en_unit(r"megapascals?\b", "MPa"),
    _en_unit(r"kilopascals?\b", "kPa"),
    _en_unit(r"hectopascals?\b", "hPa"),
    _en_unit(r"pascals?\b", "Pa"),
    _en_unit(r"millibars?\b", "mbar"),
    _en_unit(r"bars?\b", "bar"),
    _en_unit(r"standard\s+atmospheres?\b|atmospheres?\b", "atm"),
    _en_unit(r"torr\b", "Torr"),
    _en_unit(r"millimeters?\s+of\s+mercury\b|millimetres?\s+of\s+mercury\b", "mmHg"),
    _en_unit(r"centimeters?\s+of\s+water\b|centimetres?\s+of\s+water\b", "cmH₂O"),
    _en_unit(r"pounds?\s+per\s+square\s+inch\b", "psi"),

    # Energy / heat / power
    _en_unit(r"gigawatt[ -]?hours?\b", "GWh"),
    _en_unit(r"megawatt[ -]?hours?\b", "MWh"),
    _en_unit(r"kilowatt[ -]?hours?\b", "kWh"),
    _en_unit(r"watt[ -]?hours?\b", "Wh"),
    _en_unit(r"gigaelectronvolts?\b", "GeV"),
    _en_unit(r"megaelectronvolts?\b", "MeV"),
    _en_unit(r"kiloelectronvolts?\b", "keV"),
    _en_unit(r"electronvolts?\b", "eV"),
    _en_unit(r"gigajoules?\b", "GJ"),
    _en_unit(r"megajoules?\b", "MJ"),
    _en_unit(r"kilojoules?\b", "kJ"),
    _en_unit(r"joules?\b", "J"),
    _en_unit(r"kilocalories?\b", "kcal"),
    _en_unit(r"calories?\b", "cal"),
    _en_unit(r"gigawatts?\b", "GW"),
    _en_unit(r"megawatts?\b", "MW"),
    _en_unit(r"kilowatts?\b", "kW"),
    _en_unit(r"milliwatts?\b", "mW"),
    _en_unit(r"watts?\b", "W"),
    _en_unit(r"horsepower\b", "hp"),

    # Frequency
    _en_unit(r"terahertz\b", "THz"),
    _en_unit(r"gigahertz\b", "GHz"),
    _en_unit(r"megahertz\b", "MHz"),
    _en_unit(r"kilohertz\b", "kHz"),
    _en_unit(r"hertz\b", "Hz"),

    # Electrical / magnetic / photometric / radiation
    _en_unit(r"megavolts?\b", "MV"),
    _en_unit(r"kilovolts?\b", "kV"),
    _en_unit(r"millivolts?\b", "mV"),
    _en_unit(r"volts?\b", "V"),
    _en_unit(r"kiloamps?\b|kiloamperes?\b", "kA"),
    _en_unit(r"milliamps?\b|milliamperes?\b", "mA"),
    _en_unit(r"microamps?\b|microamperes?\b", "μA"),
    _en_unit(r"amps?\b|amperes?\b", "A"),
    _en_unit(r"megaohms?\b", "MΩ"),
    _en_unit(r"kiloohms?\b", "kΩ"),
    _en_unit(r"ohms?\b", "Ω"),
    _en_unit(r"coulombs?\b", "C"),
    _en_unit(r"microfarads?\b", "μF"),
    _en_unit(r"nanofarads?\b", "nF"),
    _en_unit(r"picofarads?\b", "pF"),
    _en_unit(r"millifarads?\b", "mF"),
    _en_unit(r"farads?\b", "F"),
    _en_unit(r"millihenr(?:y|ies)\b", "mH"),
    _en_unit(r"microhenr(?:y|ies)\b", "μH"),
    _en_unit(r"henr(?:y|ies)\b", "H"),
    _en_unit(r"milliteslas?\b", "mT"),
    _en_unit(r"microteslas?\b", "μT"),
    _en_unit(r"teslas?\b", "T"),
    _en_unit(r"webers?\b", "Wb"),
    _en_unit(r"siemens\b", "S"),
    _en_unit(r"lumens?\b", "lm"),
    _en_unit(r"lux\b", "lx"),
    _en_unit(r"candelas?\b", "cd"),
    _en_unit(r"becquerels?\b", "Bq"),
    _en_unit(r"grays?\b", "Gy"),
    _en_unit(r"sieverts?\b", "Sv"),

    # Chemistry base units
    _en_unit(r"millimoles?\b", "mmol"),
    _en_unit(r"micromoles?\b", "μmol"),
    _en_unit(r"nanomoles?\b", "nmol"),
    _en_unit(r"moles?\b", "mol"),
    _en_unit(r"milliequivalents?\b", "mEq"),
    _en_unit(r"equivalents?\b", "Eq"),

    # Area standalone
    _en_unit(r"hectares?\b", "ha"),
    _en_unit(r"acres?\b", "acre"),

    # Volume standalone
    _en_unit(r"milliliters?\b|millilitres?\b", "mL"),
    _en_unit(r"microliters?\b|microlitres?\b", "μL"),
    _en_unit(r"nanoliters?\b|nanolitres?\b", "nL"),
    _en_unit(r"liters?\b|litres?\b", "L"),
    _en_unit(r"US\s+gallons?\b|U\.S\.\s+gallons?\b", "US gal"),
    _en_unit(r"imperial\s+gallons?\b", "imp gal"),

    # Mass standalone.  Bare “pounds” is intentionally omitted because GBP.
    _en_unit(r"metric\s+tons?\b|tonnes?\b", "t"),
    _en_unit(r"kilograms?\b|kilos?\b", "kg"),
    _en_unit(r"milligrams?\b", "mg"),
    _en_unit(r"micrograms?\b", "μg"),
    _en_unit(r"nanograms?\b", "ng"),
    _en_unit(r"picograms?\b", "pg"),
    _en_unit(r"grams?\b", "g"),
    _en_unit(r"ounces?\b", "oz"),
    _en_unit(r"carats?\b", "ct"),

    # Length standalone
    _en_unit(r"kilometers?\b|kilometres?\b", "km"),
    _en_unit(r"decimeters?\b|decimetres?\b", "dm"),
    _en_unit(r"centimeters?\b|centimetres?\b", "cm"),
    _en_unit(r"millimeters?\b|millimetres?\b", "mm"),
    _en_unit(r"micrometers?\b|micrometres?\b|microns?\b", "μm"),
    _en_unit(r"nanometers?\b|nanometres?\b", "nm"),
    _en_unit(r"picometers?\b|picometres?\b", "pm"),
    _en_unit(r"meters?\b|metres?\b", "m"),
    _en_unit(r"nautical\s+miles?\b", "nmi"),
    _en_unit(r"miles?\b", "mi"),
    _en_unit(r"feet\b|foot\b", "ft"),
    _en_unit(r"inches?\b", "in"),
    _en_unit(r"yards?\b", "yd"),

    # Angle (bare English degrees is safe enough as an angle symbol; explicit
    # Celsius/Fahrenheit patterns above take precedence.)
    _en_unit(r"milliradians?\b", "mrad"),
    _en_unit(r"radians?\b", "rad"),
    _en_unit(r"steradians?\b", "sr"),
    _en_unit(r"degrees?\b", "°", True),
]

_EN_AMOUNT = r"(?P<amount>[+-]?\d+(?:\.\d+)?)"


def normalize_english_units(text: str) -> str:
    for unit_pattern, symbol, tight in _EN_UNIT_SPECS:
        combined = re.compile(
            _EN_AMOUNT + r"\s+(?:" + unit_pattern.pattern + r")",
            unit_pattern.flags,
        )
        text = combined.sub(
            lambda m, s=symbol, t=tight: _format_unit(m.group("amount"), s, t),
            text,
        )
    return text


def normalize_units(text: str) -> str:
    # Chinese raw-unit pass is safe to run again for already-numeric forms and
    # catches mixed output from the model such as “38千帕”.
    text = normalize_chinese_units(text)
    text = normalize_english_units(text)
    return text

def apply_numeric_itn(text: str, language_hint: str | None = None) -> str:
    """Apply conservative local ITN.

    The function name is kept for backward compatibility with the existing
    server import. `language_hint` is advisory. Mixed Chinese/English text is
    supported. V4 covers numbers, ratio notation, common currencies, a broad
    deterministic set of physical / engineering / chemistry units, plus
    explicit dates, clock times, and common durations.
    """
    if not text:
        return text
    result = normalize_dates_times_raw(text)
    result = normalize_chinese_ratio_measure_words(result)
    result = normalize_chinese_ratios_raw(result)
    # Run Chinese units before generic Chinese-number ITN so single-digit forms
    # such as “三公斤” and unit-prefix forms such as “三千克” are handled safely.
    result = normalize_chinese_currencies_raw(result)
    result = normalize_chinese_units(result)
    result = normalize_chinese_numbers(result)
    result = normalize_english_numbers(result)
    result = normalize_ratios(result)
    result = normalize_currencies(result)
    result = normalize_units(result)
    result = normalize_dates_times(result)
    return result
