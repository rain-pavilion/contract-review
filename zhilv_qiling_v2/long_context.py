"""Bounded, auditable long-contract windows and cross-clause dependency plans.

Every nonempty clause is split without dropping its tail. Cross-clause windows are
additional targeted checks; their coverage is reported separately from text coverage.
No model call or network request happens while planning.
"""
from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal, InvalidOperation
import re
from dataclasses import dataclass, replace
from typing import Any, Sequence

from .models import Clause


@dataclass(frozen=True)
class Segment:
    clause_id: str
    start: int
    end: int
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {"clause_id": self.clause_id, "start": self.start,
                "end": self.end, "text": self.text}


@dataclass(frozen=True)
class ReviewWindow:
    window_id: str
    kind: str
    segments: tuple[Segment, ...]
    reason: str = ""
    dependency_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"window_id": self.window_id, "kind": self.kind,
                "reason": self.reason, "segments": [s.as_dict() for s in self.segments],
                "dependency_ids": list(self.dependency_ids)}


TOPICS = {
    "payment": ("价款", "总价", "金额", "付款", "支付", "结算", "预付款", "尾款"),
    "delivery": ("交付", "交货", "验收", "履行期限", "完成时间", "交期"),
    "liability": ("违约", "赔偿", "免责", "责任", "损失"),
    "authority": ("授权", "委托", "代理", "签署", "代表人", "资质", "许可"),
    "termination": ("解除", "终止", "撤销", "退还", "退款"),
}



@dataclass(frozen=True)
class ObligationValue:
    """A literal value in a narrow obligation slot, never a conflict verdict."""
    clause_id: str
    slot: str
    scope: str
    value: str
    start: int
    end: int
    anchor: Segment


_PAYMENT_SLOTS = {
    '合同总金额': 'total', '合同总价': 'total', '合同价款': 'total',
    '预付款': 'advance', '首付款': 'first_payment', '首期款': 'first_instalment',
    '第二期款': 'second_instalment', '第三期款': 'third_instalment',
    '尾款': 'tail', '质保金': 'retention',
}
_PAYMENT_VALUE = re.compile(
    r'(?P<slot>' + '|'.join(sorted(_PAYMENT_SLOTS, key=len, reverse=True)) + r')'
    r'\s*(?:金额\s*)?(?:(?:应为|合计|共计|计|为|人民币|CNY|RMB|[：:￥¥])\s*)*'
    r'(?P<number>\d+(?:[,，]\d{3})*(?:\.\d{1,2})?)\s*(?P<unit>亿元|万元|元)')
_DATE_VALUE = re.compile(
    r'(?<!\d)(?P<year>[12]\d{3})(?:年(?P<cm>\d{1,2})月(?P<cd>\d{1,2})日'
    r'|(?P<sep>[-/])(?P<nm>\d{1,2})(?P=sep)(?P<nd>\d{1,2}))(?!\d)')
_SCOPE_KINDS = {'项目': 'project', '标的': 'object', '设备': 'device', '系统': 'system', '服务': 'service'}
_SCOPE_KIND = '|'.join(_SCOPE_KINDS)
_SCOPE_BOUNDARY = r'(?=$|[\s，,：:。；;]|的|应|须|需|于|在|最终|全部|交付|交货|验收|' + '|'.join(_PAYMENT_SLOTS) + r')'
_NAMED_SCOPES = (
    re.compile(r'(?P<kind>' + _SCOPE_KIND + r')\s*[“「《\"](?P<name>[^”」》\"\n，,。；;]{1,40})[”」》\"]'),
    re.compile(r'(?P<kind>' + _SCOPE_KIND + r')\s*(?P<name>[A-Za-z0-9][A-Za-z0-9_-]{0,29}|[甲乙丙丁戊己庚辛壬癸一二三四五六七八九十]{1,4})' + _SCOPE_BOUNDARY),
    re.compile(r'[“「《\"](?P<name>[^”」》\"\n，,。；;]{1,40})[”」》\"]\s*(?P<kind>' + _SCOPE_KIND + r')'),
)
_GENERIC_SCOPE = re.compile(r'本(?P<kind>合同|项目|标的|设备|系统|服务)')
_STAGE_WORD = r'最终|全部|初步|首批|第[一二三四五六七八九十0-9]+批'
_DATE_EVENT = re.compile(r'(?P<stage>' + _STAGE_WORD + r')?(?P<event>交付|交货|验收)')
_EXAMPLE_VALUE_FIELD = re.compile(r'示例|例如|假设|演示|样例')
_DATE_LINK_LEFT = re.compile(
    r'^(?:日期|时间|期限|截止日期|截止日)?\s*'
    r'(?:(?:确定为|不得晚于|不晚于|最迟|定为|应当|应|须|需|于|在|为|[：:])\s*)*$')
_DATE_LINK_RIGHT = re.compile(
    r'^\s*(?:之前|前|当日|之日)?\s*(?:(?:完成|进行|应|须|需|予以)\s*)*$')


def _field_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    # Search outside the value match so the comma in 1,000 is not a delimiter.
    marks = '。；;\n，,'
    left = max(text.rfind(mark, 0, start) for mark in marks) + 1
    rights = [pos for mark in marks if (pos := text.find(mark, end)) >= 0]
    return left, min(rights) if rights else len(text)


def _value_scope(field: str, *, contract_total: bool = False) -> str | None:
    """Require the same explicit object, or an explicit local 本合同/本项目 scope.

    Quoted Chinese names and simple alphanumeric object IDs are recognized.
    Unparsed object names, multiple objects and missing scope cause abstention;
    no cross-sentence scope, identity alias or commercial meaning is inferred.
    """
    named, spans = set(), []
    for pattern in _NAMED_SCOPES:
        for match in pattern.finditer(field):
            named.add(_SCOPE_KINDS[match.group('kind')] + ':' + match.group('name').strip())
            spans.append(match.span())
    generic = list(_GENERIC_SCOPE.finditer(field))
    spans.extend(match.span() for match in generic)
    remainder = list(field)
    for left, right in spans:
        remainder[left:right] = ' ' * (right-left)
    if re.search(_SCOPE_KIND, ''.join(remainder)) or len(named) > 1:
        return None
    if named:
        return next(iter(named))
    scopes = {('contract' if match.group('kind') == '合同'
               else _SCOPE_KINDS[match.group('kind')] + ':this') for match in generic}
    if len(scopes) == 1:
        return next(iter(scopes))
    return 'contract' if not scopes and contract_total else None


def _value_anchor(clause: Clause, parts: Sequence[Segment], start: int, end: int, size: int) -> Segment:
    containing = [part for part in parts if part.start <= start and part.end >= end]
    if containing:
        return max(containing, key=lambda part: min(start-part.start, part.end-end))
    left = max(0, min(start-size//2, len(clause.text)-size))
    right = min(len(clause.text), left+size)
    return Segment(clause.clause_id, left, right, clause.text[left:right])


def _stage_key(value: str | None) -> str:
    if value in ('最终', '全部'):
        return 'final'
    if value == '初步':
        return 'preliminary'
    if value == '首批':
        return 'batch:1'
    if value and value.startswith('第'):
        return 'batch:' + _reference_key(value[:-1] + '条').removeprefix('article:')
    return 'unspecified'


def _date_slot(field: str, value_start: int, value_end: int) -> str | None:
    # A range, two different events, mixed milestones or a remote unrelated date
    # must not get the same-obligation priority merely by sharing topic words.
    events = list(_DATE_EVENT.finditer(field))
    slots = {('delivery' if item.group('event') in ('交付', '交货') else 'acceptance',
              _stage_key(item.group('stage'))) for item in events}
    milestones = {_stage_key(item.group()) for item in re.finditer(_STAGE_WORD, field)}
    if len(list(_DATE_VALUE.finditer(field))) != 1 or len(slots) != 1:
        return None
    event, stage = next(iter(slots))
    if milestones and milestones != {stage}:
        return None
    linked = any((item.end() <= value_start and _DATE_LINK_LEFT.fullmatch(field[item.end():value_start]))
                 or (item.start() >= value_end and _DATE_LINK_RIGHT.fullmatch(field[value_end:item.start()]))
                 for item in events)
    return 'date:' + event + ':' + stage if linked else None


def _obligation_value_pairs(clauses: Sequence[Clause], segments: dict[str, list[Segment]], size: int
                            ) -> list[tuple[ObligationValue, ObligationValue]]:
    """Propose literal differences in the same narrow slot, without legal judgment.

    The supported grammar is deliberately conservative: fixed payment-stage
    amounts or one explicitly linked delivery/acceptance date per local field.
    Bounds, ranges, unrecognized names and missing scope are not guessed. Source
    conditions, precedence, amendments and actual contradictions need model/human
    review. No exhaustive semantic coverage is asserted.
    """
    groups: dict[tuple[str, str], list[ObligationValue]] = {}
    for clause in clauses:
        for match in _PAYMENT_VALUE.finditer(clause.text):
            left, right = _field_bounds(clause.text, match.start(), match.end())
            field = clause.text[left:right]
            scope = _value_scope(field, contract_total=_PAYMENT_SLOTS[match.group('slot')] == 'total')
            if scope is None or _EXAMPLE_VALUE_FIELD.search(field) or re.search(r'预算|估算|上限|下限|最高|最低|不超过|不得超过|不少于|参考', field):
                continue
            tax = {'exclusive' if item in ('不含税', '未税', '税前') else 'inclusive'
                   for item in re.findall(r'不含税|未税|税前|含税|税后', field)}
            if len(tax) > 1:
                continue
            try:
                value = Decimal(re.sub(r'[,，]', '', match.group('number')))
                value *= {'元': 1, '万元': 10000, '亿元': 100000000}[match.group('unit')]
            except InvalidOperation:
                continue
            slot = 'payment:' + _PAYMENT_SLOTS[match.group('slot')] + ':tax=' + next(iter(tax), 'unspecified')
            start, end = match.start('number'), match.end('unit')
            row = ObligationValue(clause.clause_id, slot, scope, str(value.normalize()), start, end,
                _value_anchor(clause, segments[clause.clause_id], start, end, size))
            groups.setdefault((slot, scope), []).append(row)
        for match in _DATE_VALUE.finditer(clause.text):
            left, right = _field_bounds(clause.text, match.start(), match.end())
            field = clause.text[left:right]
            slot = _date_slot(field, match.start()-left, match.end()-left)
            scope = _value_scope(field)
            if slot is None or scope is None or _EXAMPLE_VALUE_FIELD.search(field):
                continue
            try:
                value = date(int(match.group('year')), int(match.group('cm') or match.group('nm')),
                             int(match.group('cd') or match.group('nd'))).isoformat()
            except ValueError:
                continue
            row = ObligationValue(clause.clause_id, slot, scope, value, match.start(), match.end(),
                _value_anchor(clause, segments[clause.clause_id], match.start(), match.end(), size))
            groups.setdefault((slot, scope), []).append(row)
    return [(left, right) for rows in groups.values() for index, left in enumerate(rows)
            for right in rows[index+1:] if left.clause_id != right.clause_id and left.value != right.value]


def _window(kind: str, segments: Sequence[Segment], reason: str = "") -> ReviewWindow:
    keys = [f"{s.clause_id}:{s.start}:{s.end}" for s in segments]
    # Keep primary text IDs stable; reciprocal cross checks share a real call
    # only when their complete clause/range segment sets are identical.
    key = "|".join(sorted(set(keys)) if kind == "cross_clause" else keys)
    return ReviewWindow(hashlib.sha256((kind + key).encode()).hexdigest()[:16],
                        kind, tuple(segments), reason)


def split_clause(clause: Clause, chunk_chars: int = 1800, overlap: int = 160) -> list[Segment]:
    size = max(200, min(int(chunk_chars), 12000))
    overlap = max(0, min(int(overlap), size // 3))
    text = clause.text or ""
    segments = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        segments.append(Segment(clause.clause_id, start, end, text[start:end]))
        if end == len(text):
            break
        start = end - overlap
    return segments


REFERENCE = re.compile(r"第[一二三四五六七八九十百千万零〇两0-9]+条|(?<![0-9.])\d+(?:\.\d+)+(?![0-9.])")


def _reference_key(label: str) -> str:
    """Canonicalize actual clause labels, including 第二条 / 第2条 aliases."""
    if not label.startswith("第"):
        return ".".join(str(int(part)) for part in label.split("."))
    value = label[1:-1]
    if value.isdigit():
        return "article:" + str(int(value))
    digits = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
    digits.update({str(i): i for i in range(10)})
    if not any(ch in "十百千万" for ch in value):
        return "article:" + str(int("".join(str(digits[ch]) for ch in value)))
    total = section = number = 0
    for ch in value:
        if ch in digits:
            number = digits[ch]
        else:
            unit = {"十": 10, "百": 100, "千": 1000, "万": 10000}[ch]
            if unit == 10000:
                total += (section + number or 1) * unit
                section = 0
            else:
                section += (number or 1) * unit
            number = 0
    return "article:" + str(total + section + number)


@dataclass(frozen=True)
class ReferenceDependency:
    dependency_id: str
    source_id: str
    target_id: str
    reference_start: int
    reference_end: int
    reference_text: str
    anchor: Segment
    source_segments: tuple[Segment, ...]
    target_segments: tuple[Segment, ...]


def _dependencies(clauses: Sequence[Clause], segments: dict[str, list[Segment]],
                  size: int) -> tuple[list[ReferenceDependency], list[dict[str, Any]]]:
    labels: dict[str, list[str]] = {}
    for clause in clauses:
        label = re.sub(r"\s", "", clause.label or "").lstrip("（([")
        match = REFERENCE.match(label)
        if match:
            labels.setdefault(_reference_key(match.group()), []).append(clause.clause_id)
    dependencies, issues = [], []
    for clause in clauses:
        for match in REFERENCE.finditer(clause.text):
            targets = list(dict.fromkeys(labels.get(_reference_key(match.group()), [])))
            if targets == [clause.clause_id]:
                continue
            if not targets:
                # A bare decimal such as 0.5% is data, not necessarily a clause
                # citation. Unknown 第N条 and explicitly referenced decimals
                # remain pending; an external-law citation is not guessed into
                # a nonexistent contract clause ID either.
                prefix = clause.text[max(0, match.start() - 20):match.start()]
                suffix = clause.text[match.end():match.end() + 8]
                quantity = bool(re.match(r"\s*(?:[%％]|(?:亿|万)?元|年|月|日|天|倍|吨|米|平方米)", suffix))
                explicit_decimal = not quantity and bool(re.search(
                    r"(?:按|依|见|依据|依照|根据|参照|参见|条款|第)\s*$", prefix)
                    or re.match(r"\s*(?:条|款|项|节|的?约定|的?规定)", suffix))
                if match.group().startswith("第") or explicit_decimal:
                    issues.append({"source_clause_id": clause.clause_id, "reference_start": match.start(),
                        "reference_end": match.end(), "reference_text": match.group(),
                        "candidate_target_clause_ids": [], "status": "pending_unresolved_target",
                        "note": "target label not resolved; may be an external law citation; no contract ID invented"})
                continue
            if len(targets) != 1:
                issues.append({"source_clause_id": clause.clause_id, "reference_start": match.start(),
                    "reference_end": match.end(), "reference_text": match.group(),
                    "candidate_target_clause_ids": targets, "status": "pending_ambiguous_target"})
                continue
            target = targets[0]
            containing = [s for s in segments[clause.clause_id]
                          if s.start <= match.start() and s.end >= match.end()]
            if containing:
                anchor = max(containing, key=lambda s: min(match.start() - s.start, s.end - match.end()))
            else:
                # A reference token itself can straddle a primary chunk boundary.
                start = max(0, min(match.start() - size // 2, len(clause.text) - size))
                end = min(len(clause.text), start + size)
                anchor = Segment(clause.clause_id, start, end, clause.text[start:end])
            key = f"{clause.clause_id}:{target}:{match.start()}:{match.end()}"
            dependencies.append(ReferenceDependency(hashlib.sha256(key.encode()).hexdigest()[:16],
                clause.clause_id, target, match.start(), match.end(), match.group(), anchor,
                tuple(segments[clause.clause_id]), tuple(segments[target])))
    return dependencies, issues


def _segment_range(segment: Segment) -> dict[str, Any]:
    return {"clause_id": segment.clause_id, "start": segment.start, "end": segment.end}


def _dependency_status(dependency: ReferenceDependency, windows: Sequence[ReviewWindow]) -> dict[str, Any]:
    """Only joint windows carrying the actual reference anchor prove a check."""
    checked: set[tuple[int, int]] = set()
    used = []
    for window in windows:
        if window.kind != "cross_clause" or not any(s.clause_id == dependency.source_id
                and s.start <= dependency.reference_start and s.end >= dependency.reference_end
                for s in window.segments):
            continue
        sources = [i for i, part in enumerate(dependency.source_segments) if any(
            s.clause_id == part.clause_id and s.start <= part.start and s.end >= part.end for s in window.segments)]
        targets = [i for i, part in enumerate(dependency.target_segments) if any(
            s.clause_id == part.clause_id and s.start <= part.start and s.end >= part.end for s in window.segments)]
        if sources and targets:
            checked.update((left, right) for left in sources for right in targets)
            used.append(window.window_id)
    source_count, target_count = len(dependency.source_segments), len(dependency.target_segments)
    required = source_count * target_count
    return {"dependency_id": dependency.dependency_id, "source_clause_id": dependency.source_id,
        "target_clause_id": dependency.target_id, "reference_start": dependency.reference_start,
        "reference_end": dependency.reference_end, "reference_text": dependency.reference_text,
        "source_anchor": _segment_range(dependency.anchor),
        "required_checks": required, "checked_checks": len(checked), "pending_checks": required - len(checked),
        "complete": len(checked) == required, "window_ids": list(dict.fromkeys(used)),
        "pending_source_segments": [_segment_range(part) for i, part in enumerate(dependency.source_segments)
                                    if sum(left == i for left, _ in checked) < target_count],
        "pending_target_segments": [_segment_range(part) for i, part in enumerate(dependency.target_segments)
                                    if sum(right == i for _, right in checked) < source_count]}


def _source_context_key(clause: Clause) -> tuple[str, str] | None:
    metadata = clause.metadata or {}
    group = metadata.get("table_group_id") or metadata.get("article_group_id")
    source = metadata.get("source_id") or metadata.get("document_id")
    return (str(source), str(group)) if source and group else None


def _source_context_groups(clauses: Sequence[Clause]) -> list[tuple[tuple[str, str], list[Clause]]]:
    groups: dict[tuple[str, str], list[Clause]] = {}
    for clause in clauses:
        metadata = clause.metadata or {}
        source = metadata.get("source_id") or metadata.get("document_id")
        ids = list(dict.fromkeys([*(metadata.get("table_group_ids") or []),
                                 *(metadata.get("article_group_ids") or [])]))
        if not ids:
            key = _source_context_key(clause)
            ids = [key[1]] if key else []
        for group_id in ids:
            if source and group_id and clause.text:
                groups.setdefault((str(source), str(group_id)), []).append(clause)
    return [(key, group) for key, group in groups.items()
            if len(group) > 1 or any((clause.metadata or {}).get("requires_full_context") for clause in group)]


def _group_covered(group: Sequence[Clause], window: ReviewWindow) -> bool:
    # One actually supplied window must contain the complete grouped context.
    # Executing disconnected fragments is not an acknowledgement of the group.
    for clause in group:
        parts = sorted((part for part in window.segments if part.clause_id == clause.clause_id),
                       key=lambda part: (part.start, part.end))
        end = 0
        for part in parts:
            if part.start > end or clause.text[part.start:part.end] != part.text:
                return False
            end = max(end, part.end)
        if end < len(clause.text):
            return False
    return True


def source_context_checks(clauses: Sequence[Clause], windows: Sequence[ReviewWindow]) -> list[dict[str, Any]]:
    return [{"source_id": key[0], "group_id": key[1],
             "clause_ids": [clause.clause_id for clause in group],
             "window_ids": [window.window_id for window in windows if _group_covered(group, window)],
             "context_supplied": any(_group_covered(group, window) for window in windows),
             "meaning": "原文上下文供给检查；不证明OCR准确、表格已还原或法律结论正确。"}
            for key, group in _source_context_groups(clauses)]


def build_review_plan(clauses: Sequence[Clause], *, chunk_chars: int = 1800,
                      overlap: int = 160, cross_pairs: int = 12) -> tuple[list[ReviewWindow], dict[str, Any]]:
    """Pack adjacent clauses and add distant references / shared-topic checks.

    Cross checks are prioritized, not advertised as exhaustive pairwise inference.
    Every source character remains covered by the primary ``text`` windows.
    """
    clauses = [c for c in clauses if c.text]
    size = max(200, min(int(chunk_chars), 12000))
    segments = {c.clause_id: split_clause(c, size, overlap) for c in clauses}
    windows: list[ReviewWindow] = []
    packed: list[Segment] = []
    chars = 0
    # Keep a source-local article/table together when it fits the existing
    # window size. Neither clause identities nor source characters are edited.
    packs: list[list[Clause]] = []
    for clause in clauses:
        key = _source_context_key(clause)
        if packs and key and _source_context_key(packs[-1][-1]) == key:
            packs[-1].append(clause)
        else:
            packs.append([clause])
    for group in packs:
        group_chars = sum(len(clause.text) for clause in group)
        if packed and len(group) > 1 and group_chars <= size and chars + group_chars > size:
            windows.append(_window("text", packed))
            packed, chars = [], 0
        for segment in [part for clause in group for part in segments[clause.clause_id]]:
            if packed and chars + len(segment.text) > size:
                windows.append(_window("text", packed))
                packed, chars = [], 0
            packed.append(segment)
            chars += len(segment.text)
    if packed:
        windows.append(_window("text", packed))

    pairs: dict[tuple[str, str], tuple[int, str]] = {}
    positions = {c.clause_id: i for i, c in enumerate(clauses)}
    dependencies, reference_issues = _dependencies(clauses, segments, size)
    ambiguous = [item for item in reference_issues if item["status"] == "pending_ambiguous_target"]
    unresolved = [item for item in reference_issues if item["status"] == "pending_unresolved_target"]
    explicit_pairs = {tuple(sorted((d.source_id, d.target_id), key=positions.get)) for d in dependencies}
    for pair in explicit_pairs:
        pairs[pair] = (100, "explicit_reference")
    groups: dict[str, list[str]] = {name: [] for name in TOPICS}
    topics_by_clause: dict[str, set[str]] = {}
    for c in clauses:
        topics = {name for name, words in TOPICS.items() if any(w in c.text for w in words)}
        topics_by_clause[c.clause_id] = topics
        for topic in topics:
            groups[topic].append(c.clause_id)
    # Linear candidate generation: nearby provisions plus distant same-topic
    # endpoints. We explicitly do not enumerate every pair in a long document.
    for topic, ids in groups.items():
        for i, cid in enumerate(ids):
            targets = ids[i + 1:i + 3] + ([ids[0], ids[-1]] if ids else [])
            for other in targets:
                if other == cid:
                    continue
                pair = tuple(sorted((cid, other), key=positions.get))
                common = topics_by_clause[cid] & topics_by_clause[other]
                if pair not in pairs:
                    pairs[pair] = (len(common), "shared_topic:" + ",".join(sorted(common)))
    ordered = sorted(pairs.items(), key=lambda item: (-item[1][0], item[0]))
    budget = max(0, int(cross_pairs))
    cross: dict[str, ReviewWindow] = {}
    # The old one-pair/one-window limit is retained as an actual call-window
    # budget. Expanding a long reference must not silently multiply that limit.
    for dependency in dependencies:
        # Visit the real anchor segment first so a small budget cannot revert to
        # the previous topic-rich-head-only behavior.
        source_parts = sorted(dependency.source_segments, key=lambda part: (
            not (part.start <= dependency.reference_start and part.end >= dependency.reference_end), part.start))
        for source_part in source_parts:
            for target_part in dependency.target_segments:
                if len(cross) >= budget:
                    break
                selected = list(dict.fromkeys((dependency.anchor, source_part, target_part)))
                window = _window("cross_clause", selected, "explicit_reference")
                cross.setdefault(window.window_id, window)
            if len(cross) >= budget:
                break
        if len(cross) >= budget:
            break
    # Numeric/date anchors get the remaining budget before generic topic heads.
    # Source-local groups that exceed one primary window receive a bounded
    # complete-context call if possible; impossible/omitted groups stay pending.
    for key, group in _source_context_groups(clauses):
        if any(_group_covered(group, window) for window in [*windows, *cross.values()]):
            continue
        if len(cross) >= budget or sum(len(clause.text) for clause in group) > 3 * size:
            continue
        window = _window("cross_clause", [Segment(clause.clause_id, 0, len(clause.text), clause.text)
            for clause in group], "source_context_group:" + key[1])
        cross.setdefault(window.window_id, window)
    # They only propose source-value differences; they do not emit review issues.
    obligation_pairs = _obligation_value_pairs(clauses, segments, size)
    obligation_windows: dict[str, ReviewWindow] = {}
    obligation_candidates = []
    anchored_pairs = set()
    clause_texts = {clause.clause_id: clause.text for clause in clauses}
    for left, right in obligation_pairs:
        reason = 'shared_obligation_value_difference:' + left.slot + ':scope=' + left.scope
        window = _window('cross_clause', (left.anchor, right.anchor), reason)
        obligation_windows.setdefault(window.window_id, window)
        obligation_candidates.append({'window_id': window.window_id, 'slot': left.slot,
            'scope': left.scope, 'candidate_only': True, 'legal_conflict_proven': False,
            'values': [{'clause_id': item.clause_id, 'start': item.start, 'end': item.end,
                        'source_value': clause_texts[item.clause_id][item.start:item.end],
                        'normalized_value': item.value, 'anchor': _segment_range(item.anchor)}
                       for item in (left, right)]})
        if len(cross) < budget:
            cross.setdefault(window.window_id, window)
        if window.window_id in cross:
            anchored_pairs.add(tuple(sorted((left.clause_id, right.clause_id), key=positions.get)))
    # Ordinary shared-topic pairs remain targeted, one representative window
    # each. They never claim exhaustive inference and never displace references.
    words = tuple(w for group in TOPICS.values() for w in group)
    for (left_id, right_id), (_, reason) in ordered:
        if (left_id, right_id) in explicit_pairs or (left_id, right_id) in anchored_pairs or len(cross) >= budget:
            continue
        selected = []
        for cid in (left_id, right_id):
            candidates = segments[cid]
            selected.append(max(candidates, key=lambda s: sum(w in s.text for w in words)))
        window = _window("cross_clause", selected, reason)
        cross.setdefault(window.window_id, window)
    dependency_plans = [_dependency_status(d, list(cross.values())) for d in dependencies]
    for dependency_plan in dependency_plans:
        dependency_plan["scheduled_checks"] = dependency_plan.pop("checked_checks")
        dependency_plan["scheduled_window_ids"] = dependency_plan.pop("window_ids")
        dependency_plan["plan_complete"] = dependency_plan.pop("complete")
    cross_windows = [replace(window, dependency_ids=tuple(d["dependency_id"] for d in dependency_plans
                     if window.window_id in d["scheduled_window_ids"])) for window in cross.values()]
    windows.extend(cross_windows)
    group_checks = source_context_checks(clauses, windows)
    group_pending = [row["group_id"] for row in group_checks if not row["context_supplied"]]
    scheduled_pairs = {tuple(sorted({s.clause_id for s in window.segments}, key=positions.get))
                       for window in cross_windows}
    explicit_pending = sum(d["pending_checks"] for d in dependency_plans)
    explicit_plan_complete = not reference_issues and not explicit_pending
    topic_not_scheduled = sum(pair not in scheduled_pairs for pair, _ in ordered if pair not in explicit_pairs)
    obligation_pending = sum(window_id not in cross for window_id in obligation_windows)
    candidate_pairs = set(pairs) | {tuple(sorted((left.clause_id, right.clause_id), key=positions.get))
                                    for left, right in obligation_pairs}
    meta = {
        "total_clauses": len(clauses), "total_chars": sum(len(c.text) for c in clauses),
        "text_windows": sum(w.kind == "text" for w in windows),
        "cross_windows": len(cross_windows), "cross_pairs_candidates": len(candidate_pairs),
        "cross_pairs_not_scheduled": sum(pair not in scheduled_pairs for pair in candidate_pairs),
        "cross_window_budget": budget, "cross_budget_unit": "bounded_cross_clause_windows",
        "cross_max_window_chars": 3 * size,
        "explicit_reference_dependencies": dependency_plans,
        "explicit_reference_required_checks": sum(d["required_checks"] for d in dependency_plans),
        "explicit_reference_pending_checks": explicit_pending,
        "explicit_reference_plan_pending_checks": explicit_pending,
        "explicit_reference_ambiguous_targets": ambiguous,
        "explicit_reference_unresolved_targets": unresolved,
        "explicit_reference_plan_complete": explicit_plan_complete,
        "explicit_reference_scope": "reference_anchor_with_all_source_target_segment_pairs_for_resolved_contract_labels",
        "topic_pairs_not_scheduled": topic_not_scheduled,
        "source_context_groups": group_checks,
        "source_context_group_plan_pending": group_pending,
        "source_context_group_plan_complete": not group_pending,
        "cross_plan_complete": explicit_plan_complete and not topic_not_scheduled and not obligation_pending and not group_pending,
        "cross_scope": "targeted_dependency_pairs_not_exhaustive",
        "cross_pair_generation": "explicit_reference+same_obligation_value_difference+topic_neighbours+topic_endpoints",
        "obligation_value_candidates": obligation_candidates,
        "obligation_value_windows_candidates": len(obligation_windows),
        "obligation_value_windows_scheduled": sum(window_id in cross for window_id in obligation_windows),
        "obligation_value_windows_pending": obligation_pending,
        "obligation_value_scope": "same_explicit_object_or_local_generic_scope_and_same_stage_event_tax_basis; candidate_not_legal_conflict; narrow_grammar_not_exhaustive",
        "potential_topic_pairs_upper_bound": sum(len(ids)*(len(ids)-1)//2 for ids in groups.values()),
        "chunk_chars": size, "overlap": max(0, min(int(overlap), size // 3)),
    }
    return windows, meta


def coverage(clauses: Sequence[Clause], windows: Sequence[ReviewWindow], completed: set[str]) -> dict[str, Any]:
    """Separate text coverage from actual completed, anchored reference checks.

    ``complete`` requires both. Completing every scheduled window cannot hide
    explicit dependencies omitted by the cross-window planning budget.
    """
    intervals: dict[str, list[tuple[int, int]]] = {c.clause_id: [] for c in clauses if c.text}
    for window in windows:
        if window.kind != "text" or window.window_id not in completed:
            continue
        for segment in window.segments:
            intervals.setdefault(segment.clause_id, []).append((segment.start, segment.end))
    done = 0
    complete_clauses, pending = [], []
    for clause in clauses:
        if not clause.text:
            continue
        end = total = 0
        for left, right in sorted(intervals.get(clause.clause_id, [])):
            total += max(0, right - max(left, end))
            end = max(end, right)
        done += total
        (complete_clauses if total >= len(clause.text) else pending).append(clause.clause_id)
    total_chars = sum(len(c.text or "") for c in clauses)
    source_segments: dict[str, list[Segment]] = {c.clause_id: [] for c in clauses if c.text}
    for window in windows:
        if window.kind == "text":
            for segment in window.segments:
                if segment not in source_segments.setdefault(segment.clause_id, []):
                    source_segments[segment.clause_id].append(segment)
    size = max((len(s.text) for parts in source_segments.values() for s in parts), default=1800)
    for clause in clauses:
        if clause.text and not source_segments.get(clause.clause_id):
            source_segments[clause.clause_id] = split_clause(clause, size)
    nonempty = [c for c in clauses if c.text]
    dependencies, reference_issues = _dependencies(nonempty, source_segments, size)
    done_cross = [w for w in windows if w.kind == "cross_clause" and w.window_id in completed]
    reference_checks = [_dependency_status(d, done_cross) for d in dependencies]
    explicit_pending = sum(d["pending_checks"] for d in reference_checks)
    reference_complete = not reference_issues and not explicit_pending
    group_checks = source_context_checks(nonempty, [window for window in windows if window.window_id in completed])
    group_pending = [row["group_id"] for row in group_checks if not row["context_supplied"]]
    return {"reviewed_chars": done, "total_chars": total_chars,
            "text_coverage": round(done / total_chars, 4) if total_chars else 1.0,
            "completed_clause_ids": complete_clauses, "unreviewed_clause_ids": pending,
            "text_complete": not pending,
            "explicit_reference_complete": reference_complete,
            "explicit_reference_pending_checks": explicit_pending,
            "explicit_reference_checks": reference_checks,
            "explicit_reference_ambiguous_targets": [item for item in reference_issues if item["status"] == "pending_ambiguous_target"],
            "explicit_reference_unresolved_targets": [item for item in reference_issues if item["status"] == "pending_unresolved_target"],
            "source_context_group_checks": group_checks,
            "source_context_group_pending": group_pending,
            "source_context_group_complete": not group_pending,
            "complete": not pending and reference_complete and not group_pending}
