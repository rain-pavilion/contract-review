"""Prepare a one-call v2 diagnostic from the immutable failed v1 runner, offline."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
prior = root / "tools/acceptance_results/initial_source_selection_real_20261118_v1"
target = root / "tools/acceptance_results/initial_source_selection_real_20261118_v2"
assert not target.exists(), "Fresh v2 directory required"
source = (prior / "probe.py").read_text(encoding="utf-8")
changes = {
    "at most three real native requests": "at most one real native request",
    '20261118-v1': '20261118-v2',
    'llm_native_reasoning_format="none"': 'llm_native_reasoning_format="deepseek"',
    'llm_cross_clause_pairs=0, llm_max_calls=3, llm_initial_max_attempts=3':
        'llm_cross_clause_pairs=0, llm_max_calls=1, llm_initial_max_attempts=1',
    'assert actual <= 3 and manifest["counts"]["complete_calls"] <= 3':
        'assert actual <= 1 and manifest["counts"]["complete_calls"] <= 1',
    'for key in ("body", "raw_bytes"):\n                blob =':
        'for key in ("body", "raw_bytes"):\n                if key not in value:\n                    continue\n                blob =',
    '    manifest = capture.summary()':
        '    write("runtime_return.json", {"meta": meta, "degraded": degraded, "error": error,\n'
        '        "candidates": [candidate.to_dict() for candidate in candidates], "events": audit.events,\n'
        '        "elapsed_with_instrumentation_seconds": elapsed})\n'
        '    manifest = capture.summary()',
    '        "prepared_at": datetime.now(timezone.utc).isoformat()}':
        '        "prior_native_configuration_failure": {"path": str(RUN.with_name("initial_source_selection_real_20261118_v1") / "result.json"),\n'
        '            "change": "reasoning_format none -> deepseek, matching historical working HTTP body; budget 3 -> 1"},\n'
        '        "prepared_at": datetime.now(timezone.utc).isoformat()}',
}
for old, new in changes.items():
    assert old in source, f"Expected exact original text missing: {old}"
    source = source.replace(old, new)
compile(source, str(target / "probe.py"), "exec")
target.mkdir()
(target / "probe.py").write_text(source, encoding="utf-8", newline="\n")
print(str(target / "probe.py"))
