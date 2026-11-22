"""Measure complete official prompt token budgets; no answers or model generation."""
import json
import os
from pathlib import Path
import time
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from acceptance_lawbench import TASKS, source, load_task, prompt
from acceptance_common import ROOT, digest, write_json


def main():
    from transformers import AutoTokenizer
    start = time.perf_counter()
    snapshot = ROOT / 'tools/acceptance_vendor/LawBench'
    source(snapshot)
    folder = ROOT / 'zhilv_qiling_v2/models/pretrained/legalone-8b'
    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    result = {'scope': 'official_zero_shot_all_20_tasks_input_budget_only', 'actual_model_calls': 0,
              'prompt_truncation': False, 'answer_appended': False, 'tasks': {}, 'items': 0}
    maximum = 0
    for task in TASKS:
        path, rows = load_task(snapshot, 'zero_shot', task)
        prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt(row)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True) for row in rows]
        lengths = [len(ids) for ids in tokenizer(prompts, add_special_tokens=False)['input_ids']]
        maximum = max(maximum, max(lengths))
        result['tasks'][task] = {'items': len(lengths), 'min_prompt_tokens': min(lengths),
            'max_prompt_tokens': max(lengths), 'source_sha256': digest(path.read_bytes())}
        result['items'] += len(lengths)
    result.update(max_prompt_tokens=maximum, proposed_output_budget=4096,
        required_context_tokens=maximum+4096, seconds=time.perf_counter()-start)
    write_json(ROOT/'tools/acceptance_results/lawbench_full_input_preflight.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
