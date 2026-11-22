"""诊断 LegalOne 输出尾部乱码：tokenizer 与权重词表是否对齐、EOS 是否生效。"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zhilv_qiling_v2 import local_llm  # noqa: E402

MODEL_DIR = ROOT / "zhilv_qiling_v2" / "models" / "pretrained" / "legalone-1.7b"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype="auto", trust_remote_code=True)
    model = model.to("cuda").eval()

    print("tokenizer.vocab_size:", tok.vocab_size, "| len(tok):", len(tok))
    print("model.config.vocab_size:", model.config.vocab_size)
    print("eos:", tok.eos_token_id, repr(tok.eos_token), "| pad:", tok.pad_token_id)
    print("added_tokens:", [t.content for t in getattr(tok, "added_tokens_decoder", {}).values()][:8])

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": '只回 JSON：{"ok": true}'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            repetition_penalty=1.05,
        )
    gen = out[0][inputs["input_ids"].shape[-1]:]
    print("生成 token ids 尾部:", gen[-10:].tolist())
    print("decode(skip_special=True):", repr(tok.decode(gen, skip_special_tokens=True)))
    print("decode(skip_special=False)[:160]:", repr(tok.decode(gen, skip_special_tokens=False)[:160]))


if __name__ == "__main__":
    main()
