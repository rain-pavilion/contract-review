"""修 bench.py 里两处调用：Retriever 没有 use_rerank 参数、rewrite_query 的签名要核对。"""

from __future__ import annotations

import pathlib

path = pathlib.Path("zhilv_qiling_v2/bench.py")
text = path.read_text(encoding="utf-8")

old = '''    retriever_plain = Retriever(index, config, ranker=None, use_rerank=False)
    retriever_ltr = Retriever(index, config, ranker=ranker, use_rerank=bool(ranker))'''
new = '''    retriever_plain = Retriever(index, config, ranker=None)
    retriever_ltr = Retriever(index, config, ranker=ranker)'''
assert old in text
text = text.replace(old, new, 1)

# use_rerank 是 search() 的关键字参数：纯 RRF 与 +LTR 各走一遍
old_search = '''                result = retriever.search(plan, top_k=top_k)'''
new_search = '''                result = retriever.search(
                    plan, top_k=top_k, use_rerank=(name == "rrf_ltr" and ranker is not None)
                )'''
assert old_search in text
text = text.replace(old_search, new_search, 1)

# 顺手清掉 _load_retrieval_models 里那行占位（会误导后来人）
old_placeholder = '''    index = Retriever.__mro__  # 占位，避免 lint 抱怨；真正的索引在下面构造
    from .retrieve import StatuteIndex

    built = StatuteIndex(statutes, config)
    return statutes, specs, ranker, built'''
new_placeholder = '''    from .retrieve import StatuteIndex

    return statutes, specs, ranker, StatuteIndex(statutes, config)'''
assert old_placeholder in text
text = text.replace(old_placeholder, new_placeholder, 1)

# 未使用的导入清掉
text = text.replace("from .retrieve import Retriever, load_statutes", "from .retrieve import load_statutes", 1)

path.write_text(text, encoding="utf-8")
print("bench.py 签名已修")
