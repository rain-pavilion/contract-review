"""探《政府采购法》《实施条例》《招标投标法》全文的公开可取性（只探测，不写入库）。"""

from __future__ import annotations

import re
import urllib.error
import urllib.request

CANDIDATES = [
    ("政府采购法·国务院公报", "https://www.gov.cn/gongbao/content/2002/content_61616.htm"),
    ("政府采购法·人大网法律库", "http://www.npc.gov.cn/npc/c2/c30834/201905/t20190521_296505.html"),
    ("政府采购法实施条例·国务院公报", "https://www.gov.cn/zhengce/content/2015-02/27/content_9501.htm"),
    ("招标投标法·国务院公报", "https://www.gov.cn/gongbao/content/2017/content_5219124.htm"),
    ("财政部·政采法专栏", "https://www.mof.gov.cn/zhengwuxinxi/zhengcefabu/"),
]

KEY = ("政府采购法", "第八十五条", "集中采购", "招标投标法", "第六十八条")

for label, url in CANDIDATES:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
        text = body.decode("utf-8", errors="replace")
        flat = re.sub(r"<[^>]+>", "", text)
        hits = [k for k in KEY if k in flat]
        arts = sorted(set(re.findall(r"第[一二三四五六七八九十百零]{1,7}条", flat)))[:4]
        print(f"[{resp.status}] {label:26s} {len(body):>7d} B  关键词 {hits[:3]}  条文样例 {arts}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[ERR] {label:26s} {type(exc).__name__}: {str(exc)[:60]}")
