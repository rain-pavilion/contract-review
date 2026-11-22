"""
Hugging Face Spaces 入口（Docker SDK 或 Gradio-free 静态服务都可）。

Space 只暴露一个端口，默认 7860；本文件把 v2 的 HTTP 服务绑到 0.0.0.0:7860 后
直接 serve_forever，前端就是包内的 webui.html。

    # Space 的 Dockerfile 里最后一行写：
    CMD ["python", "deploy/huggingface/app.py"]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("ZLQL_HOST", "0.0.0.0")
os.environ.setdefault("ZLQL_PORT", os.environ.get("PORT", "7860"))

# Space 的工作目录是仓库根，包在根下的 zhilv_qiling_v2/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from zhilv_qiling_v2.api import serve  # noqa: E402


def main() -> int:
    httpd = serve(verbose=True)
    print(f"[space] 已就绪 http://0.0.0.0:{httpd.server_address[1]}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
