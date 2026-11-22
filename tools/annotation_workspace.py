"""Initialize or serve the local manual blind-annotation workspace (no models)."""
from __future__ import annotations
import argparse
import getpass
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zhilv_qiling_v2.annotation_workspace import AnnotationWorkspace, handle_request, page_bytes


def make_server(workspace, host="127.0.0.1", port=8630):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # Access tokens and annotation content never enter access logs.

        def send_payload(self, status, content, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(content)

        def dispatch(self):
            parsed = urlparse(self.path)
            if self.command == "GET" and parsed.path in ("/", "/annotations", "/annotation.html"):
                return self.send_payload(200, page_bytes(), "text/html; charset=utf-8")
            body = None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > 16 * 1024 * 1024:
                    return self.send_payload(413, b'{"error":"request too large"}', "application/json")
                if self.command == "POST":
                    if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                        return self.send_payload(415, b'{"error":"application/json required"}', "application/json")
                    body = json.loads(self.rfile.read(length))
                query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
                result = handle_request(workspace, self.command, parsed.path, body,
                                        self.headers.get("Authorization", ""), query)
                status = result.pop("status", 200)
                self.send_payload(status, json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            except (ValueError, TypeError):
                self.send_payload(400, b'{"error":"invalid JSON or request fields"}', "application/json")
            except Exception:
                self.send_payload(500, b'{"error":"annotation workspace operation failed"}', "application/json")

        do_GET = dispatch
        do_POST = dispatch

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init")
    init.add_argument("--workspace", type=Path, default=ROOT / "data" / "annotation_workspace")
    init.add_argument("--admin-id", required=True, help="Actual administrator identifier; not an invented reviewer")
    serve = sub.add_parser("serve")
    serve.add_argument("--workspace", type=Path, default=ROOT / "data" / "annotation_workspace")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8630)
    load = sub.add_parser("import-candidates")
    load.add_argument("--workspace", type=Path, default=ROOT / "data" / "annotation_workspace")
    load.add_argument("--candidates", type=Path, required=True)
    load.add_argument("--guidelines", type=Path, required=True)
    load.add_argument("--name", required=True)
    args = parser.parse_args()
    workspace = AnnotationWorkspace(args.workspace)
    if args.action == "init":
        print(json.dumps(workspace.bootstrap(args.admin_id), ensure_ascii=False, indent=2))
        return 0
    if args.action == "import-candidates":
        token = getpass.getpass("Actual administrator token (not echoed): ")
        actor = workspace.authenticate(token)
        result = workspace.create_dataset(actor, {"name": args.name,
            "jsonl": args.candidates.read_text(encoding="utf-8-sig"),
            "guidelines": args.guidelines.read_text(encoding="utf-8-sig")})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    server = make_server(workspace, args.host, args.port)
    print(f"Manual annotation workspace: http://{args.host}:{server.server_port}/annotations", flush=True)
    print("No human labels or identities are generated. Initialize explicitly before login.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
