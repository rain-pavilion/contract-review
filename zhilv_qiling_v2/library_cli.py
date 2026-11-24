"""Host-only administration for contracts.sqlite and laws.sqlite.

Passwords are read interactively or from owner-readable local files, never
command-line arguments. Import reads legacy SQLite in read-only mode.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path


def _output(value):
    print(json.dumps(value, ensure_ascii=True, indent=2, default=str))


def _private_file(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        if os.name == "nt":
            result = subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r",
                getpass.getuser() + ":(R,W)"], capture_output=True)
            if result.returncode:
                raise OSError("无法设置凭据文件私有权限")
        else:
            path.chmod(0o600)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _password(args):
    if getattr(args, "password_file", None):
        value = Path(args.password_file).read_text(encoding="utf-8-sig").strip()
        if value.startswith("{"):
            return json.loads(value)["password"]
        return value
    first = getpass.getpass("Password (at least 12 characters): ")
    if first != getpass.getpass("Repeat password: "):
        raise ValueError("两次口令不一致")
    return first


def _admin(args, store):
    if args.admin_credentials:
        credentials = json.loads(Path(args.admin_credentials).read_text(encoding="utf-8-sig"))
        username, password = credentials["username"], credentials["password"]
    else:
        username = input("Administrator username: ")
        password = getpass.getpass("Administrator password: ")
    session = store.login(username, password)
    actor = store.authenticate(session["token"])
    if actor["role"] != "admin":
        store.logout(session["token"])
        raise ValueError("需要管理员账户")
    return actor, session["token"]


def parser():
    p = argparse.ArgumentParser(description="合同与法条双库管理（仅在服务器本机运行）")
    p.add_argument("--root", help="双库目录；默认data/libraries")
    p.add_argument("--admin-credentials", help="本机管理员凭据JSON文件，不会输出其内容")
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-admin", help="仅空库允许初始化首位管理员")
    init.add_argument("--username", default="admin")
    init.add_argument("--password-file")
    init.add_argument("--generate-credentials", metavar="FILE", help="生成随机口令并只保存到私有文件")
    user = sub.add_parser("create-user")
    user.add_argument("--username", required=True)
    user.add_argument("--role", choices=["user", "admin"], default="user")
    user.add_argument("--password-file")
    sub.add_parser("status")
    migration = sub.add_parser("migrate", help="无显式路径时导入当前主工作台/答辩/旧库和法条")
    migration.add_argument("--workspace", action="append", default=[])
    migration.add_argument("--jsonl", action="append", default=[])
    crawl = sub.add_parser("crawl", help="从配置中的官方来源执行有界法条采集")
    crawl.add_argument("--source", action="append")
    crawl.add_argument("--max-documents", type=int, default=10)
    sub.add_parser("sources")
    verify = sub.add_parser("verify", help="人工核对后记录来源、效力及日期；不是自动法律核验")
    verify.add_argument("--version", required=True)
    verify.add_argument("--validity", choices=["effective", "unknown", "abolished", "expired", "not_yet_effective"], required=True)
    verify.add_argument("--effective-from")
    verify.add_argument("--effective-to")
    verify.add_argument("--region")
    verify.add_argument("--domain", action="append")
    verify.add_argument("--evidence", required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("--version", required=True)
    publish.add_argument("--note", default="")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.root:
        os.environ["ZLQL_LIBRARY_ROOT"] = str(Path(args.root).resolve())
    from .library_api import libraries, library_paths, user_library_path
    from .config import WORKSPACE
    store, laws = libraries()
    token = None
    try:
        if args.command == "status":
            _output({"initialized": store.initialized(), "contracts_db": str(library_paths()[0]),
                "laws_db": str(library_paths()[1]), "users_db": str(user_library_path()),
                "laws": laws.stats(include_unpublished=True)})
            return 0
        if args.command == "sources":
            from .law_crawler import source_catalog
            _output({"items": source_catalog()})
            return 0
        if args.command == "init-admin":
            if store.initialized():
                raise ValueError("管理员已初始化；不会覆盖账户或凭据文件")
            credential_file = None
            if args.generate_credentials:
                password = secrets.token_urlsafe(27)
                credential_file = _private_file(args.generate_credentials,
                    {"username": args.username, "password": password,
                     "notice": "本机初始化凭据，请登录后妥善保存；不要加入版本控制或发送给他人。"})
            else:
                password = _password(args)
            try:
                actor = store.bootstrap_admin(args.username, password)
            except Exception:
                if credential_file:
                    credential_file.unlink(missing_ok=True)
                raise
            _output({"initialized": True, "username": actor["username"],
                "credentials_file": str(credential_file) if credential_file else None})
            return 0
        actor, token = _admin(args, store)
        if args.command == "create-user":
            _output(store.create_user(actor, args.username, _password(args), role=args.role))
        elif args.command == "migrate":
            workspaces = [Path(p) for p in args.workspace]
            jsonls = [Path(p) for p in args.jsonl]
            if not workspaces and not jsonls:
                workspaces = [WORKSPACE / "zhilv_qiling_v2/reports" / name for name in
                    ("workspace.sqlite", "defense_workspace_20261121.sqlite")]
                workspaces.append(WORKSPACE / "data/reviews.sqlite")
                jsonls = [WORKSPACE / "data/法条结构化数据.jsonl",
                    WORKSPACE / "data/contract_packages/beijing_procurement_20261121/statutes.jsonl"]
            results = {"contracts": [], "laws": []}
            for path in workspaces:
                if not path.is_file():
                    results["contracts"].append({"source": str(path), "error": "文件不存在"})
                    continue
                results["contracts"].append({"source": str(path), **store.import_legacy(actor, path)})
            for path in jsonls:
                if not path.is_file():
                    results["laws"].append({"source": str(path), "error": "文件不存在"})
                    continue
                results["laws"].append({"source": str(path), **laws.import_jsonl(actor, path)})
            results["ownership"] = "未知原上传者的历史记录仅管理员可见；原始文件未被修改"
            results["law_status"] = "导入法条均需审核；不自动发布或推定现行有效"
            _output(results)
        elif args.command == "crawl":
            from .law_crawler import crawl_sources
            _output(crawl_sources(laws, actor, source_ids=args.source, max_documents=args.max_documents))
        elif args.command == "verify":
            changes = {"validity": args.validity, "verification": {"verified": True, "evidence": args.evidence}}
            for key in ("effective_from", "effective_to", "region"):
                if getattr(args, key):
                    changes[key] = getattr(args, key)
            if args.domain:
                changes["domains"] = args.domain
            _output(laws.set_version_metadata(actor, args.version, changes))
        elif args.command == "publish":
            _output(laws.publish_version(actor, args.version, note=args.note))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        # Exceptions never include password, secret token or contract contents.
        print(json.dumps({"error": str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 1
    finally:
        if token:
            store.logout(token)


if __name__ == "__main__":
    raise SystemExit(main())
