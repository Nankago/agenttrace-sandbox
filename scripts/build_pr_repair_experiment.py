from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from agenttrace_sandbox.data_builders import (
    BUG_FIX_NEGATIVE_KEYWORDS,
    BUG_FIX_POSITIVE_KEYWORDS,
    bug_fix_quality,
    files_from_diff,
    keyword_hits,
    linked_issue_number,
    make_repair_card,
)
from agenttrace_sandbox.sft_export import export_repair_corpus, export_repair_sft


DEFAULT_REPOS = [
    "aio-libs/aiohttp",
    "apache/airflow",
    "ansible/ansible",
    "boto/botocore",
    "celery/celery",
    "django/django",
    "encode/django-rest-framework",
    "encode/httpx",
    "fastapi/fastapi",
    "home-assistant/core",
    "huggingface/transformers",
    "ipython/ipython",
    "jupyter/notebook",
    "jupyterlab/jupyterlab",
    "matplotlib/matplotlib",
    "microsoft/playwright-python",
    "networkx/networkx",
    "numpy/numpy",
    "pallets/flask",
    "pallets/werkzeug",
    "pandas-dev/pandas",
    "psf/requests",
    "pydantic/pydantic",
    "python/mypy",
    "python-poetry/poetry",
    "pytest-dev/pytest",
    "python-pillow/Pillow",
    "scikit-learn/scikit-learn",
    "scrapy/scrapy",
    "sqlalchemy/sqlalchemy",
]

GITHUB_API = "https://api.github.com"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def github_request(path: str, params: dict[str, str] | None = None, accept: str = "application/vnd.github+json") -> str:
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    request = urllib.request.Request(f"{GITHUB_API}{path}{query}")
    request.add_header("Accept", accept)
    request.add_header("User-Agent", "agenttrace-sandbox")
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 403 and exc.headers.get("x-ratelimit-remaining") == "0":
                reset_at = int(exc.headers.get("x-ratelimit-reset") or "0")
                wait_seconds = max(30, reset_at - int(time.time()) + 5)
                print({"event": "github_rate_limit_wait", "path": path, "wait_seconds": wait_seconds}, flush=True)
                time.sleep(wait_seconds)
                continue
            if exc.code in {403, 429, 502, 503, 504} and attempt < 2:
                wait_seconds = 30 * (attempt + 1)
                print({"event": "github_retry", "path": path, "status": exc.code, "wait_seconds": wait_seconds}, flush=True)
                time.sleep(wait_seconds)
                continue
            raise


def github_json(path: str, params: dict[str, str] | None = None) -> Any:
    text = github_request(path, params=params)
    return json.loads(text) if text else {}


def fetch_pull_pages(repo: str, max_prs: int) -> list[dict[str, Any]]:
    pulls: list[dict[str, Any]] = []
    for page in range(1, (max_prs + 99) // 100 + 1):
        batch = github_json(
            f"/repos/{repo}/pulls",
            {
                "state": "closed",
                "per_page": "100",
                "sort": "updated",
                "direction": "desc",
                "page": str(page),
            },
        )
        if not isinstance(batch, list) or not batch:
            break
        pulls.extend(item for item in batch if isinstance(item, dict))
        if len(batch) < 100 or len(pulls) >= max_prs:
            break
        time.sleep(0.2)
    return pulls[:max_prs]


def text_prefilter(pr: dict[str, Any], min_hits: int) -> bool:
    text = "\n".join([str(pr.get("title") or ""), str(pr.get("body") or "")])
    positive = keyword_hits(text, BUG_FIX_POSITIVE_KEYWORDS)
    negative = keyword_hits(text, BUG_FIX_NEGATIVE_KEYWORDS)
    if len(positive) < min_hits:
        return False
    if negative and not any(word in positive for word in ["fix", "bug", "regression", "error", "failure"]):
        return False
    return True


def fetch_pr_record(repo: str, pr: dict[str, Any]) -> dict[str, Any] | None:
    number = int(pr.get("number") or 0)
    if not number:
        return None
    title = str(pr.get("title") or "")
    body = str(pr.get("body") or "")
    issue_number = linked_issue_number(title, body)
    issue: dict[str, Any] = {}
    if issue_number:
        try:
            raw_issue = github_json(f"/repos/{repo}/issues/{issue_number}")
            issue = raw_issue if isinstance(raw_issue, dict) else {}
        except Exception:
            issue = {}
    try:
        diff = github_request(f"/repos/{repo}/pulls/{number}", accept="application/vnd.github.v3.diff")
    except Exception:
        diff = ""
    if not diff:
        return None
    record = {
        "id": f"{repo}#{number}",
        "repo": repo,
        "pr_number": number,
        "pr_title": title,
        "pr_body": body,
        "pr_url": pr.get("html_url", ""),
        "issue_number": issue_number,
        "issue_title": issue.get("title", ""),
        "issue_body": issue.get("body", ""),
        "diff": diff,
        "files": files_from_diff(diff),
    }
    record.update(bug_fix_quality(record))
    return record


def high_quality_record(record: dict[str, Any], min_bug_score: float) -> bool:
    if float(record.get("bug_fix_score", 0)) < min_bug_score:
        return False
    if not record.get("is_bug_fix"):
        return False
    if not record.get("source_files"):
        return False
    if not record.get("test_files"):
        return False
    if record.get("docs_only") or record.get("tests_only") or record.get("dependency_only"):
        return False
    if record.get("low_signal_only") or record.get("large_patch"):
        return False
    return True


def load_cached_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            source_id = str(row.get("id") or "")
            if not source_id or source_id in seen:
                continue
            sanitize_cached_issue_link(row)
            row.update(bug_fix_quality(row))
            records.append(row)
            seen.add(source_id)
    return records


def sanitize_cached_issue_link(record: dict[str, Any]) -> None:
    linked = linked_issue_number(str(record.get("pr_title") or ""), str(record.get("pr_body") or ""))
    if linked == record.get("issue_number"):
        return
    record["issue_number"] = linked
    record["issue_title"] = ""
    record["issue_body"] = ""


def fetch_repo_records(
    repo: str,
    max_prs_per_repo: int,
    min_text_hits: int,
    min_bug_score: float,
    workers: int,
) -> list[dict[str, Any]]:
    pulls = [pr for pr in fetch_pull_pages(repo, max_prs_per_repo) if text_prefilter(pr, min_text_hits)]
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_pr_record, repo, pr) for pr in pulls]
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception:
                continue
            if record and high_quality_record(record, min_bug_score):
                records.append(record)
    return sorted(records, key=lambda item: int(item.get("pr_number") or 0), reverse=True)


def select_cards(records: list[dict[str, Any]], target: int, min_card_quality: float) -> list[dict[str, Any]]:
    cards = []
    seen: set[str] = set()
    for record in records:
        source_id = str(record.get("id") or "")
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        if not high_quality_record(record, min_bug_score=0):
            continue
        card = make_repair_card(record)
        quality = card.get("quality", {}) if isinstance(card.get("quality"), dict) else {}
        if float(quality.get("overall", 0)) < min_card_quality:
            continue
        if not quality.get("has_test_evidence") or not quality.get("has_source_patch"):
            continue
        cards.append(card)
    cards.sort(
        key=lambda item: (
            float(item.get("quality", {}).get("overall", 0)),
            float(item.get("quality", {}).get("bug_fix_score", 0)),
            len(item.get("test_files", [])),
        ),
        reverse=True,
    )
    return cards[:target]


def card_source_ids(cards: list[dict[str, Any]]) -> set[str]:
    return {str(card.get("id")) for card in cards if card.get("id")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/experiments/second_stage_2k"))
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--max-prs-per-repo", type=int, default=500)
    parser.add_argument("--min-text-hits", type=int, default=1)
    parser.add_argument("--min-bug-score", type=float, default=3.0)
    parser.add_argument("--min-card-quality", type=float, default=0.85)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repo", action="append", dest="repos")
    parser.add_argument("--cache", action="append", type=Path, default=[])
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args()

    repos = args.repos or DEFAULT_REPOS
    out = args.output_root
    github_dir = out / "github"
    github_dir.mkdir(parents=True, exist_ok=True)

    default_cache = sorted(Path("data/experiments/first_stage_500/github").glob("*.jsonl"))
    cached_records = load_cached_records(default_cache + args.cache)
    all_records = list(cached_records)
    fetch_summary: dict[str, Any] = {
        "target": args.target,
        "repos": repos,
        "cached_records": len(cached_records),
        "fetched": {},
        "token_present": bool(os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")),
    }

    if not args.no_fetch:
        if not fetch_summary["token_present"]:
            raise SystemExit("GITHUB_TOKEN or GH_TOKEN is required for fetching enough PRs for the 2k experiment.")
        for repo in repos:
            cache_path = github_dir / f"{repo.replace('/', '_')}.jsonl"
            print({"event": "fetch_repo", "repo": repo}, flush=True)
            if cache_path.exists():
                records = load_cached_records([cache_path])
                source = "cache"
            else:
                try:
                    records = fetch_repo_records(
                        repo,
                        max_prs_per_repo=args.max_prs_per_repo,
                        min_text_hits=args.min_text_hits,
                        min_bug_score=args.min_bug_score,
                        workers=args.workers,
                    )
                except Exception as exc:
                    print({"event": "fetch_repo_error", "repo": repo, "error": f"{type(exc).__name__}: {exc}"}, flush=True)
                    records = []
                source = "github"
                write_jsonl(cache_path, records)
            all_records.extend(records)
            fetch_summary["fetched"][repo] = len(records)
            cards_so_far = select_cards(all_records, args.target, args.min_card_quality)
            print({"event": "repo_done", "repo": repo, "source": source, "records": len(records), "selected_cards_so_far": len(cards_so_far)}, flush=True)
            if len(cards_so_far) >= args.target:
                break

    cards = select_cards(all_records, args.target, args.min_card_quality)
    selected_ids = card_source_ids(cards)
    selected_records = [record for record in all_records if str(record.get("id")) in selected_ids]
    selected_records.sort(key=lambda item: str(item.get("id")))

    write_jsonl(github_dir / "selected_prs.jsonl", selected_records)
    write_jsonl(out / "wiki" / "repair_cards.jsonl", cards)
    corpus_path = out / "corpus" / "repair_corpus_semantic.jsonl"
    sft_path = out / "sft" / "repair_sft_diff_only.jsonl"
    corpus_count = export_repair_corpus(
        out / "wiki" / "repair_cards.jsonl",
        corpus_path,
        min_quality=args.min_card_quality,
        max_evidence_chars=1200,
        boilerplate_policy="semantic",
    )
    sft_count = export_repair_sft(
        out / "wiki" / "repair_cards.jsonl",
        sft_path,
        variant="diff-only",
        boilerplate_policy="semantic",
    )

    summary = {
        **fetch_summary,
        "records_total_before_selection": len({str(record.get("id")) for record in all_records if record.get("id")}),
        "selected_prs": len(selected_records),
        "repair_cards": len(cards),
        "corpus_records": corpus_count,
        "sft_records": sft_count,
        "quality_avg": sum(float(card.get("quality", {}).get("overall", 0)) for card in cards) / len(cards) if cards else 0,
        "has_test_evidence": sum(1 for card in cards if card.get("quality", {}).get("has_test_evidence")),
        "has_source_patch": sum(1 for card in cards if card.get("quality", {}).get("has_source_patch")),
        "output_root": str(out),
    }
    (out / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
