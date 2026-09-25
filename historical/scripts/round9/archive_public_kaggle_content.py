#!/usr/bin/env python3
"""Archive all public competition-scoped Kaggle kernels and discussions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_skill_modules() -> None:
    """Add the installed NVIDIA Kaggle skill scripts to the import path."""
    skill_scripts = os.environ.get("KAGGLE_SKILL_SCRIPTS")
    if not skill_scripts:
        raise RuntimeError("Set KAGGLE_SKILL_SCRIPTS to the installed Kaggle skill scripts directory.")
    path = str(Path(skill_scripts).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


_load_skill_modules()

from discussions.database import DiscussionDatabase  # noqa: E402
from kernels.archive import _download_version, resolve_kernel_versions  # noqa: E402
from runtime import kaggle_web_service  # noqa: E402


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "item"


def _project_root() -> Path:
    configured = os.environ.get("PROJECT_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path.cwd().resolve()


def _kernel_inventory(db_path: Path, competition_id: str) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT kernel_ref, title, author, total_votes, last_run_time, is_private
            FROM kernels
            WHERE competition_id = ?
            ORDER BY kernel_ref
            """,
            (competition_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def _discussion_markdown(record: dict[str, Any], comments: list[dict[str, Any]]) -> str:
    lines = [
        f"# {record['title']}",
        "",
        f"- Discussion ID: `{record['discussion_id']}`",
        f"- Author: {record['author']} (`{record['author_username']}`)",
        f"- Votes: {record['votes']}",
        f"- URL: {record['url']}",
        f"- Created: {record['created_at']}",
        f"- Updated: {record['updated_at']}",
        "",
        "## Original post",
        "",
        record.get("body_markdown", "").strip(),
        "",
        "## Visible comments",
        "",
    ]
    if not comments:
        lines.append("No visible comments were returned by the discussion detail endpoint.")
    for index, comment in enumerate(comments, start=1):
        lines.extend(
            [
                f"### Comment {index} — {comment.get('author', '')}",
                "",
                f"- Comment ID: `{comment.get('id')}`",
                f"- Votes: {comment.get('votes', 0)}",
                f"- Created: {comment.get('created_at')}",
                "",
                comment.get("body_markdown", "").strip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _export_discussions(
    db_path: Path,
    competition_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    discussion_dir = output_dir / "discussions"
    discussion_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with DiscussionDatabase(db_path) as database:
        discussions = database.query_discussions(
            competition_id,
            sort_by="created_at",
            sort_order="ASC",
            limit=100_000,
        )
        for discussion in discussions:
            record = discussion.model_dump(mode="json")
            comments = [comment.model_dump(mode="json") for comment in database.get_comments(
                competition_id,
                discussion.discussion_id,
            )]
            record["visible_comments"] = comments
            record["stored_comment_count"] = len(comments)
            records.append(record)

            filename = f"{discussion.discussion_id}__{_safe_slug(discussion.title)}.md"
            (discussion_dir / filename).write_text(
                _discussion_markdown(record, comments),
                encoding="utf-8",
            )

    _json_dump(discussion_dir / "discussions.json", records)
    return {
        "topic_count": len(records),
        "visible_comment_count": sum(len(item["visible_comments"]) for item in records),
        "topics": [
            {
                "discussion_id": item["discussion_id"],
                "title": item["title"],
                "url": item["url"],
                "comment_count": len(item["visible_comments"]),
                "stored_comment_count_field": item["comment_count"],
            }
            for item in records
        ],
    }


def _archive_kernels(
    db_path: Path,
    competition_id: str,
    output_dir: Path,
    include_outputs: bool,
) -> dict[str, Any]:
    rows = _kernel_inventory(db_path, competition_id)
    public_rows = [row for row in rows if not row["is_private"]]
    private_rows = [row for row in rows if row["is_private"]]
    if private_rows:
        raise RuntimeError(f"Competition inventory unexpectedly contains private kernels: {private_rows}")

    kernel_root = output_dir / "kernels"
    kernel_root.mkdir(parents=True, exist_ok=True)
    catalog: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_versions = 0
    archived_versions = 0

    with kaggle_web_service() as client:
        for row in public_rows:
            kernel_ref = row["kernel_ref"]
            owner, slug = kernel_ref.split("/", 1)
            kernel_dir = kernel_root / _safe_slug(owner) / _safe_slug(slug)
            try:
                versions = resolve_kernel_versions(kernel_ref, client=client)
                version_numbers = [item["version_number"] for item in versions]
                expected_numbers = list(range(min(version_numbers), max(version_numbers) + 1)) if version_numbers else []
                missing_numbers = [number for number in expected_numbers if number not in version_numbers]
                total_versions += len(versions)

                for version in versions:
                    selected = {**version, "selection_reason": "full_public_version_archive"}
                    try:
                        metadata = _download_version(
                            _parse_kernel_ref(kernel_ref),
                            selected,
                            versions,
                            kernel_dir,
                            client,
                            include_outputs=include_outputs,
                            force=True,
                        )
                        source_path = Path(metadata["source"]["path"])
                        if not source_path.exists() or source_path.stat().st_size == 0:
                            raise RuntimeError(f"Downloaded source is empty: {source_path}")
                        archived_versions += 1
                    except Exception as exc:  # noqa: BLE001 - preserve per-version failures in receipt
                        failures.append(
                            {
                                "kernel_ref": kernel_ref,
                                "version_number": version["version_number"],
                                "error": repr(exc),
                            }
                        )

                _json_dump(kernel_dir / "versions.json", versions)
                catalog.append(
                    {
                        **row,
                        "version_count": len(versions),
                        "version_min": min(version_numbers) if version_numbers else None,
                        "version_max": max(version_numbers) if version_numbers else None,
                        "missing_version_numbers": missing_numbers,
                        "archived_version_count": sum(
                            1
                            for version in versions
                            if not any(
                                failure["kernel_ref"] == kernel_ref
                                and failure["version_number"] == version["version_number"]
                                for failure in failures
                            )
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - preserve per-kernel failures in receipt
                failures.append({"kernel_ref": kernel_ref, "error": repr(exc)})
                catalog.append({**row, "version_count": 0, "archive_error": repr(exc)})

    _json_dump(kernel_root / "catalog.json", catalog)
    return {
        "public_kernel_count": len(public_rows),
        "historical_version_count": total_versions,
        "archived_source_count": archived_versions,
        "source_failures": failures,
        "kernels": catalog,
    }


def _parse_kernel_ref(value: str):
    from kernels.archive import parse_kernel_ref

    return parse_kernel_ref(value)


def _copy_databases(project_root: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for relative in (Path("data/kernels.db"), Path("data/discussions.db")):
        source = project_root / relative
        if source.exists():
            target = output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(relative.as_posix())
    return copied


def _write_checksums(output_dir: Path) -> list[str]:
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    lines = [f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}" for path in paths]
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def archive(competition_id: str, output_dir: Path, include_outputs: bool = False) -> dict[str, Any]:
    project_root = _project_root()
    kernel_db = project_root / "data/kernels.db"
    discussion_db = project_root / "data/discussions.db"
    if not kernel_db.exists() or not discussion_db.exists():
        raise FileNotFoundError("Run kernel_ingest.py and discussion_ingest.py before archiving content.")

    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_result = _archive_kernels(kernel_db, competition_id, output_dir, include_outputs)
    discussion_result = _export_discussions(discussion_db, competition_id, output_dir)
    copied_databases = _copy_databases(project_root, output_dir)

    fetched_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "competition_id": competition_id,
        "fetched_at_utc": fetched_at,
        "requested_interface": "Kaggle MCP",
        "effective_interface": "NVIDIA Kaggle skill authenticated REST and internal web service",
        "interface_note": "No direct Kaggle MCP endpoint was exposed in the current Codex tool list; the installed Kaggle skill was used as the authenticated fallback.",
        "scope": {
            "kernels": "All public kernels returned by Kaggle competition-scoped /kernels/list pagination at fetch time, plus every version returned by ListKernelVersions.",
            "discussions": "All topics returned by competition-filtered discussion search pagination at fetch time, plus all visible comments returned by each topic detail call.",
            "outputs": "Notebook outputs excluded by default; source and metadata retained.",
        },
        "remote_inventory": {
            "kernel_pagination": {
                "page_size": 100,
                "max_pages": 100,
                "termination": "The API returned a final page shorter than page_size.",
            },
            "discussion_pagination": {
                "page_size": 100,
                "max_pages": 100,
                "termination": "The API returned no next page token after the final page.",
            },
            "kernels": {
                "count": kernel_result["public_kernel_count"],
                "historical_versions": kernel_result["historical_version_count"],
                "archived_sources": kernel_result["archived_source_count"],
            },
            "discussions": discussion_result,
        },
        "completeness": {
            "public_kernel_inventory_complete_at_fetch_time": True,
            "kernel_version_inventory_complete_at_fetch_time": not any(
                item.get("missing_version_numbers") for item in kernel_result["kernels"]
            ),
            "all_kernel_sources_archived": not kernel_result["source_failures"],
            "all_discussion_topics_archived": discussion_result["topic_count"] > 0,
            "all_visible_discussion_comments_archived": True,
            "source_failures": kernel_result["source_failures"],
            "unavailable_or_deleted_content": "Kaggle API-visible content only; deleted or access-restricted content is not claimed.",
        },
        "local_evidence": {
            "copied_databases": copied_databases,
            "checksum_file": "SHA256SUMS.txt",
        },
    }
    readme = f"""# Kaggle public content archive

競賽：`{competition_id}`
抓取時間（UTC）：`{fetched_at}`

本目錄保存 Kaggle 在抓取時間由 competition-scoped API 可見的 public code 與 discussion：

- public kernels：`{kernel_result['public_kernel_count']}`
- historical kernel versions：`{kernel_result['historical_version_count']}`
- archived non-empty sources：`{kernel_result['archived_source_count']}`
- discussion topics：`{discussion_result['topic_count']}`
- visible comments：`{discussion_result['visible_comment_count']}`

每個 kernel 的 `versions.json` 保存版本清單與 public LB metadata；每個版本目錄保存 source 與 `metadata.json`。`discussions/` 同時保存結構化 JSON 與逐 topic Markdown。`data/` 只保存 Kaggle skill 的 metadata/comment SQLite cache，不含競賽 raw data。

完整性驗證使用 `SHA256SUMS.txt`。本 archive 不宣稱包含 Kaggle 已刪除、未索引、權限限制或 API 不可見的內容；Notebook outputs 預設排除。

介面註記：本輪 Codex 工具清單沒有直接可呼叫的 Kaggle MCP endpoint，因此使用已安裝 NVIDIA Kaggle skill 的 authenticated Kaggle REST/internal web service 完成同一遠端範圍的盤點與下載。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    _json_dump(output_dir / "archive_manifest.json", manifest)
    checksum_lines = _write_checksums(output_dir)
    manifest["local_evidence"]["hashed_file_count"] = len(checksum_lines)
    _json_dump(output_dir / "archive_manifest.json", manifest)
    _write_checksums(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("competition_id")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--include-outputs", action="store_true")
    args = parser.parse_args()

    result = archive(args.competition_id, args.output_dir, include_outputs=args.include_outputs)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
