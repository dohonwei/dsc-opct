from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex"
CHECKLIST = ROOT / "docs/elsarticle/dcs_opct_v11_author_confirmation_checklist.md"

TODO_LINE = "% TODO(author): add anonymized repository URL or archival DOI before submission."
PLACEHOLDER = (
    "An anonymized, checksum-verified reproducibility package has been prepared for deposition "
    "in a read-only repository or archival service; its permanent URL or DOI will be inserted "
    "here before submission."
)
CHECKLIST_OPEN = (
    "- [ ] **匿名代码入口**：将 `outputs/dcs_opct_v11_submission_artifacts_crossfit_v22.zip` "
    "上传到匿名只读仓库/归档平台，在主文 `Data and code availability` 中替换 URL 或 DOI "
    "占位说明，并实际从该入口下载一次、核对旁车 SHA-256 后再提交。"
)


def validate_url(value: str) -> str:
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Repository location must be a complete HTTPS URL")
    if any(char in value for char in "{}\\\n\r"):
        raise ValueError("Repository URL contains characters unsafe for a LaTeX \\url field")
    return value


def exact_once(text: str, needle: str, label: str) -> None:
    count = text.count(needle)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}; found {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize the public repository link in the DCS-OPCT BSPC submission."
    )
    parser.add_argument("repository_url", help="Verified public HTTPS repository or archival DOI URL")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the replacement. Without this flag, perform a dry-run validation only.",
    )
    args = parser.parse_args()
    repository_url = validate_url(args.repository_url)

    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    checklist = CHECKLIST.read_text(encoding="utf-8")
    exact_once(manuscript, TODO_LINE, "repository TODO line")
    exact_once(manuscript, PLACEHOLDER, "repository placeholder sentence")
    exact_once(checklist, CHECKLIST_OPEN, "open repository checklist item")

    replacement = (
        "A checksum-verified reproducibility package, source code, aggregate outputs, and validation "
        f"reports are publicly available at \\url{{{repository_url}}}."
    )
    checklist_done = (
        "- [x] **公开代码入口**：已发布 v22 校验归档、源代码、聚合结果和验证报告，"
        f"并核对远端 main 提交、SHA-256 旁车文件与本地上传归档。入口：{repository_url}。"
    )

    print(f"Validated repository URL: {repository_url}")
    print(f"Manuscript target: {MANUSCRIPT}")
    print(f"Checklist target: {CHECKLIST}")
    if not args.apply:
        print("Dry run passed; no files changed. Re-run with --apply after download verification.")
        return

    manuscript = manuscript.replace(TODO_LINE + "\n", "", 1).replace(PLACEHOLDER, replacement, 1)
    checklist = checklist.replace(CHECKLIST_OPEN, checklist_done, 1)
    MANUSCRIPT.write_text(manuscript, encoding="utf-8")
    CHECKLIST.write_text(checklist, encoding="utf-8")
    print("Repository link finalized. Recompile both PDFs and build a new immutable submission package.")


if __name__ == "__main__":
    main()
