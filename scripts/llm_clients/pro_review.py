#!/usr/bin/env python3
"""External pro review of a project DECOMPOSITION (deep-project).

Sends the proposed split structure (project-manifest.md) plus the original
requirements and interview to OpenAI gpt-5.5-pro for a second-opinion critique,
then writes the review to <planning_dir>/reviews/pro-review.md.

gpt-5.5-pro is a reasoning model served only on the OpenAI Responses API
(/v1/responses), not /v1/chat/completions.

Usage:
  uv run pro_review.py --planning-dir /path/to/planning [--requirements /path/to/req.md]

Returns JSON on stdout. Review is advisory: the calling agent decides what to
integrate. Exits 0 on success, 1 if no review could be produced.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

MODEL_DEFAULT = "gpt-5.5-pro"
TIMEOUT_SECONDS = 900  # pro reasoning over a decomposition can run several minutes

SYSTEM_PROMPT = (
    "You are a senior software architect giving a second opinion on a PROJECT "
    "DECOMPOSITION: how a vague, high-level project was split into independently "
    "plannable units (each will later be handed to a deep-planning step). The "
    "calling agent will integrate your feedback before finalizing, so be concrete "
    "and decisive. Evaluate:\n"
    "1. Right-sizing — is each unit independently plannable and implementable, "
    "neither too large (should be split) nor too granular (should be merged)?\n"
    "2. Completeness — are there missing units, scope gaps, or implied work not "
    "captured by any split?\n"
    "3. Dependencies & order — is the declared execution order correct and "
    "acyclic? Are any cross-unit dependencies wrong or missing?\n"
    "4. Boundaries — are responsibilities cleanly separated, or do units overlap "
    "in a way that will cause rework?\n"
    "5. Sequencing risk — anything that should be built first to de-risk the rest?\n"
    "Output a prioritized, specific critique with actionable changes. If the "
    "decomposition is sound, say so plainly and list only genuine improvements. "
    "Do not invent requirements that are not present."
)


def ensure_api_keys_loaded():
    """Populate OPENAI_API_KEY from common .env files if unset (VAN: ~/van-agents/.env)."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    for f in (Path.home() / ".env", Path.home() / "van-agents" / ".env", Path.home() / ".zshenv"):
        if not f.exists():
            continue
        try:
            for raw in f.read_text().splitlines():
                line = raw.strip()
                if line.startswith("export "):
                    line = line[len("export "):]
                if line.startswith("OPENAI_API_KEY=") and "=" in line:
                    os.environ["OPENAI_API_KEY"] = line.partition("=")[2].strip().strip('"').strip("'")
                    return
        except OSError:
            continue


def _read_if_exists(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def build_user_prompt(planning_dir: Path, requirements_file: Path | None) -> str:
    manifest = _read_if_exists(planning_dir / "project-manifest.md")
    interview = _read_if_exists(planning_dir / "deep_project_interview.md")
    requirements = _read_if_exists(requirements_file) if requirements_file else ""

    parts = []
    if requirements:
        parts.append("## ORIGINAL REQUIREMENTS\n\n" + requirements)
    if interview:
        parts.append("## INTERVIEW / CLARIFICATIONS\n\n" + interview)
    parts.append("## PROPOSED DECOMPOSITION (project-manifest.md)\n\n" + manifest)
    parts.append(
        "Review the proposed decomposition above against the requirements. "
        "Give your prioritized critique."
    )
    return "\n\n---\n\n".join(parts)


def main():
    ensure_api_keys_loaded()

    parser = argparse.ArgumentParser(description="Pro review of a project decomposition")
    parser.add_argument("--planning-dir", required=True, type=Path)
    parser.add_argument("--requirements", type=Path, default=None,
                        help="Path to the original requirements markdown (optional)")
    args = parser.parse_args()

    manifest_path = args.planning_dir / "project-manifest.md"
    if not manifest_path.exists():
        print(json.dumps({"success": False, "error": f"project-manifest.md not found in {args.planning_dir}"}))
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(json.dumps({"success": False, "error": "OPENAI_API_KEY not set (checked env + ~/.env + ~/van-agents/.env)"}))
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print(json.dumps({"success": False, "error": "openai package not installed"}))
        sys.exit(1)

    model = os.environ.get("OPENAI_MODEL", MODEL_DEFAULT)
    user_prompt = build_user_prompt(args.planning_dir, args.requirements)

    try:
        client = OpenAI(api_key=api_key, timeout=TIMEOUT_SECONDS)
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        analysis = response.output_text
    except Exception as e:
        print(json.dumps({"success": False, "model": model, "error": str(e)}))
        sys.exit(1)

    reviews_dir = args.planning_dir / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    review_file = reviews_dir / "pro-review.md"
    review_file.write_text(
        f"# Decomposition Review (OpenAI {model})\n\n"
        f"**Generated:** {datetime.now().isoformat()}\n\n---\n\n{analysis}\n"
    )

    print(json.dumps({
        "success": True,
        "provider": "openai",
        "model": model,
        "review_file": str(review_file),
        "analysis_chars": len(analysis),
    }))


if __name__ == "__main__":
    main()
