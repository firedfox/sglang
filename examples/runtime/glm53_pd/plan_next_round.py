#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print a controlled MTP comparison manifest. Never contact or change a server."""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import uuid
from pathlib import Path


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def run_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", value):
        raise argparse.ArgumentTypeError(
            "use 1-96 letters, digits, dots, underscores or hyphens"
        )
    return value


def file_digest(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_plan(args):
    jobs = []
    for round_number in range(1, args.rounds + 1):
        profiles = ("5/1/6", "3/1/4")
        if round_number % 2 == 0:
            profiles = tuple(reversed(profiles))
        for profile in profiles:
            profile_key = profile.replace("/", "-")
            namespace = f"{args.run_id}-mtp{profile_key}-r{round_number}"
            for phase in ("fresh_namespace", "replay"):
                tag = f"{namespace}-{phase}"
                output = str(Path(args.output_dir) / f"{tag}.jsonl")
                command = [
                    sys.executable,
                    "-m",
                    "sglang.benchmark.serving",
                    "--backend",
                    "sglang",
                    "--base-url",
                    args.base_url,
                    "--pd-separated",
                    "--model",
                    "glm-5.3",
                    "--tokenizer",
                    args.tokenizer,
                    "--dataset-name",
                    "random",
                    "--dataset-path",
                    args.dataset,
                    "--tokenize-prompt",
                    "--random-range-ratio",
                    "1",
                    "--random-input-len",
                    "131072",
                    "--random-output-len",
                    "1024",
                    "--num-prompts",
                    "96",
                    "--max-concurrency",
                    "32",
                    "--request-rate",
                    "inf",
                    "--warmup-requests",
                    "0",
                    "--temperature",
                    "0",
                    "--seed",
                    str(args.seed),
                    "--extra-request-body",
                    json.dumps({"extra_key": namespace}),
                    "--cache-report",
                    "--output-details",
                    "--tag",
                    tag,
                    "--output-file",
                    output,
                ]
                jobs.append(
                    {
                        "job_id": tag,
                        "round": round_number,
                        "phase": phase,
                        "required_prefill_and_decode_eagle_profile": profile,
                        "cache_namespace": namespace,
                        "output_file": output,
                        "output_file_must_not_exist": True,
                        "argv": command,
                        "shell_command": shlex.join(command),
                    }
                )
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "dataset": args.dataset,
        "dataset_sha256": file_digest(args.dataset),
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "expected_completed_per_job": 96,
        "expected_prompt_tokens_per_job": 96 * 131072,
        "expected_completion_tokens_per_job": 96 * 1024,
        "jobs": jobs,
        "requirements": [
            "Use the corrected benchmark client from this branch; the plan does not change worker configuration.",
            "Before each profile pair, set BOTH P and D to the required EAGLE profile and verify readiness.",
            "Keep the current deployment's queue 64, calibrated cost model and 1500 ms budget unchanged in this MTP comparison.",
            "Save actual P/D/router arguments and revisions separately; this plan cannot verify unseen deployment manifests.",
            "Warm up model startup and graph capture separately; never profile formal measurements.",
            "Create the output directory. Do not append to an existing result file or reuse a namespace for a new round.",
            "Execute jobs in order. Reuse a namespace only for its immediately following replay; do not clear shared L3.",
            "Fresh namespace does not prevent within-run prefix reuse. Classify replay by measured cache layers.",
            "Compare like cache phases and actual cache layer mixes, with 96/96 success and verified token totals.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="results/glm53-next")
    parser.add_argument("--run-id", type=run_id, default=f"pd1-{uuid.uuid4().hex}")
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args(argv)
    if not args.base_url.startswith(("http://", "https://")):
        parser.error("--base-url must start with http:// or https://")
    plan = build_plan(args)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=False,
        )
        plan["client_checkout_revision"] = (
            revision.stdout.strip() if revision.returncode == 0 else None
        )
    except OSError:
        plan["client_checkout_revision"] = None
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
