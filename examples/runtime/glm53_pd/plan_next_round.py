#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print a controlled MTP comparison manifest. Never contact or change a server."""

import argparse
import hashlib
import json
import math
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


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def seed_number(value):
    number = nonnegative_int(value)
    if number >= 2**32:
        raise argparse.ArgumentTypeError("must be smaller than 2**32")
    return number


def ratio(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be a finite ratio between 0 and 1")
    return number


def run_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", value):
        raise argparse.ArgumentTypeError(
            "use 1-96 letters, digits, dots, underscores or hyphens"
        )
    return value


def file_digest(path):
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def workload_configuration(args):
    """Describe the selected experiment without guessing missing report inputs."""
    gsp_fields = (
        "gsp_num_groups",
        "gsp_prompts_per_group",
        "gsp_system_prompt_len",
        "gsp_question_len",
        "gsp_range_ratio",
    )
    gsp_switches = ("gsp_fast_prepare", "gsp_ordered", "gsp_send_routing_key")
    if args.workload == "native-random":
        if not args.dataset:
            raise ValueError("native-random requires --dataset")
        if any(getattr(args, name) is not None for name in gsp_fields) or any(
            getattr(args, name) for name in gsp_switches
        ):
            raise ValueError("--gsp-* options require --workload chat-gsp")
        return (
            [
                "--backend",
                "sglang",
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
            ],
            {
                "backend": "sglang",
                "dataset_name": "random",
                "comparison_scope": "independent_baseline",
                "expected_prompt_tokens_per_job": 96 * 131072,
                "expected_prompt_tokens_source": "fixed_native_input_ids",
                "generation_parameters": {
                    "random_input_len": 131072,
                    "random_output_len": 1024,
                    "random_range_ratio": 1,
                    "tokenize_prompt": True,
                },
            },
        )
    if args.workload != "chat-gsp":
        raise ValueError("Select --workload chat-gsp or native-random explicitly")
    if args.dataset is not None:
        raise ValueError("--dataset applies only to --workload native-random")
    missing = [
        "--" + name.replace("_", "-")
        for name in gsp_fields
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "chat-gsp requires explicit generation settings: " + ", ".join(missing)
        )
    if args.gsp_num_groups * args.gsp_prompts_per_group != 96:
        raise ValueError("--gsp-num-groups times --gsp-prompts-per-group must equal 96")
    if args.gsp_system_prompt_len + args.gsp_question_len <= 0:
        raise ValueError("The combined system/question target length must be positive")
    if args.gsp_range_ratio != 1:
        # This ratio randomizes output lengths too, not only prompt lengths.
        raise ValueError(
            "This fixed-output comparison requires --gsp-range-ratio 1; other ratios randomize the 1024-token output budgets"
        )
    command = [
        "--backend",
        "sglang-oai-chat",
        "--dataset-name",
        "generated-shared-prefix",
        "--gsp-num-groups",
        str(args.gsp_num_groups),
        "--gsp-prompts-per-group",
        str(args.gsp_prompts_per_group),
        "--gsp-system-prompt-len",
        str(args.gsp_system_prompt_len),
        "--gsp-question-len",
        str(args.gsp_question_len),
        "--gsp-range-ratio",
        str(args.gsp_range_ratio),
        "--gsp-output-len",
        "1024",
        "--gsp-num-turns",
        "1",
        "--gsp-group-distribution",
        "uniform",
    ]
    for name in gsp_switches:
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    return command, {
        "backend": "sglang-oai-chat",
        "dataset_name": "generated-shared-prefix",
        "comparison_scope": "same_reported_backend_and_dataset_family",
        # Decoded random tokens are retokenized and then chat-templated. The
        # nominal generator lengths do not determine the server's prompt usage.
        "expected_prompt_tokens_per_job": None,
        "expected_prompt_tokens_source": "unknown_until_server_usage_is_recorded",
        "generation_parameters": {
            **{name: getattr(args, name) for name in (*gsp_fields, *gsp_switches)},
            "gsp_output_len": 1024,
            "gsp_num_turns": 1,
            "gsp_group_distribution": "uniform",
            "tokenize_prompt": False,
        },
    }


def build_plan(args):
    workload_argv, workload_metadata = workload_configuration(args)
    jobs = []
    for round_number in range(1, args.rounds + 1):
        profiles = ("5/1/6", "3/1/4")
        if round_number % 2 == 0:
            profiles = tuple(reversed(profiles))
        for profile in profiles:
            profile_key = profile.replace("/", "-")
            namespace = (
                f"{args.run_id}-{args.workload}-mtp{profile_key}-r{round_number}"
            )
            for phase in ("fresh_namespace", "replay"):
                tag = f"{namespace}-{phase}"
                output = str(Path(args.output_dir) / f"{tag}.jsonl")
                command = [
                    sys.executable,
                    "-m",
                    "sglang.benchmark.serving",
                    *workload_argv,
                    "--base-url",
                    args.base_url,
                    "--pd-separated",
                    "--model",
                    "glm-5.3",
                    "--tokenizer",
                    args.tokenizer,
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
                        "required_decode_eagle_profile": profile,
                        "prefill_configuration": "preserve_validated_deployment",
                        "cache_namespace": namespace,
                        "output_file": output,
                        "output_file_must_not_exist": True,
                        "argv": command,
                        "shell_command": shlex.join(command),
                    }
                )
    return {
        "schema_version": 2,
        "run_id": args.run_id,
        "workload": args.workload,
        **workload_metadata,
        "original_inputs_reconstructed": False,
        "dataset": args.dataset,
        "dataset_sha256": file_digest(args.dataset),
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "expected_completed_per_job": 96,
        "expected_completion_tokens_per_job": 96 * 1024,
        "jobs": jobs,
        "requirements": [
            "Use the corrected benchmark client from this branch; the plan does not change worker configuration.",
            "Change Decode to the required EAGLE profile; preserve the validated Prefill configuration. The supplied pd1 Prefill command does not enable EAGLE.",
            "Before comparing profiles, verify P/D resolved arguments, target/draft KV layout compatibility, startup capacities and readiness; the plan does not infer or repair them.",
            "Keep the current deployment's queue 64, calibrated cost model and 1500 ms budget unchanged in this MTP comparison.",
            "Save actual P/D/router arguments and revisions separately; this plan cannot verify unseen deployment manifests.",
            "The report identifies chat/GSP but omits its generation settings and seed. This is an explicitly configured new comparison, not reconstructed original inputs.",
            "Keep tokenizer revision, generation settings and actual generated payload/cache identity fixed. GSP disk cache keys omit some preparation/order settings; routing keys contain per-run values.",
            "For chat/GSP, record server-reported prompt usage; do not require 131072 tokens per request or 12582912 total from nominal generator lengths.",
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
    parser.add_argument(
        "--workload", choices=("chat-gsp", "native-random"), required=True
    )
    parser.add_argument(
        "--dataset", help="Natural-language dataset file for native-random only"
    )
    parser.add_argument("--gsp-num-groups", type=positive_int)
    parser.add_argument("--gsp-prompts-per-group", type=positive_int)
    parser.add_argument("--gsp-system-prompt-len", type=nonnegative_int)
    parser.add_argument("--gsp-question-len", type=nonnegative_int)
    parser.add_argument("--gsp-range-ratio", type=ratio)
    parser.add_argument("--gsp-fast-prepare", action="store_true")
    parser.add_argument("--gsp-ordered", action="store_true")
    parser.add_argument("--gsp-send-routing-key", action="store_true")
    parser.add_argument("--output-dir", default="results/glm53-next")
    parser.add_argument("--run-id", type=run_id, default=f"pd1-{uuid.uuid4().hex}")
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--seed", type=seed_number, required=True)
    args = parser.parse_args(argv)
    if not args.base_url.startswith(("http://", "https://")):
        parser.error("--base-url must start with http:// or https://")
    try:
        plan = build_plan(args)
    except ValueError as exc:
        parser.error(str(exc))
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
