#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print or execute the GLM-5.3 HTTP PD baseline; see the adjacent README."""

import argparse
import json
import math
import os
import sys

ALL_CP_ENV = "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER"
DEFAULT_IB_DEVICES = ",".join(f"mlx5_bond_{i}" for i in range(8))
DEFAULT_OTLP_ENDPOINT = "otel-collector-service.ops.svc.cluster.local:4317"


def nonempty(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def port_number(value):
    number = positive_int(value)
    if number > 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def ib_devices(value):
    devices = [device.strip() for device in value.split(",")]
    if not all(devices):
        raise argparse.ArgumentTypeError("expected comma-separated IB device names")
    return ",".join(devices)


def json_object(value):
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"must be a JSON object: {error}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    # The runtime validates the policy's coefficients before model loading.
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="GLM-5.3 HTTP PD baseline: JSON dry-run unless --run is given.",
        allow_abbrev=False,
    )
    roles = parser.add_subparsers(dest="role", required=True)
    for role in ("router", "prefill", "decode"):
        role_parser = roles.add_parser(
            role,
            allow_abbrev=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        role_parser.add_argument(
            "--run", action="store_true", help="Execute after printing the launch plan"
        )
        role_parser.add_argument(
            "--host", type=nonempty, default=os.environ.get("POD_IP") or "0.0.0.0"
        )
        role_parser.add_argument("--port", type=port_number, default=8000)
        role_parser.add_argument(
            "--otlp-traces-endpoint", type=nonempty, default=DEFAULT_OTLP_ENDPOINT
        )
        if role == "router":
            role_parser.add_argument("--namespace", type=nonempty, default="sglang2")
            role_parser.add_argument(
                "--group-name", type=nonempty, default="sglang-pd2"
            )
            role_parser.add_argument("--worker-port", type=port_number, default=8000)
            role_parser.add_argument(
                "--prometheus-port", type=port_number, default=9090
            )
            role_parser.add_argument(
                "--max-concurrent-requests",
                type=positive_int,
                default=256,
                help="Per-router HTTP request budget; divide it across router replicas",
            )
        else:
            role_parser.add_argument(
                "--model-path",
                type=nonempty,
                default=os.environ.get("GLM53_MODEL_PATH"),
                help="GLM-5.3 checkpoint path or ID; falls back to GLM53_MODEL_PATH",
            )
            role_parser.add_argument(
                "--ib-devices",
                type=ib_devices,
                default=os.environ.get("GLM53_IB_DEVICES") or DEFAULT_IB_DEVICES,
            )
            role_parser.add_argument(
                "--chat-template",
                type=nonempty,
                help="Optional worker template; use the same file on P and D",
            )
            role_parser.add_argument(
                "--all-cp-ranks-transfer",
                action="store_true",
                help="Experimental: opt in on BOTH P and D; default explicitly disables it",
            )
            if role == "decode":
                role_parser.add_argument(
                    "--enable-dp-admission",
                    action="store_true",
                    help="Opt in to per-rank slot and full-generation KV reservations",
                )
            else:
                role_parser.add_argument(
                    "--hicache-storage-prefetch-cost-model",
                    type=json_object,
                    help="Opt in to cost_aware L3 prefetch with calibrated JSON rates",
                )
                role_parser.add_argument(
                    "--prefill-time-budget-ms",
                    type=positive_float,
                    help="Estimated GPU prefill time per batch; requires --prefill-time-model",
                )
                role_parser.add_argument(
                    "--prefill-time-model",
                    type=json_object,
                    help="Calibrated JSON coefficients; requires --prefill-time-budget-ms",
                )
    return parser


def router_command(args):
    # HTTP workers own tokenization, chat templates and GLM output parsing.
    return [
        sys.executable,
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--service-discovery",
        "--prefill-selector",
        "rbg.workloads.x-k8s.io/role-name=prefill",
        f"rbg.workloads.x-k8s.io/group-name={args.group_name}",
        "--decode-selector",
        "rbg.workloads.x-k8s.io/role-name=decode",
        f"rbg.workloads.x-k8s.io/group-name={args.group_name}",
        "--service-discovery-namespace",
        args.namespace,
        "--service-discovery-port",
        str(args.worker_port),
        "--prefill-policy",
        "cache_aware",
        "--decode-policy",
        "power_of_two",
        "--cache-threshold",
        "0.3",
        "--balance-abs-threshold",
        "16",
        "--balance-rel-threshold",
        "1.5",
        "--eviction-interval-secs",
        "120",
        "--max-tree-size",
        "67108864",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--prometheus-port",
        str(args.prometheus_port),
        "--prometheus-host",
        "0.0.0.0",
        "--worker-startup-check-interval",
        "2",
        "--health-check-endpoint",
        "/health",
        "--health-check-timeout-secs",
        "30",
        "--health-check-interval-secs",
        "50",
        "--health-failure-threshold",
        "3",
        "--health-success-threshold",
        "1",
        "--cb-failure-threshold",
        "5",
        "--cb-success-threshold",
        "2",
        "--cb-timeout-duration-secs",
        "15",
        "--retry-max-retries",
        "2",
        "--retry-initial-backoff-ms",
        "100",
        "--retry-max-backoff-ms",
        "1000",
        "--max-concurrent-requests",
        str(args.max_concurrent_requests),
        "--rate-limit-tokens-per-second",
        "0",
        "--queue-size",
        "0",
        "--history-backend",
        "memory",
        "--enable-trace",
        "--otlp-traces-endpoint",
        args.otlp_traces_endpoint,
        "--request-id-headers",
        "x-request-id",
        "x-trace-id",
        "x-correlation-id",
        "--json-log",
        "--pool-idle-timeout-secs",
        "50",
        "--request-timeout-secs",
        "7200",
        "--log-level",
        "info",
    ]


def worker_command(args):
    argv = [
        "sglang",
        "serve",
        "--model-path",
        args.model_path,
        "--served-model-name",
        "glm-5.3",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tp-size",
        "8",
        "--dtype",
        "auto",
        "--tokenizer-worker-num",
        "32",
        "--detokenizer-worker-num",
        "16",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--context-length",
        "1048576",
        "--page-size",
        "64",
        "--moe-a2a-backend",
        "deepep",
        "--moe-dense-tp-size",
        "1",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--schedule-policy",
        "fcfs",
        "--disaggregation-mode",
        args.role,
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-ib-device",
        args.ib_devices,
        "--enable-cache-report",
        "--enable-metrics",
        "--enable-mfu-metrics",
        "--enable-metrics-for-all-schedulers",
        "--enable-trace",
        "--trace-modules",
        "request,mooncake",
        "--otlp-traces-endpoint",
        args.otlp_traces_endpoint,
        "--reasoning-parser",
        "glm45",
        "--tool-call-parser",
        "glm47",
        "--trust-remote-code",
    ]
    if args.chat_template is not None:
        argv.extend(["--chat-template", args.chat_template])
    if args.role == "prefill":
        argv.extend(
            [
                "--watchdog-timeout",
                "3600",
                "--dist-timeout",
                "3600",
                "--enable-prefill-cp",
                "--attn-cp-size",
                "8",
                "--cp-strategy",
                "interleave",
                "--mem-fraction-static",
                "0.8",
                "--max-prefill-tokens",
                "131072",
                "--chunked-prefill-size",
                "131072",
                "--prefill-max-requests",
                "256",
                "--enable-hierarchical-cache",
                "--hicache-host-memory-fraction",
                "0.8",
                "--hicache-write-policy",
                "write_through",
                "--hicache-io-backend",
                "direct",
                "--hicache-mem-layout",
                "page_first_direct",
                "--hicache-storage-backend",
                "mooncake",
                "--hicache-storage-prefetch-policy",
                (
                    "cost_aware"
                    if args.hicache_storage_prefetch_cost_model is not None
                    else "timeout"
                ),
                "--deepep-mode",
                "normal",
                "--deepep-config",
                '{"normal_dispatch":{"num_sms":64},"normal_combine":{"num_sms":64}}',
            ]
        )
        if args.hicache_storage_prefetch_cost_model is not None:
            argv.extend(
                [
                    "--hicache-storage-prefetch-cost-model",
                    args.hicache_storage_prefetch_cost_model,
                ]
            )
        if args.prefill_time_budget_ms is not None:
            argv.extend(
                [
                    "--prefill-time-budget-ms",
                    str(args.prefill_time_budget_ms),
                    "--prefill-time-model",
                    args.prefill_time_model,
                ]
            )
    else:
        argv.extend(
            [
                "--dp-size",
                "8",
                "--ep-size",
                "8",
                "--enable-dp-attention",
                "--enable-dp-lm-head",
                "--load-balance-method",
                "total_tokens",
                "--max-running-requests",
                "256",
                "--mem-fraction-static",
                "0.85",
                "--deepep-mode",
                "low_latency",
                "--disable-radix-cache",
                "--stream-response-default-include-usage",
            ]
        )
        if args.enable_dp_admission:
            argv.extend(
                [
                    "--enable-dp-admission",
                    "--dp-admission-max-queued-requests",
                    "0",
                    "--dp-admission-queue-timeout",
                    "1.0",
                    "--dp-admission-snapshot-timeout",
                    "30.0",
                ]
            )
    return argv


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.role == "router":
        command = router_command(args)
        env_overrides = {}
    else:
        if args.model_path is None:
            parser.error("prefill/decode require --model-path or GLM53_MODEL_PATH")
        if args.role == "prefill" and (
            (args.prefill_time_budget_ms is None) != (args.prefill_time_model is None)
        ):
            parser.error(
                "--prefill-time-budget-ms and --prefill-time-model are required together"
            )
        command = worker_command(args)
        # An inherited value of 1 must not silently opt a worker into the experiment.
        env_overrides = {ALL_CP_ENV: "1" if args.all_cp_ranks_transfer else "0"}

    print(
        json.dumps(
            {"role": args.role, "argv": command, "env": env_overrides}, indent=2
        ),
        flush=True,
    )
    if args.run:
        environment = os.environ.copy()
        environment.update(env_overrides)
        os.execvpe(command[0], command, environment)
    return 0


if __name__ == "__main__":
    sys.exit(main())
