# GLM-5.3 PD GPU acceptance and calibration

These commands are a procedure for a dedicated test deployment. They have not
been run against GLM-5.3 GPUs as part of this change. Use the same model revision,
tokenizer, sampling settings, Router budget, P/D replica counts, and hardware for
each comparison. Record the resolved per-rank capacities before sending load.

For the later operator-reported pd1 configuration and its 96 x 128K:1K C32
workload, use [NEXT_ROUND.md](NEXT_ROUND.md). It retains the calibrated policies
while checking the observed EAGLE improvement, and describes the corrected
benchmark schema and new policy diagnostics. The initial isolation matrix below
remains useful when validating an individual policy on a separate deployment.

## Compare one policy at a time

| Run | Prefill | Decode | Compare with |
| --- | --- | --- | --- |
| A | Launcher default `timeout` | Launcher default | Baseline, including the Gateway fixes in this branch |
| B | Same as A | `--enable-dp-admission` | A |
| C | Calibrated `--hicache-storage-prefetch-cost-model` | Same as B | B |
| D | Calibrated `--prefill-time-budget-ms` and `--prefill-time-model` | Same as B | B |

Test the combination of C and D only after each passes separately. Keep the
existing MTP configuration fixed during these comparisons. Adaptive MTP,
all-CP-rank transfer, communication SM changes, and kernel changes need separate
experiments; no gain from them is assumed by this baseline.

## Reproducible requests and a capacity scan

Use the current `sglang.benchmark.serving` entrypoint. Native `/generate` accepts
token IDs, so `--tokenize-prompt` and `--random-range-ratio 1` fix input lengths.
Each request has one prompt and `n=1`. Streaming and `ignore_eos` are the
benchmark defaults; temperature is fixed below. The `random` dataset repeats or
truncates natural-language tokens from the supplied ShareGPT-format JSON; it is
a synthetic long-context workload, not a substitute for production traffic.

```bash
export GLM53_ROUTER_URL=http://ROUTER_IP:8000
export GLM53_PREFILL_URL=http://PREFILL_IP:8000
export GLM53_DECODE_URL=http://DECODE_IP:8000
export GLM53_TOKENIZER=/models/GLM-5.3/snapshots/30333038ada1f1dacb294a93270305a890b50c14
export GLM53_DATASET=/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
export GLM53_RUN_ID=baseline-r1
mkdir -p results/glm53

bench_case() {
  local name="$1" input="$2" output="$3" concurrency="$4" count="$5"
  shift 5
  python3 -m sglang.benchmark.serving \
    --backend sglang \
    --base-url "$GLM53_ROUTER_URL" \
    --pd-separated \
    --model glm-5.3 \
    --tokenizer "$GLM53_TOKENIZER" \
    --dataset-name random \
    --dataset-path "$GLM53_DATASET" \
    --tokenize-prompt \
    --random-range-ratio 1 \
    --random-input-len "$input" \
    --random-output-len "$output" \
    --num-prompts "$count" \
    --max-concurrency "$concurrency" \
    --request-rate inf \
    --warmup-requests 0 \
    --temperature 0 \
    --seed 20261026 \
    --extra-request-body "{\"extra_key\":\"${GLM53_RUN_ID}-${name}-c${concurrency}\"}" \
    --cache-report \
    --output-details \
    --output-file "results/glm53/${GLM53_RUN_ID}-${name}-c${concurrency}.jsonl" \
    "$@"
}

for c in 1 8 32 64 128 256; do
  bench_case short 2048 256 "$c" 512
done

for c in 1 2 4 8 16; do
  bench_case context128k 131072 256 "$c" 32
done

# Start with c=1. Continue only if a request fits the measured rank capacity.
for c in 1 2 4 8; do
  bench_case near1m 1040000 256 "$c" 8
done
```

The near-1M case leaves context room for output and is initially a capacity
smoke test. Eight requests cannot establish a stable p99. An individual 400
from rank admission means that this request cannot fit that Decode engine's
capacity, not that the GPU is slow. Increase samples only after startup,
correctness, and capacity checks pass.

Warm up startup and CUDA graph capture separately. Then use a fresh run ID and
`extra_key` for formal measurements. This avoids old cache namespaces; requests
within a run may still share prefixes, so retain cache reports. The benchmark
appends JSONL to `--output-file`; use a new run ID rather than mixing experiments.

Delay statistics include successful requests. Always report the total offered
requests, `completed`, errors, throughput, and latency together. Lower p99 after
rejecting more work is not, by itself, an improvement. If you have service SLOs,
add `--goodput ttft:YOUR_MS tpot:YOUR_MS e2el:YOUR_MS`; these bounds are in
milliseconds and must come from the service's requirements.

## Separate GPU, host, and storage cache hits

```bash
export GLM53_CACHE_KEY="${GLM53_RUN_ID}-prefix128k"

bench_prefix() {
  local stage="$1"
  python3 -m sglang.benchmark.serving \
    --backend sglang --base-url "$GLM53_ROUTER_URL" --pd-separated \
    --model glm-5.3 --tokenizer "$GLM53_TOKENIZER" \
    --dataset-name generated-shared-prefix --tokenize-prompt \
    --gsp-num-groups 8 --gsp-prompts-per-group 4 \
    --gsp-system-prompt-len 131008 --gsp-question-len 64 \
    --gsp-output-len 256 --gsp-range-ratio 1 --gsp-num-turns 1 \
    --max-concurrency 4 --request-rate inf --warmup-requests 0 \
    --seed 20261026 \
    --extra-request-body "{\"extra_key\":\"${GLM53_CACHE_KEY}\"}" \
    --cache-report --output-details \
    --output-file "results/glm53/${GLM53_RUN_ID}-${stage}.jsonl"
}

bench_prefix cold-prefix
bench_prefix immediate-replay
```

This creates eight prefix groups with four requests each. Each input has
131,072 tokens. The first request of each group can be cold; later requests can
reuse the group's prefix.

| Scenario | Preparation | Evidence required |
| --- | --- | --- |
| Cold | Fresh `extra_key` | No prior cache hit for the first group request |
| GPU/L1 | Immediate replay of the same namespace | `device_cached_tokens > 0` |
| Host/L2 | Warm, evict GPU entries with unrelated namespaces, then replay | `host_cached_tokens > 0`; require zero storage hits for a pure L2 sample |
| Storage/L3 | Warm, wait for persistence, clear idle local caches, then replay | `storage_cached_tokens > 0` |

Immediate replay is not an L2 experiment. There is no benchmark flag to clear
only GPU while retaining host cache; choose eviction traffic from the actual
GPU/host pool capacities and verify the resulting layer hits.

For the L3 experiment, after all test traffic and write-through backups finish,
flush each idle Prefill worker's local GPU and host cache:

```bash
curl -fsS -X POST "${GLM53_PREFILL_URL}/flush_cache?timeout=60"
bench_prefix l3-replay
```

This endpoint does not clear shared L3. Do not call the storage-backend clear
endpoint as part of this procedure. Give each comparison its own namespace and
repeat warm, persistence, local flush, and replay. For cost-aware stopping,
compare useful cached tokens, recomputation and TTFT alongside
`sglang:storage_prefetch_unfulfilled_tokens_total{reason="cost_aware"}`. That
counter describes unfulfilled token attempts, not final cache misses or savings.

## Admission failures and cancellation

To exercise the Gateway's strict 256-request limit in the test deployment:

```bash
bench_case gateway-overload 2048 4096 512 1024
```

To exercise Decode admission, stay below the Gateway limit and fill rank KV
budgets with long requests that each fit individually. Record 400, 429, and 503
separately. With the corrected client, `http_status_counts` and `error_counts`
are always written; `--output-details` also records each request's status,
success, finish reason, and actual completion tokens. An HTTP 200 stream can
still fail, so do not substitute status counts for `completed`/`failed_requests`.
Collect Gateway metrics and use HTTP probes for independent status verification:

```bash
curl -sS \
  -D results/glm53/status.headers \
  -o results/glm53/status.body \
  -w '%{http_code}\n' \
  -H 'Content-Type: application/json' \
  -d '{"text":"Status probe","sampling_params":{"temperature":0,"max_new_tokens":16},"stream":false}' \
  "${GLM53_ROUTER_URL}/generate"

curl -fsS "${GLM53_DECODE_URL}/v1/loads?include=core,memory,disagg,queues"
```

Useful Gateway metrics include `smg_http_responses_total{status_code="429"}`,
`smg_http_rate_limit_total{result="rejected"}`, and
`smg_router_upstream_responses_total{status_code="429"}`. Interpret Decode
capacities and `dp_admission_reserved_tokens`/`dp_admission_reserved_requests`
per `dp_rank`, not as totals for the TP8 engine. A benchmark against deliberately
unavailable workers needs `--ready-check-timeout-sec 0 --warmup-requests 0` to
reach the measurement phase.

The benchmark has no cancellation flag. Test client disconnect separately
during long prefill, KV transfer, and decode streaming, using a unique request
ID. Confirm local queue removal or eventual scheduler KV release, then verify
that subsequent requests can be admitted. A returned Gateway permit is not an
acknowledgement of GPU cleanup. The existing
`test/registered/disaggregation/test_disaggregation_chunked_prefill_abort.py`
provides a small-model fixture; it does not test a running GLM-5.3 deployment.

## Collect a prefill calibration sample

Default request OTLP traces and benchmark TTFT do not contain a complete
per-batch GPU cost dataset. Use a small, explicitly profiled sample, then disable
profiling for performance comparisons. The existing detailed annotation request
is available on the Prefill worker:

```bash
(
  set -e
  trap 'curl -fsS -X POST "${GLM53_PREFILL_URL}/stop_profile"' EXIT
  curl -fsS -X POST "${GLM53_PREFILL_URL}/start_profile" \
    -H 'Content-Type: application/json' \
    -d '{"activities":["CPU","GPU"],"output_dir":"/tmp/glm53-prefill-calibration","profile_prefix":"glm53-prefill","detailed_annotations":true}'

  bench_case calibration 131072 256 1 4
)
```

Trace files are written in the worker container. Ordinary benchmark `--profile`
does not automatically request `detailed_annotations`. For ordinary EXTEND,
the detailed fields use global logical chunk lengths `q` and prefixes `p`:

```text
c_sq   = sum(q)
c_sk   = sum(p + q)
c_sqsq = sum(q*q)
c_sqsk = sum(q*(p + q))
sum(p*q) = c_sqsk - c_sqsq
```

The subshell stops the profiler when the benchmark completes or fails. Do not
rely on a fixed `num_steps` exceeding the number of Prefill forwards in this
sample; the profiler could otherwise stay active and delay trace export.

Do not divide these fields by CP8. Associate the shapes with measured GPU
execution durations; the CPU annotation range's duration is not GPU completion
time. Distinguish target EXTEND from MTP DRAFT/VERIFY, and include the relevant
work consistently in calibration. Cover different prefixes, chunk lengths,
batch sizes, and cache-hit conditions, then validate nonnegative coefficients
on held-out cases. These interfaces provide a way to collect evidence; they do
not supply measured GLM-5.3/B300 coefficients.

Retain TTFT/TPOT/ITL/E2E distributions, admitted throughput, rejections, cache
layer hits, retractions, per-rank KV commitments, transfer durations, host
memory, and MTP acceptance length for each run. Evaluate combined policies only
after their independent correctness and capacity checks pass.
