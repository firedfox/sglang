# Continue the pd1 GLM-5.3 optimization

This procedure follows the operator's 2026-10-27 report. The report describes
one TP8/CP8 Prefill engine, one TP8/attention-DP8/EP8 Decode engine, Mooncake L3,
and a PD router exposed by `sglang-pd1` / NodePort 31180. The separate
`sglang-pd1-single2` deployment remains scaled to zero. These are reported facts,
not a deployment discovered or changed by this development session.

The runtime is reported as v0.5.18 with 29 backports, including
`0041-dp-admission-kvfix.patch`. This branch has a different source base. The
deployed YAML, backport and raw benchmark JSON are needed before applying these
changes to that runtime. Nothing in this directory patches a live ConfigMap or
starts a benchmark automatically.

## What the current measurements establish

The operator reported the following for 96 requests, 131,072 input tokens and
1,024 output tokens each, at client concurrency 32:

| Reported stage | Duration | TTFT p50 | TPOT p50 | Output tokens/s |
| --- | --- | --- | --- | --- |
| A: v0.5.18 without patches | 106.2 s | 12.3 s | 11.5 ms | 928 |
| B: patches, hot cache | 58.9 s | 2.6 s | 11.7 ms | 1670 |
| C: policy flags | 61.0 s | 2.6 s | 11.7 ms | 1611 |
| D: EAGLE five steps / six draft tokens | 43.7 s | 1.66 s | 11.2 ms | 2248 |

D is the best reported result: its output throughput is 39.5% above C and 34.6%
above B. Keep its `5/1/6` steps/top-k/draft configuration as the candidate
baseline. The 3.5% difference between B and C is not sufficient to attribute a
regression to any one policy: several switches changed together, and repetitions
and raw results are not available here. B versus A also changes the stated cache
condition. The TPOT values are medians; multiplying a median by throughput does
not measure mean active decode concurrency or GPU utilization.

For this fixed-length test, 96 successful full outputs should total **98,304
output tokens**, with **12,582,912 input tokens**. Verify these counts in raw
results before comparing throughput. Discovered client defects below do not
prove that the earlier report used that client or contained incorrect results.

## Changes prepared in this branch, in priority order

### 1. Make the next measurement auditable

Use this branch's `python/sglang/benchmark/serving.py` as the benchmark client.
The main workload timer now stops immediately after the request tasks finish.
Profiler flushing and `/server_info` collection occur outside that interval;
finite-rate traffic no longer waits for an extra arrival after the last request.

Native `/generate` requires actual `meta_info.completion_tokens`. Empty streams,
bare `[DONE]`, native error objects, and abort finish reasons do not become
successful requested-length outputs. Completion counts are read even when text
is empty. A stream must report a normal finish reason or end with `[DONE]`;
receiving partial counts followed only by EOF is a failure. This accepts the
normal v0.5.18 protocol without requiring both terminal markers.
All-failed main measurement runs still write a result, with undefined latency
statistics set to `null`. The existing all-failed warmup check remains; use zero
warmup requests for deliberate failure tests after warming the model separately.

The result has `measurement_schema_version: 2`:

| Field | Interpretation |
| --- | --- |
| `duration`, `workload_duration` | The same measured request interval; metadata collection is excluded |
| `workload_start_time`, `workload_end_time` | Client monotonic timestamps, not cross-host wall clocks |
| `metadata_collection_duration` | Separate time spent obtaining `/server_info` |
| `offered_requests`, `completed`, `failed_requests` | Report together, including deliberate overload cases |
| `http_status_counts`, `error_counts` | Always retained, including without `--output-details` |
| `inflight_concurrency` | Mean measured in-flight requests, including failures |
| `max_concurrent_requests` | Exact peak from request start/end intervals, not whole-second buckets |
| `request_details[]` | With `--output-details`: success, status, reason, error, interval, actual tokens, cache details and speculative counters |
| `request_speculative_metrics` | This run's successful responses, with counter coverage and weighted ratios |
| `accept_length`, `accept_length_source` | Legacy first-worker/first-rank lifetime observation; not a whole-run MTP measurement |

For successful responses that all supply final counters, the new MTP acceptance
length is `sum(completion_tokens) / sum(spec_verify_ct)`. Draft acceptance rate
is `sum(spec_num_correct_drafts) / sum(spec_num_proposed_drafts)`. Missing
coverage produces `null`, not a fabricated ratio or an average of request
averages. A deployed older runtime may not supply every counter.

Existing arrays remain available. JSON consumers must allow `null` latency
statistics and the string `"inf"` for unlimited request rate. ITL still uses
chunk interpolation, and peak tokens/s remains a client-side estimate. Neither
is a measurement of individual GPU token completion times.

### 2. Release Decode admission capacity promptly

With `--enable-dp-admission`, publish the load snapshot after result processing
and allocator group release. Actual ledger releases, or a reduction in the
allocator token floor, trigger one early publication per batch. Retried writes
and independent `/v1/loads` reads do not lose the notification. Ordinary decode
steps retain the existing 15-iteration publication interval. Existing controller
polling limits still apply; this is not an immediate cross-process wakeup.

Terminal output alone does not release a reservation. Mooncake deferred cleanup,
KV/Mamba ownership and backup ownership still protect capacity. Stale references
to fully released requests in the running list no longer hide free request slots.
Default-off behavior, ACK ordering and the snapshot schema are unchanged.

The regression tests reproduce the feedback delay and show the queued request
becoming dispatchable after release. They do not establish that this delay is
the dominant issue at C32, or measure a GPU throughput gain.

Keep `--dp-admission-max-queued-requests 64` fixed during the MTP comparison.
This is a controller-wide FIFO capacity. Save the deployed queue deadline rather
than assuming the launcher's one-second default is deployed. Also inspect the
actual router arguments: an external `--dp-aware` rank selection can pin a
request before `total_tokens` gets to choose a rank. Do not change that behavior
in the middle of the MTP experiment.

### 3. Observe what Prefill and L3 policies actually decide

The existing Prefill batch log gains the following fields only when the time
budget is enabled:

```text
time-forward-id, time-budget-ms, time-estimated-ms,
time-limited-reqs, time-budget-overruns,
time-sum-x, time-sum-hx, time-sum-x2
```

Here `x` is a committed logical extend length and `h` is its cached prefix.
The shape is snapshotted before requests advance. `time-limited-reqs` counts
committed requests actually shortened below the shape allowed by other hard
limits and alignment; rejected candidates do not count. Zero does not rule out
the budget preventing another complete request from joining the batch.
`time-budget-overruns` counts committed minimum-progress exceptions.
`time-estimated-ms` is a model estimate, not GPU timing. Neither these shapes nor
the existing inter-log input-throughput statistic replace the profiled GPU
calibration procedure in [GPU_ACCEPTANCE.md](GPU_ACCEPTANCE.md).

L3 lifecycle counts are now exported as:

```text
sglang:storage_prefetch_outcomes_total{outcome="stopped_cost_aware", ...}
sglang:storage_prefetch_outcomes_total{outcome="stopped_deadline", ...}
```

There are 14 fixed outcome labels. The collector consumes cumulative deltas,
including when a backend reuses its stats object. Repeated polling does not add
events. Outcomes overlap; do not sum them into a request count. Rank-agreed
events can be replicated across ranks, so choose a representative rank rather
than multiplying the request count by CP/TP size. The existing unfulfilled-token
counter continues to measure token attempts, not saved compute or final misses.

For a small diagnostic sample, enable the existing worker DEBUG log level. Each
cost-aware IO attempt emits at its existing terminal boundary:

```text
HiCache prefetch cost decision {JSON}
```

The JSON includes the configured model, last actual evaluation, effective read
rate and its source, remaining/read/recompute spans, margin, fixed cost, local
vote, rank-agreed stop reason and fallback reason. INFO does not construct these
diagnostic snapshots. Disable DEBUG for formal performance measurements.

The cancel test is:

```text
remaining_tokens / effective_read_rate
  > cancel_margin * (recompute_fixed_seconds + recompute_tokens / recompute_rate)
```

For example, if effective read rate is exactly 34k tokens/s, recompute is 47k,
the margin is 1.2, fixed cost is zero, the observation is valid, and partial KV
is reusable (`recompute_tokens == remaining_tokens`), the policy cancels.
Already reading most of that same reusable span does not reverse this ratio.
A larger margin, positive fixed cost, a different observed read rate, an
insufficient sampling window, or an independent all-or-nothing sidecar can
change the decision. Thus the reported "34k versus 47k, kept prefetch" warrants
the actual decision record and deployed cost JSON; it does not by itself prove
that the policy malfunctioned. Do not change the calibrated model on that ratio
alone.

## Controlled next experiment

First repeat the candidate `5/1/6` and earlier `3/1/4` profiles while keeping the
1500 ms budget, both calibrated models, queue 64, router, weights and hardware
fixed. The offline planner creates three rounds by default, alternates profile
order, and pairs each fresh namespace with an immediate replay: **12 jobs**.
It only prints a manifest; it neither reconfigures workers nor runs requests.

```bash
mkdir -p results/glm53-next
python3 examples/runtime/glm53_pd/plan_next_round.py \
  --base-url http://YOUR_NODE_IP:31180 \
  --tokenizer /models/GLM-5.3/snapshots/30333038ada1f1dacb294a93270305a890b50c14 \
  --dataset /datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --output-dir results/glm53-next \
  > results/glm53-next/plan.json
```

The manifest includes literal argument arrays and shell-quoted commands,
required P/D profile, unique output paths, seed, expected token totals and a
dataset SHA-256 when the file is present. `client_checkout_revision` identifies
this checkout, not whatever installed module another interpreter may import.
Run with this branch's client environment and record its resolved package path
and revision. The supplied dataset/seed is a new reproducible comparison; it
does not reconstruct the earlier workload without that workload's raw inputs.
Verify that the dataset exists and is valid on the benchmark host, and record
the hash of the file actually loaded. The random dataset loader can download a
default dataset when the supplied path is missing or invalid; a missing digest
in this offline plan is not evidence that the execution host used the same data.

Before each profile change, drain the dedicated test workload, configure both
P and D consistently, verify readiness, and warm up graph capture separately.
Keep other traffic out of the measurement window. Run manifest jobs in order.
Do not reuse an existing result file: the benchmark appends JSONL. Use each
namespace only for its intended fresh/replay pair. A fresh namespace can still
have within-run reuse, and immediate replay can hit L1 or L2; neither is proof
of an L3-only test. Compare equivalent observed cache-layer mixes. Do not clear
shared L3 for this comparison.

Accept a comparison only with 96/96 successful requests, expected token totals,
matching cache conditions, stable model/runtime/configuration, and raw detail
files. Report median and range across repetitions, TTFT/TPOT tails, failures,
cache layers, MTP counter coverage and admission/transfer observations together.
Do not combine schema-1 and schema-2 throughput denominators in one A/B result.

After this confirms the baseline, make one local change per experiment:

1. Check C16, C32 and C48 around the measured operating point with `5/1/6` fixed.
   Use the service's latency SLO and admitted throughput to choose concurrency.
2. If real admission waits coincide with released-but-stale capacity, compare
   the release-feedback patch at that same load; retain cancellation checks.
3. If logs show material time-budget clipping, compare the 1500 ms policy
   against a single controlled alternative using the same model and requests.
4. If L3 stopping or waiting dominates, isolate the actual cache tier and
   evaluate the cost model using its logged effective inputs.

Further draft depth, chunk size, CP layout, communication SM allocation and
kernel changes remain separate experiments. The current observations do not
justify selecting a larger draft, reducing CP, or changing the SM split yet.

## Deployment inputs needed to finish the backport

Provide the repository/location or files for
`sglang/pd1/sglang-pd1-glm53.rbg.yaml`, the complete applied
`0041-dp-admission-kvfix.patch` and patch ordering, plus raw results behind
`bench-pd1-20261027-glm53/report.html`. The live resolved P/D/router arguments,
cost/time JSON and per-rank startup capacities are also useful. The report HTML
alone cannot establish final token counts, failures or MTP counter coverage.

Review these changes against the actual v0.5.18 backport before producing its
next ConfigMap. Preserve the separately scaled-down deployment. Until that
integration and hardware acceptance run, this branch contains tested CPU/HTTP/
IPC changes and an executable measurement procedure, not new online performance
results.
