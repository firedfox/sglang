# pd1 v0.5.18 follow-up backports

This package targets the operator-supplied GLM-5.3 pd1 deployment, not the
current development branch's runtime layout. The supplied 29 patches were
replayed in filename order against official v0.5.18 commit
`71de97b264b04dcd514cf904003028aefe9775c8`; every `git apply --check` and apply
succeeded without conflicts. Every archive file also matched the supplied
ConfigMap data. `baseline_inputs.json` records their hashes.

The three additional runtime patches are:

| Patch | Purpose | Version-specific adaptation |
| --- | --- | --- |
| `0042-dp-release-feedback.patch` | Publish released admission capacity after result processing | Retains `0041`'s optional `req.kv`, with Mamba and backup ownership on Req; does not introduce a newer load publisher |
| `0043-prefill-time-diagnostics.patch` | Count committed time-budget cuts and snapshot estimated batch shapes | Uses the old `Req.extend_range` layout |
| `0044-l3-cost-diagnostics.patch` | Explain actual cost estimates and agreed termination in a terminal DEBUG record | Preserves the old `all_ranks_ready_or_any_terminated` rule and existing cache-observation counters |

These are additional patches after `0041`. Do not apply the newer runtime's
commits directly to this older source. The benchmark client changes belong to
the development checkout and can be used separately against the v0.5.18 HTTP
API; they are not copied wholesale into the older worker package.

## Build the reviewed deployment bundle

Use Python with PyYAML installed. Supply the original files locally:

```bash
python3 examples/runtime/glm53_pd/backport_v0518/build_bundle.py \
  --patches-zip /path/to/patches.zip \
  --configmap /path/to/live-fix-patch-cm.yaml \
  --deployment /path/to/sglang-pd1-glm53.rbg.yaml \
  --output-dir /path/to/pd1-followup
```

The builder verifies the entire baseline, the extension hashes, source-state
manifest, deployment identity and namespace. It rejects incomplete or altered
inputs and an existing output directory. The input manifests and full generated
deployment bundle are not stored in this source directory.

The generated bundle contains:

| File | Role |
| --- | --- |
| `configmap.yaml` | New immutable ConfigMap with a content-derived name; 29 original plus 3 new patches |
| `patches/` | The complete patch stack, exact source-state hashes and startup installer |
| `runtime-only.patch.json` | Guarded JSON Patch changing only Prefill and Decode patch-volume references |
| `runtime-only.rollback.json` | Guarded inverse patch restoring the original ConfigMap references |
| `runtime-only.rbg.yaml` | Complete candidate RBG for inspection |
| `baseline.rbg.yaml` | Supplied baseline RBG for comparison |
| `router-rank-selection.patch.json` and rollback | Separate experiment removing only the router's `--dp-aware` line |
| `router-log-level.patch.json` and rollback | Separate experiment changing only router `RUST_LOG` to `info` |
| `review.json` | Input hashes, resource names, patch count and exact change scope |

The runtime-only candidate preserves the supplied P/D arguments, queue capacity,
cost/time models, MTP settings, all-CP transfer environment, replica counts,
Router and Mooncake templates. The two router candidates are independent
experiments, not additional changes in the runtime-only candidate. Do not use a
directory-wide apply command: the directory intentionally contains alternatives
and rollback files.

### Startup application and rollback

The generated `fix-patch.sh` invokes the standard-library `apply-stack.py`
before the worker starts. `stack.json` identifies all affected source files in
three states: official v0.5.18, the supplied 29-patch baseline, and the final
32-patch candidate.

The installer verifies patch bytes and source hashes, applies the required
suffix in a temporary source copy with GNU patch and zero fuzz, and verifies
the final hashes before installing changed bytes. A fully installed candidate
is a no-op on a second run. An unknown or partially modified source tree is
rejected; it does not guess which overlapping hunks were previously applied.
This avoids relying on per-patch reverse checks after later patches have changed
the same context. Installation failures stop worker startup.

The source must match a recorded state. An image tag alone does not establish
that its source bytes match the upstream release. A custom image mismatch must
be investigated and rebuilt against its actual source; there is no force/bypass
option. The supplied image contents have not been pulled or inspected here.

Use the generated source preflight before a rollout:

```bash
python3 /path/to/pd1-followup/patches/apply-stack.py \
  --source-dir /path/to/v0518-source \
  --patch-dir /path/to/pd1-followup/patches \
  --dry-run
```

On the deployment host, inspect `review.json` and use the resource names it
records. For the supplied pd1 resource:

```bash
kubectl apply --server-side --field-manager=sglang-pd1-followup \
  -f /path/to/pd1-followup/configmap.yaml

kubectl -n sglang1 patch rolebasedgroups.workloads.x-k8s.io sglang-pd1 \
  --type=json --patch-file=/path/to/pd1-followup/runtime-only.patch.json
```

Server-side apply avoids storing the large patch ConfigMap again in a
client-side last-applied annotation. The JSON Patch tests resource identity,
role positions, worker names, complete startup commands and old references
before replacing them. The builder requires the supplied fail-closed startup
sequence, with `set -euo pipefail` immediately followed by the patch installer
before the server starts. Keep the original
ConfigMap for rollback. If readiness or acceptance fails, the inverse patch
restores the original references for replacement containers:

```bash
kubectl -n sglang1 patch rolebasedgroups.workloads.x-k8s.io sglang-pd1 \
  --type=json --patch-file=/path/to/pd1-followup/runtime-only.rollback.json
```

This is configuration rollback for replacement workers, not an instruction to
reverse source patches inside a running process. Observe the RBG controller's
rollout, startup, readiness and request-drain behavior before sending benchmark
load. These commands were prepared but were not executed against a cluster.

The supplied Service document is `ClusterIP`, with no NodePort field. The
operator separately reported NodePort 31180. This bundle emits no Service
mutation, so applying its narrow RBG patch cannot inadvertently replace the
live exposure configuration with the uploaded desired Service. It also does not
address the separate scaled-down `sglang-pd1-single2` resource.

## What the uploaded evidence changes

The current evidence consists of `bench-run5-eagle65-summary.jsonl`,
`bench-report-data.json`, `models-params.json`, the RBG manifest, patch archive
and patch ConfigMap. These are source inputs for this review; summary values
are not a new benchmark produced by this development work.

| Observation | Consequence |
| --- | --- |
| Actual backend is `sglang-oai-chat`; dataset is `generated-shared-prefix` | Native/random testing is a different experiment, not a replay of this result |
| Reported 96 completions, 98,304 output tokens, 43.726501 s | The reported 2248.156 output tokens/s is arithmetically consistent |
| Reported input total is 13,099,124, not 12,582,912 | The old client sums dataset prompt estimates; do not treat either a nominal 128K label or this old sum as verified server usage |
| Report contains only aggregate results | Per-request failures, final usage and cache tiers cannot be reconstructed |
| Eight rank acceptance values average to 3.730115 | This is an unweighted mean with unknown windows, not an all-request weighted acceptance length |
| Client limit is C32 but old reported peak is 40 | Old whole-second concurrency buckets cannot establish actual in-flight peak |
| Old token peak is 1024 while mean throughput is 2248 | Chat chunk observations cannot be interpreted as per-token arrival rates |
| Prefill E2E histogram sums to 192; Decode TTFT/E2E each sum to 96 | Sampling windows/labels need verification before comparing those distributions |

The corrected client now applies final-usage and terminal-event auditing to
`sglang-oai-chat`. It requests usage by default, requires actual completion
counts, rejects errors and truncated streams, and records server prompt counts
with coverage and source. It reports Chat token peak as `null` without
per-token timestamps. v0.5.18 streaming Chat does not supply per-request
speculative counters; coverage and weighted acceptance remain unknown rather
than being manufactured from the rank averages. Other shared Chat backends
retain their prior protocol behavior.

### Parameters now verified from the files

The supplied cost model has recompute rate 47,000 tokens/s, margin 1.2 and a
minimum observation of 0.2 seconds. It has no configured read rate; fixed
recompute cost defaults to zero. Therefore the reported 34k read rate is not a
coefficient in that JSON. A valid, reusable-partial-KV sample with that effective
rate would vote to cancel: its threshold is about 39,166.7 tokens/s. Missing or
short samples, a different observed rate, independent sidecars and the older
multi-rank agreement can change the final result. Use the terminal DEBUG
decision record to identify which condition occurred.

The time model is fixed 120 ms, linear 0.001 ms/token and quadratic
`1.2e-7 ms/token²`. It predicts 2312.656 ms for one cold 131072-token request;
at 1500 ms its unconstrained fit is 103152 tokens, or 103104 with page64
alignment. Additional query-tile constraints can reduce that further. These
are model predictions, not GPU timings. Hot-prefix requests can exercise a
very different part of the model.

The old cache metric remains
`sglang:cache_operations_total{operation="prefetch",reason=...}`. It records
each rank's terminal reason: a rank that already finished its own read may
report `complete` even when another rank caused a cost stop. Do not sum it into
a global cost-canceled request count. The new terminal log includes
`local_vote`, `agreed_can_terminate`, `agreed_stop_reason` and
`consensus_rule="all_ranks_ready_or_any_terminated"` for this older implementation.

### Configuration questions to resolve before further MTP tuning

The uploaded Prefill command omits speculative flags; Decode explicitly uses
EAGLE steps 5, top-k 1, draft tokens 6. Both roles already enable all-CP-rank
transfer. The runtime-only bundle preserves this baseline. Before increasing
draft depth or adding Prefill speculation, inspect the running resolved
arguments, target/draft pools, metadata transfer and initial MTP state using
the actual image. Source paths for draft transfer are conditional on Prefill's
own speculative state; successful text generation alone does not verify that
the intended draft-state path was exercised.

The router still has external DP-aware routing, request limit 1600, five
retries, broad DEBUG environment logging and older model/parser arguments.
Its separate Gateway image does not run the Python patch installer. Thus these
29 or 32 worker patches do not establish that the development branch's Rust
Gateway admission/cancellation fixes are deployed. The separate rank-selection
experiment removes only external rank pinning, so the Decode controller can
select a rank with its own `total_tokens` policy; compare this under the same
workload before combining it with another change. Model/parser alignment and a
Gateway upgrade require checking that image's CLI and HTTP path.

The supplied Decode manifest leaves the queue timeout unspecified. The applied
0040 code defaults it to 1.0 second and the snapshot timeout to 30.0 seconds;
this conclusion assumes there are no additional runtime argument overrides.
Queue capacity 64 belongs to the controller as a whole.

## Next benchmark plan

Use the development branch's corrected client in a dedicated client environment,
with its `python` directory selected in `PYTHONPATH`. Record its revision and
resolved package path. Do not replace the worker's v0.5.18 package with the
current runtime merely to obtain the newer benchmark client.

The uploaded summaries do not contain the original GSP groups, question/system
lengths, preparation/order flags or seed. Obtain them from the original command
or explicitly choose a new controlled workload. The planner requires that
choice rather than inventing it:

```bash
mkdir -p results/glm53-chat-next
python3 examples/runtime/glm53_pd/plan_next_round.py \
  --workload chat-gsp --base-url "$GLM53_ROUTER_URL" \
  --tokenizer "$GLM53_TOKENIZER" --seed "$BENCH_SEED" \
  --gsp-num-groups "$GSP_GROUPS" \
  --gsp-prompts-per-group "$GSP_PROMPTS_PER_GROUP" \
  --gsp-system-prompt-len "$GSP_SYSTEM_PROMPT_LEN" \
  --gsp-question-len "$GSP_QUESTION_LEN" \
  --gsp-range-ratio 1 --output-dir results/glm53-chat-next \
  > results/glm53-chat-next/plan.json
```

This fixes 96 single-turn requests, 1024 output-token budgets and C32. Range
ratio 1 is required because the generator also varies output lengths for other
ratios. Actual Chat prompt-token totals remain unknown until server usage is
recorded. Optional generation/preparation/order flags must also match across
comparisons. The planner emits three alternating rounds, each with two Decode
profiles and fresh/replay pairs, for 12 jobs. It does not change Prefill or run
requests. Check the P/D compatibility question above before profile changes.

An explicit `--workload native-random --dataset ... --seed ...` still produces
the independent token-ID 128K baseline. Its known input total must not be used
as a validity check for Chat/GSP results. Save actual generated payload/cache
identity as well as CLI parameters; GSP's local cache and preparation choices
can otherwise affect reproducibility.

## CPU validation

The runtime backports were tested against the reconstructed v0.5.18 tree:

- DP release and load snapshot backends: **47 tests and 4 subtests passed**,
  including real local ZMQ. The original scheduler methods fail the expected
  post-release publication assertion; the backport passes it.
- Prefill/L3 diagnostic tests plus the older PrefillAdder group: **43 tests
  passed**. These use real control methods with CPU tensors; L3 collective
  outcomes are test doubles, not a new multiprocess/GPU transfer run.
- Corrected Chat benchmark plus existing client regressions: **48 tests and
  52 subtests passed**, including localhost HTTP.
- Planner and original launcher: **28 standard-library tests passed**.
- The packaged runtime tests and existing older PrefillAdder/load-snapshot
  groups ran together: **90 tests and 29 subtests passed**.
- Bundle builder, source-state installer, planner and original launcher ran
  together: **72 tests and 116 subtests passed**. This includes 19 builder cases
  and 25 installer cases. The combined runs overlap the groups above; their
  counts must not be added as unique tests.

The older runtime test environment uses Transformers 5.12.1 and Tokenizers
0.22.2. CPU PyTorch is 2.14.1; the target source declares 2.13.0. No numerical,
CUDA, RDMA or deployed-image compatibility claim follows from these CPU tests.

In particular, the older transfer queue does not implement the newer
`resolve_deferred_releases`/`is_abort_release_safe` protocol. The backport tests
retain ownership during `Transferring` and release admission after the older
queue's existing cleanup clears ownership. They do not establish RDMA physical
drain or add a new drain acknowledgement protocol.

To run the included backport tests, point imports at the prepared older source:

```bash
PYTHONPATH=/path/to/prepared-v0518/python \
SGLANG_CACHE_DIR=/tmp/sglang-pd1-backport-cache \
python3 -m pytest -q \
  examples/runtime/glm53_pd/backport_v0518/tests
```

Full stack replay was performed with the supplied patch bytes: 32 patches from
the official affected source files, or 3 from the 29-patch baseline, produced
the same candidate hashes. Repeated application was a no-op, unknown state was
rejected without changes, and all 43 affected Python files compiled. Actual
generated JSON patches and their inverses were checked against the supplied
RBG. The generated bundle's validation files record these checks.
Hardware startup, MTP state, cancellation/transfer behavior and measured gains
remain deployment acceptance work; this package contains no new online result.
