# Validation record for the GLM-5.3 PD development branch

This record covers CPU, HTTP, IPC, and scheduling checks performed while building
`feature/glm53-pd-optimization-20261026`, based on
`ef11eb3b76354ffbabd9f17a2343e79522e41cf2`. It is not a GLM-5.3 GPU benchmark or
a production deployment result. No model weights were loaded for these checks.

## Initial branch checks

| Area | Result | What the checks establish |
| --- | --- | --- |
| Gateway build | Full library and `routing_tests` targets compiled | Rust code and the HTTP integration target compile together |
| Strict admission | 22 regular unit tests passed | Permit lifetime, zero-refill admission, cancellation, queue capacity, and configuration |
| Admission wakeup stress | 1 explicit test passed, 200,000 rounds | The zero-refill waiter does not lose the release wakeup in the exercised race |
| PD cancellation | 9 unit and 13 HTTP routing tests passed | Early failures drop the paired request, preserve status/Retry-After, and account for the right worker |
| Decode rank admission | 195 tests and 168 subtests passed | Controller/scheduler accounting, HTTP failures before stream headers, cancellation/cleanup, transport errors, supported configurations, and compatibility |
| L3 cost policy | 47 tests passed, including 3 multiprocess Gloo tests | Cost selection, rank agreement, sticky cancellation reason, partial completion, lifecycle, and argument validation |
| Prefill time budget | 96 tests passed | Pure cost model, real PrefillAdder paths, alignment and continuation, load-back, hard limits, configuration, and a real CPU EAGLE bigram cache/checkpoint/tail cycle |
| Combined argument compatibility | 44 tests and 70 subtests passed | Decode admission, L3 cost, Prefill time, hook registry, documented fields, legacy positional fields, and load-balance methods |
| Standard-library launcher | 17 tests passed | Role commands, matching model/MTP settings, input validation, dry-run behavior, and mocked execution |
| Documentation | 4 modified MDX pages compiled; Mintlify broken-links passed | MDX syntax and internal documentation links |

Counts are reported by test group; they are not a unique-test grand total.
Several argument and compatibility tests occur in more than one check.
An independent reviewer also checked the time-budget selector against an
exhaustive legal-chunk oracle for 100,000 seeded random cases; all matched.
The repository's Python formatting/import checks and configured Ruff rules
passed for the new scheduling code, as did `git diff --check`.

## Follow-up after the operator's pd1 report

The operator separately reported a deployed v0.5.18 backport and four GPU runs.
At this stage, the supporting files and deployed patches had not yet been
supplied. The following checks validate this branch's new code; they neither
reproduce those GPU measurements nor establish a performance gain on pd1.
The later file-based reconstruction is recorded separately below.

| Area | Result | What the checks establish |
| --- | --- | --- |
| Benchmark measurement | 34 tests and 27 subtests passed | Actual localhost native SSE/JSON, truncated EOF and abort/error handling, all-failed main-run JSON, workload timing, weighted MTP counters, exact in-flight intervals, existing prompt/goodput/reasoning/steady-state compatibility |
| Decode release feedback | 83 tests and 21 subtests passed | Real scheduler/EAGLE result control flow, deferred ownership, allocator floor, stale running references, event coalescing, failed-write retry, ZMQ roundtrips and default-off compatibility |
| L3 diagnostics | 78 tests passed, including multiprocess Gloo cases | Cumulative outcome deltas and resets, reused stats objects, bounded labels, one terminal debug record per attempt, cost inputs, rank agreement and unchanged lifecycle decisions |
| Prefill budget diagnostics and launcher/planner | 122 tests passed | The existing 99-case Prefill group, 19 launcher cases and 4 offline manifest cases; committed shape/count snapshots, input/alignment limits, EAGLE profiles, explicit queue overrides and namespace isolation |

The release-feedback regression also ran against the original scheduler methods
from `f72308a`: it failed its expected post-release publication assertion, with
no fixture error. The new implementation passes. A separate seeded comparison
of 10,000 old/new pure time-budget selector inputs found identical selections.
These are correctness checks, not CPU overhead or GPU speed benchmarks.

Local HTTP, Unix IPC and Gloo tests ran with permission to create local sockets.
No CUDA kernel, RDMA transfer or running Mooncake service was exercised. Rust
Gateway code was unchanged in this follow-up; its earlier build is recorded
above rather than claimed as a new build. Python import/format checks, the
configured Ruff rule subset and `git diff --check` were run on modified files.
Only example Markdown was changed here; the earlier MDX validation record below
is unchanged.

Reproduce the follow-up groups from the repository root in a CPU test environment:

```bash
export PYTHONPATH=python
export SGLANG_CACHE_DIR=/tmp/sglang-glm53-test-cache

python3 -m pytest -q \
  test/registered/unit/bench/test_serving_measurement.py \
  test/registered/unit/bench/test_bench_serving_prompt_len.py \
  test/registered/unit/bench/test_bench_serving_goodput.py \
  test/registered/bench_fn/test_bench_serving_reasoning_stream.py \
  test/registered/bench_fn/test_steady_state_benchmark.py

python3 -m pytest -q \
  test/registered/unit/managers/test_dp_admission_integration.py \
  test/registered/unit/managers/test_dp_admission.py \
  test/registered/unit/managers/test_load_inquirer.py \
  test/registered/unit/managers/test_scheduler_on_idle_load.py \
  test/registered/unit/managers/test_load_snapshot_backends.py

GLOO_SOCKET_IFNAME=lo python3 -m unittest \
  test.registered.unit.mem_cache.test_prefetch_cost \
  test.registered.unit.mem_cache.test_storage_prefetch_metrics \
  test.registered.unit.mem_cache.test_storage_prefetch_lifecycle \
  test.registered.unit.server_args.test_hicache_prefetch_cost_args \
  test.registered.unit.observability.test_stat_loggers_di

python3 -m unittest \
  test.registered.unit.managers.scheduler_components.test_prefill_time_budget \
  test.registered.unit.server_args.test_prefill_time_budget_args \
  test.registered.unit.managers.test_prefill_adder \
  test.registered.unit.managers.test_deterministic_chunk_align \
  test.registered.unit.managers.test_scheduler_held_row_admission \
  test.registered.unit.test_glm53_pd_launch \
  test.registered.unit.test_glm53_pd_plan
```

## Environment and limits

### File-based v0.5.18 backport follow-up

The subsequently supplied patch archive, ConfigMap, RBG manifest, calibrated
parameters and aggregate benchmark results have now been inspected. All 29
patches matched the ConfigMap and applied cleanly to official v0.5.18 commit
`71de97b264b04dcd514cf904003028aefe9775c8`. The separate
[backport package](backport_v0518/README.md) records the version-specific
adaptations, source hashes and deployment procedure.

| Area | Result | Scope |
| --- | --- | --- |
| Chat benchmark audit | 48 tests and 52 subtests passed | Actual final usage, terminal events, reasoning/tools, truncated continuous-usage streams, prompt-count sources, unknown token peak and other backend compatibility |
| v0.5.18 DP release backport | 47 tests and 4 subtests passed | Older Req ownership, actual local IPC, EAGLE result handling, old transfer queue cleanup and snapshot backends |
| v0.5.18 Prefill/L3 diagnostics | 43 tests passed | Older extend-range layout, existing cache-observation lifecycle, CPU tensors and simulated collective outcomes |
| Explicit workload planner | 28 tests passed | 9 planner cases plus the 19 unchanged launcher cases; Chat/GSP explicit parameters and separate native baseline |
| Bundle builder | 19 tests passed | Real archive/YAML checks, guarded JSON Patch and rollback, independent Router candidates, shell argument quoting and worker startup rejection on installer failure |
| Source-state installer | 25 tests and 13 subtests passed | Real GNU patch, exact source-state/hash checks, repeated application, symlink rejection, concurrent modification and installation rollback |

The packaged older-runtime tests plus its existing PrefillAdder and load-snapshot
groups also ran together: **90 tests and 29 subtests passed**. The builder,
installer, planner and unchanged launcher ran together: **72 tests and 116
subtests passed**. These are combined reruns of the groups above, not additional
unique-test totals.

Using the complete supplied inputs, the generated installer applied all **32
patches** from the official affected source files, and only the **3 extensions**
from the supplied 29-patch state. Both produced the same candidate hashes;
repeated application was a no-op. An unknown source state was rejected without
changing its files. All **43 affected Python files** compiled. The generated
runtime and optional Router JSON patches were also checked against the actual
supplied RBG, including exact rollback and independent change scope. The bundle
records these checks in its validation JSON files.

The old runtime environment pins Transformers 5.12.1 and Tokenizers 0.22.2;
it uses CPU PyTorch 2.14.1 rather than the target source's declared 2.13.0.
No GPU or actual image compatibility is inferred. The older transfer queue
does not have the newer physical-drain acknowledgement helpers; the backport
retains its existing cleanup protocol. The source replay and prepared bundle
do not establish that any Kubernetes rollout or new hardware benchmark ran.

### Initial development environment

The checks used Python 3.12 with CPU PyTorch, the repository's runtime test
dependencies, and Rust 1.90. The Gateway build used `vendored-openssl` and
Protobuf compiler 33.0. Local Unix-socket HTTP tests and Gloo loopback tests ran
with permission to create those local sockets. None of these checks exercised
CUDA kernels, GPU memory allocation, real NIC transfers, or a Mooncake cluster.

Four existing real HiCache allocator cases could not initialize in this
environment: available host memory was below the runtime's existing 10 GiB
host reserve. This failure occurred before the new cost policy initialized.
The production reserve was not weakened, and these cases are not reported as
passing. Other selected cache compatibility and ordering checks passed.

The full `mint validate` command was attempted, but Mintlify's Internet
initialization check blocked the build in this environment. The independent
MDX compilation and `mint broken-links` checks succeeded. A full documentation
site build remains for normal CI.

GPU startup, model output correctness, GLM-5.3 MTP target/draft compatibility,
per-rank KV capacity, transfer completion, and performance remain to be tested
on the deployment hardware. Use [GPU_ACCEPTANCE.md](GPU_ACCEPTANCE.md) for the
separate acceptance and calibration procedure. No throughput improvement,
latency reduction, or maximum long-context concurrency is claimed here.

For the generic Mamba load-back path, the existing GPU fixture
`test_admission_keeps_mamba_h2d_destination_on_full_overdelivery` in
`test/registered/unit/mem_cache/unified_radix_cache/test_unified_radix_cache_load_back.py`
also needs a GPU run with time budgeting enabled to inspect copied state and
slot ownership. The CPU admission tests cover control flow, not the actual H2D
copy.

## Reproducing the main checks

Run from the repository root with its CPU test dependencies installed. Choose a
writable cache directory; no checkpoint path is needed for these unit tests.

```bash
export PYTHONPATH=python
export SGLANG_CACHE_DIR=/tmp/sglang-glm53-test-cache

python3 test/registered/unit/test_glm53_pd_launch.py

python3 -m pytest \
  test/registered/unit/managers/test_dp_admission.py \
  test/registered/unit/managers/test_dp_admission_integration.py \
  test/registered/unit/managers/test_data_parallel_controller.py \
  test/registered/unit/managers/test_load_inquirer.py \
  test/registered/unit/managers/test_load_snapshot_backends.py \
  test/registered/unit/server_args/test_dp_admission_args.py \
  test/registered/unit/server_args/test_resolution_hook_registry.py \
  test/registered/unit/server_args/test_server_args_docs_coverage.py \
  test/registered/unit/server_args/test_server_args.py::TestServerArgsPositionalOrder \
  test/registered/unit/server_args/test_server_args.py::TestLoadBalanceMethod \
  test/registered/unit/entrypoints/test_dp_admission_http.py \
  test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py \
  -q

GLOO_SOCKET_IFNAME=lo python3 -m unittest \
  test.registered.unit.mem_cache.test_prefetch_cost \
  test.registered.unit.mem_cache.test_storage_prefetch_lifecycle \
  test.registered.unit.server_args.test_hicache_prefetch_cost_args

python3 -m unittest \
  test.registered.unit.managers.scheduler_components.test_prefill_time_budget \
  test.registered.unit.server_args.test_prefill_time_budget_args \
  test.registered.unit.managers.test_prefill_adder \
  test.registered.unit.managers.test_deterministic_chunk_align \
  test.registered.unit.managers.test_scheduler_held_row_admission
```

The Gateway checks use the following targets and filters. The ignored stress
test is deliberately invoked separately; an ordinary test run does not execute
its 200,000 rounds.

```bash
cargo test --manifest-path sgl-model-gateway/Cargo.toml \
  --features vendored-openssl --lib core::token_bucket::
cargo test --manifest-path sgl-model-gateway/Cargo.toml \
  --features vendored-openssl --lib middleware::concurrency::
cargo test --manifest-path sgl-model-gateway/Cargo.toml \
  --features vendored-openssl --lib app_context::tests::
cargo test --manifest-path sgl-model-gateway/Cargo.toml \
  --features vendored-openssl --lib \
  core::token_bucket::tests::refill_zero_does_not_lose_wakeups_under_stress \
  -- --ignored --exact
cargo test --manifest-path sgl-model-gateway/Cargo.toml \
  --features vendored-openssl --lib routers::http::pd_router::tests::
cargo test --manifest-path sgl-model-gateway/Cargo.toml \
  --features vendored-openssl --test routing_tests pd_routing_test
```

These are targeted regression checks, not a claim that the repository's entire
GPU and distributed CI matrix ran. The new Decode admission, L3 cost policy, and
Prefill time model remain explicit opt-ins; the latter two require calibration
for the actual model and complete TP/CP group.
