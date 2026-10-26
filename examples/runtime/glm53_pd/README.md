# GLM-5.3 HTTP prefill/decode baseline

`launch.py` builds a consistent Router, Prefill (P), and Decode (D) command for
two GPU nodes, **each with 8 B300 GPUs**: one TP8/CP8 prefill engine on the first
node and one TP8/attention-DP8/EP8 decode engine on the second. The Router uses
Kubernetes service discovery and HTTP worker endpoints. Run one role per
container; the launcher does not create Kubernetes resources or Mooncake services.

This is a reviewable starting configuration for this development branch. GPU
startup, numerical correctness, capacity, and performance acceptance are still
required. The CPU tests validate command construction and execution boundaries;
they do not measure throughput or guarantee that 256 long requests fit in memory.

## Review and launch

From the repository root, use any Python 3 environment to print a launch plan.
The launcher uses only the standard library and does not import SGLang, load a
checkpoint, contact a service, or start a worker during a dry-run.

```bash
export GLM53_MODEL_PATH=/models/GLM-5.3/snapshots/30333038ada1f1dacb294a93270305a890b50c14

python3 examples/runtime/glm53_pd/launch.py prefill
python3 examples/runtime/glm53_pd/launch.py decode
python3 examples/runtime/glm53_pd/launch.py router
```

Each invocation prints JSON with `role`, the exact `argv` array, and an `env`
object containing only the launcher's environment overrides. With `--run`, the
launcher prints the same plan, merges those overrides into the inherited
environment, and calls `os.execvpe`. Arguments are passed directly, with no shell
expansion. Other environment variables, including Mooncake configuration and
credentials, are inherited and are not included in the JSON.

On the respective containers, after reviewing the plans:

```bash
python3 examples/runtime/glm53_pd/launch.py prefill --run
python3 examples/runtime/glm53_pd/launch.py decode --run
python3 examples/runtime/glm53_pd/launch.py router --run
```

Workers execute `sglang serve` from `PATH`; the Router executes
`python -m sglang_router.launch_router` using the launcher's Python interpreter.
Install the worker package and the Rust Gateway Python extension from this branch
in the runtime environment. The strict `--rate-limit-tokens-per-second 0`
behavior relies on this branch's Gateway admission fix.

P and D must load the same GLM-5.3 checkpoint revision, tokenizer, and chat
template. Workers require `--model-path` or `GLM53_MODEL_PATH`; the CLI value takes
precedence. A local path or a model repository ID is accepted without loading or
checking it during the dry-run. Prefer a pinned local snapshot in deployment.
The Router does not load a model, tokenizer, reasoning parser, or tool parser for
this HTTP baseline. If a custom chat template is needed, pass the same
`--chat-template /etc/sglang/chat_template.jinja` to **both worker roles**. Otherwise
the workers use their checkpoint's template.

## Deployment options

Run `launch.py ROLE --help` for the role's options. The defaults preserve the
existing deployment addresses and selectors:

| Option | Applies to | Default / behavior |
| --- | --- | --- |
| `--host` | All | `POD_IP` when nonempty, otherwise `0.0.0.0` |
| `--port` | All | `8000` |
| `--otlp-traces-endpoint` | All | `otel-collector-service.ops.svc.cluster.local:4317` |
| `--model-path` | P, D | Required unless `GLM53_MODEL_PATH` is set |
| `--ib-devices` | P, D | `GLM53_IB_DEVICES`, otherwise `mlx5_bond_0` through `mlx5_bond_7`, comma-separated |
| `--chat-template` | P, D | Use the checkpoint template unless explicitly supplied |
| `--namespace` | Router | `sglang2` |
| `--group-name` | Router | `sglang-pd2` |
| `--worker-port` | Router | `8000`; must match the workers' HTTP port |
| `--prometheus-port` | Router | `9090`, exposed on `0.0.0.0` |
| `--max-concurrent-requests` | Router | `256` HTTP requests per Router process |
| `--all-cp-ranks-transfer` | P, D | Explicit experimental opt-in; see below |
| `--enable-dp-admission` | D | Explicit per-rank request and full-generation KV reservations; off by default |
| `--hicache-storage-prefetch-cost-model` | P | Calibrated JSON; explicitly selects `cost_aware` instead of the default `timeout` |
| `--prefill-time-budget-ms` and `--prefill-time-model` | P | Paired opt-in for a calibrated per-batch time estimate; both absent by default |

The Router selects pods using `rbg.workloads.x-k8s.io/role-name=prefill` or
`decode`, together with `rbg.workloads.x-k8s.io/group-name=GROUP`. Its service
account needs the existing Kubernetes discovery permissions. For example:

```bash
python3 examples/runtime/glm53_pd/launch.py router \
  --namespace sglang2 --group-name sglang-pd2 --worker-port 8000

python3 examples/runtime/glm53_pd/launch.py prefill \
  --model-path /models/GLM-5.3 --host 10.0.0.11 \
  --ib-devices mlx5_bond_0,mlx5_bond_1 \
  --otlp-traces-endpoint collector.example:4317
```

Keep the deployment's existing Mooncake master/store settings or
`SGLANG_HICACHE_MOONCAKE_CONFIG_PATH` in the worker environment. Specifying
`--ib-devices` configures PD transfer devices; it does not configure a Mooncake
storage cluster. See the [Mooncake storage setup](../../../python/sglang/srt/mem_cache/storage/mooncake_store/README.md).

## Baseline choices

| Component | Configuration and purpose |
| --- | --- |
| Router admission | Strict concurrency `256`, refill `0`, queue `0`; excess requests receive HTTP 429 instead of accumulating in a Gateway queue. `--retry-max-retries 2` limits the Gateway retry loop to two attempts. |
| Router routing | P uses `cache_aware`, D uses `power_of_two`. External `--dp-aware` is omitted; each Decode engine owns selection of its attention DP rank. |
| Both workers | TP8, FP8 E4M3 KV cache, page size 64, context limit 1,048,576, FCFS, Mooncake transfer, `glm45` reasoning and `glm47` tool parsers. |
| Both workers' MTP | `EAGLE`, three draft steps, top-k one, four draft tokens. Matching settings also let Prefill prepare the draft state required by Decode. |
| Prefill | CP8 with `interleave`, 131,072-token chunk and prefill budgets, prefill request setting 256, static GPU memory fraction 0.8, DeepEP `normal` with the existing dispatch/combine SM settings of 64. |
| Prefill cache | HiCache `write_through`, `direct`, `page_first_direct`, Mooncake L3, and `timeout` prefetch policy. Host memory auto-sizing uses fraction 0.8; fixed `--hicache-size` and `--hicache-ratio` are omitted so auto-sizing remains active. |
| Decode | Attention DP8/EP8, DP LM head, `--load-balance-method total_tokens`, `--max-running-requests 256`, static GPU memory fraction 0.85, DeepEP `low_latency`, radix cache disabled. `--max-total-tokens` is omitted so startup profiles the KV pool. |
| Observability | Worker cache reports, metrics for all schedulers, estimated MFU metrics, request/Mooncake traces, and Router Prometheus/traces remain enabled. The original tokenizer/detokenizer worker counts of 32/16 are retained pending CPU profiling. |

`total_tokens` controls Decode's internal DP load balancing; its scheduler remains
FCFS. The configured context limit is a per-request ceiling, not a promise that
many 1M-token requests can coexist. Inspect the resolved per-rank request and KV
capacities at startup. HiCache's host-memory fraction is separate from the GPU
static-memory fraction: auto-sizing accounts for visible host/cgroup limits,
local ranks, and a host reserve. The `timeout` policy uses the runtime's storage
prefetch timing rules; this example does not invent a measured L3 deadline.
See the [HiCache fields](../../../python/sglang/srt/arg_groups/fields/memory.py)
and [auto-sizing implementation](../../../python/sglang/srt/mem_cache/hicache_auto_size.py).

### Request budget and multiple Routers

This baseline assumes **one prompt and `n=1` per HTTP request**. Gateway admission
counts requests; it does not weight `n`, native batched prompts, input tokens, or
reserved KV memory. Enforce those client constraints for this baseline. The
launcher does not add payload validation or claim that returning a Gateway permit
proves remote GPU KV has already been released.

The strict limit is per Router process. With multiple Routers sharing the same
Decode engine, divide the total admission budget between them: for example,
two Routers can each use `--max-concurrent-requests 128` to keep the configured
aggregate limit at 256. Budget for rolling-update overlap too. Independent Router
limits do not coordinate across replicas, and a 256-request cap still does not
account for request length or per-rank KV pressure. Lower the aggregate limit
when long-context capacity or measured latency requires it.

### Decode rank admission

After the baseline starts correctly, review the optional capacity-aware command:

```bash
python3 examples/runtime/glm53_pd/launch.py decode --enable-dp-admission
```

The opt-in adds `--enable-dp-admission`, a zero-length admission queue, a 1-second
queue timeout, and a 30-second snapshot timeout. Add `--run` only when ready to
launch. The default launcher does not enable this new policy before GPU
acceptance. The Gateway should select Decode pods; the Decode controller then
selects and reserves a rank, with no external `--dp-aware` rank override.

The controller reserves both a request slot and the full effective prompt/output
KV budget, including the runtime's decode and PD preallocation reserves and page
alignment. A scheduler acknowledgement transfers accounting from pending
dispatch to live commitments. Merely receiving a fresh timestamp, emitting a
final response, or canceling HTTP does not refund an allocation that is still
held by transfer or retraction cleanup. This protects a shared Decode engine
across Gateway replicas; Gateway limits remain useful for limiting CPU and HTTP
work before the engine's admission decision.

Set realistic `max_new_tokens` or OpenAI output-token limits on client requests.
When no output limit is supplied, the policy reserves the effective remaining
context (subject to the server's output limit). Large declared output budgets
can therefore reduce admitted concurrency even when requests usually stop
early. This policy trades speculative overcommit for explicit capacity
guarantees; the 256-request setting is still only an upper bound.

An individually impossible request receives 400, occupied budgets receive 429,
and unavailable or stale rank capacity receives 503. With zero queue size there
is no additional Decode admission wait. The Gateway cancels the paired Prefill
on early Decode rejection and preserves 429 and any upstream `Retry-After` header.

This first version supports the fixed single-node Decode layout in the example,
text generation, early Mooncake allocation, disabled decode radix cache, and
fixed top-k-one speculation. Native `/generate`, Chat Completions, and Completions
can return admission failures before streaming headers. Streaming batches and
`n > 1` streaming, `POST /v1/responses`, and Anthropic streaming are rejected in
this mode; they are unchanged when it is off. Sessions, multimodal/embedding
inputs, beam search, LoRA, adaptive speculation, role switching, and hybrid or
offloaded KV layouts are outside the initial contract. See the
[PD admission documentation](../../../docs/docs/advanced_features/pd_disaggregation.mdx#decode-dp-admission)
for the full scope.

### L3 read cost versus recomputation

Keep the default `timeout` policy until you have calibrated storage and
recomputation costs. To opt in with your own calibration file:

```bash
python3 examples/runtime/glm53_pd/launch.py prefill \
  --hicache-storage-prefetch-cost-model "$(cat /etc/sglang/prefetch-cost.json)"
```

The launcher passes the JSON directly as one argument and selects `cost_aware`.
It checks JSON object syntax; the runtime validates all coefficients before
model loading. The model requires `recompute_tokens_per_second`; it optionally
accepts `read_tokens_per_second`, `min_observation_seconds`, `cancel_margin`,
and `recompute_fixed_seconds`. The
[HiCache cost model reference](../../../docs/docs/advanced_features/hicache_best_practices.mdx#cost-aware-prefetch)
defines units, defaults, and bounds. No throughput estimate is supplied by this
example.

Cost-based stopping compares remaining read time with recomputing the unusable
prefix. Observed progress can replace the configured read rate. Without a usable
estimate it falls back to the existing timeout policy. It retains the cross-rank
stop decision, sticky cancellation, retry suppression, and sidecar cleanup.
The initial agreed stop reason stays `cost_aware` through later cancellation
polls. Monitor `sglang:storage_prefetch_unfulfilled_tokens_total{reason="cost_aware"}`
alongside useful cached tokens and TTFT; cancellation is not a storage miss.

Calibration must describe logical tokens for the actual model, context-length
range, and complete TP/CP cache group. A rate measured for another model or
divided by CP8 again can produce the wrong decision. Validate cold/L2/L3 cases
and mixed prompt lengths before enabling this opt-in in production.

### Prefill time budget

After collecting batch timing data for this model and TP8/CP8 layout, review a
Prefill command with the measured budget and model:

```bash
python3 examples/runtime/glm53_pd/launch.py prefill \
  --prefill-time-budget-ms "$PREFILL_BUDGET_MS" \
  --prefill-time-model "$(cat /etc/sglang/prefill-time-model.json)"
```

Both options are required together. The JSON accepts `fixed_ms`,
`linear_ms_per_token`, and `quadratic_ms_per_token2`; all coefficients must be
finite and nonnegative, and at least one token coefficient must be positive.
The launcher checks the positive finite budget and JSON syntax; the runtime
checks the model before loading weights. The baseline supplies no timing model.

For a request with cached prefix `h` and newly processed tokens `x`, estimated
work is `linear_ms_per_token * x + quadratic_ms_per_token2 * (2*h*x + x*x)`.
The fixed cost is charged once per batch. Use complete cache-group logical
tokens; do not divide them by CP8. The scheduler caps candidate work before
admission and charges only the committed request shape, including continuation
and cache load-back paths. Ordinary cache-mode host load-back determines the
final logical prefix from the matched path before H2D. Selection therefore uses
the restored prefix for cost and alignment, and a successful transfer does not
later lose its destination to a time-budget rejection. A load that returns zero
FULL tokens is reselected against the actual prefix without double charging.

This is a soft scheduling estimate, not a wall-clock deadline or a TTFT promise.
The existing chunk, prefill token, memory, and alignment limits still apply.
An empty batch may admit one minimum legal chunk even when it exceeds the time
estimate, to avoid starvation; it cannot bypass the original capacity limits.
The scheduler retains a single unfinished chunk owner. It uses no rank-local
clock feedback and introduces no extra cross-rank consensus.

The initial mode requires chunked, non-PP text prefill. Decode mode, mixed
prefill/decode chunks, dynamic chunking, dLLM, embedding mode, and bounded encoder
SWA replay are rejected. It also rejects `--enable-streaming-session`, whose
arbitrary reused prefix lengths can conflict with intermediate chunk alignment.
See the
[time-budget tuning reference](../../../docs/docs/advanced_features/hyperparameter_tuning.mdx#estimated-prefill-time-budget)
for calibration and limitations. Enable this separately from L3 cost stopping
until each policy has its own acceptance result.

### Optional all-CP-rank transfer experiment

The launcher explicitly sets
`SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=0` for both worker roles, overriding
an inherited value. To test the existing all-CP-rank transfer switch, add
`--all-cp-ranks-transfer` to **both** P and D invocations and review both JSON
plans; their override becomes `1`. This switch is not accepted for the Router.

Keep both endpoints consistent when changing the experiment. It is an opt-in
transport experiment, with no 8x bandwidth or end-to-end speedup promise. Confirm
correct transfer completion and compare NIC utilization, transfer latency,
TTFT, and tail latency on the actual topology before enabling it by default.
The runtime can independently require participation of additional ranks for
particular sharded KV layouts; the environment switch is not a universal rank
participation override. See the [PD connection logic](../../../python/sglang/srt/disaggregation/common/conn.py).

## CPU checks and GPU acceptance

The [validation record](VALIDATION.md) lists the CPU, HTTP, and IPC checks that
actually ran, together with the remaining environment and GPU limits.

The [GPU acceptance and calibration procedure](GPU_ACCEPTANCE.md) contains
source-checked benchmark commands, cache replay scenarios, admission and
cancellation checks, and a profiling path for collecting calibration data.
These commands have not been run on GLM-5.3 GPUs as part of this change.

The launcher and its tests need no GPU or SGLang installation:

```bash
python3 test/registered/unit/test_glm53_pd_launch.py
```

The tests cover role-specific commands, matching MTP settings, automatic cache
sizing, strict admission, no repeated flags, input validation, explicit opt-in,
and environment handling. `--run` is checked with a mocked `execvpe`; the tests
do not start servers.

Before production use, perform GPU startup and API smoke tests for streaming,
reasoning, tools, and client cancellation; verify resolved target/draft KV
layouts and request capacities on P and D. Then compare cold/L2/L3-cache cases
and mixed prompt lengths under the same request mix. Record TTFT/TPOT percentiles,
admitted throughput, rejection/retraction counts, per-rank KV usage, host memory,
and Mooncake transfer timing. Tune chunk size, MTP, or communication SM allocations
one at a time after those measurements are available.
