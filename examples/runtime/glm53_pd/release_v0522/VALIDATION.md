# 0.5.22 迁移验证

本记录对应官方 `baa9befa381511cd4d56f83980951e5a45fffc6d` 上的 GLM-5.3 PD 功能迁移。执行环境是 CPU 开发容器，没有 GPU、目标镜像内执行环境或 Kubernetes 集群连接。源码兼容性、协议、所有权、部署导出和 CPU/Gloo 回归已经验证；模型数值、GPU/RDMA 时序与吞吐仍需实际镜像验收。

## 已执行的测试

| 测试组 | 结果 | 验证内容 |
| --- | --- | --- |
| DP admission / HTTP / IPC / context / watchdog 最终集成 | 147 passed，704 subtests passed | rank ledger、ACK、拒绝、排队、KV 实际释放、RID 生命周期、真正执行 controller 构造、实际 CPU allocator、真实子进程存活 |
| Cache / Prefill release 回归 | 265 passed，271 subtests passed，2 skipped | 成本模型、时间预算、FIFO、PP drain、Unified Cache、SDK wrapper、FPM、Bootstrap polling、终态清理 |
| Storage prefetch 生命周期 | 27 passed，16 subtests passed | localhost 双 rank Gloo 的 deadline / cost 一致性，取消、stage、insert、ACK 所有权 |
| API / Mooncake 兼容 | 333 passed，210 subtests passed | Chat、Completion、Responses、usage、BOM、web_search、HTTP status、session failure、thread health、metadata 生命周期 |
| API 后续原生工具回归 | 16 passed，14 subtests passed | 新增 native Kimi builtin filtering 与既有 Kimi / legacy API 合约；与上一行部分重叠，不累加 |
| 压测客户端测量 | 46 passed，47 subtests passed | native / Chat 请求、流式结果、token / usage 审计与统计 |
| 部署包 / 安装器 / launcher / planner 最终集成 | 81 passed，121 subtests passed | 真实 GNU patch 重放、幂等、错误源码拒绝、import 目录检查、Python tree profile、环境变量保护、升级与精确回滚、命令构造 |
| Rust Gateway | 库与 HTTP target 编译成功，44 tests passed | 22 项 admission、9 项 PD unit、12 项真实 HTTP routing、1 项 200,000 轮唤醒压力回归 |
| 实际 pd1 参数解析 | Prefill 88、Decode 77 个 argv token 均被接受 | 用当前 release 的真实参数解析器检查提供的启动参数；不加载模型 |

表中 subtests 为 pytest-subtests 的独立计数口径，不与顶层测试数量混称。各批次可能共享已有回归，不将它们简单累加为一个无重叠总量。

两个 CUDA-only 跳过项覆盖 transfer stream ordering 与 direct backend device translation。其余已列 Gloo 用例确实运行了本机多进程通信；SDK / RDMA 合约测试使用 test doubles，不代表实际 RDMA 传输压测。

## 迁移中特别验证的差异

- 实际构造 `DataParallelController`，同时覆盖关闭 admission 的 ordinary DP / attention DP 和开启 admission 的真实账本、dispatcher、SenderWrapper 与 inproc ZMQ，捕获只执行 `__new__` 的单测无法覆盖的初始化错误。
- EAGLE 接受多个 token 但 EOS / 输出上限提前终止时，reasoning usage 只统计真实输出 token。
- 新上下文预留保留 release allocator 的内存边界，不能用推测窗口保护替代 KV 容量检查。
- scheduler terminal 与物理 KV 释放分开；先终止但仍有异步所有权的请求不会提前归还 admission 额度。
- 在 release PP optional-bool / ACK / stage / insert 路径上检查缓存生命周期，容量 FIFO 不消耗或绕过 release 存储重试预算。
- 实际部署选择 Python TreeCore，安装前校验其环境变量，避免官方默认 Rust tree 跳过 Python 修复。
- deployment patch 保留原 env 的其他变量、原启动参数、Router、Mooncake 和 Service；整体 env / volume / command 漂移使校验失败。

## 可复现入口

Python 回归使用 Python 3.12 的 CPU 测试环境。核心源码与测试保持 release 的依赖定义；为离线执行协议回归，仅测试环境额外安装相应依赖，没有修改产品依赖声明。

```bash
export PYTHONPATH=python
export SGLANG_CACHE_DIR=/tmp/sglang-v0522-cache
export GLOO_SOCKET_IFNAME=lo

python -m pytest -q \
  test/registered/unit/managers/test_dp_admission.py \
  test/registered/unit/managers/test_dp_admission_integration.py \
  test/registered/unit/managers/test_dp_controller_init.py \
  test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py \
  test/registered/unit/entrypoints/test_dp_admission_http.py \
  test/registered/unit/server_args/test_dp_admission_args.py \
  test/registered/unit/managers/test_scheduler_init_req_max_new_tokens.py \
  test/registered/unit/utils/test_subprocess_watchdog.py \
  test/registered/unit/managers/test_batch_result_processor_spec_grammar.py

python -m pytest -q \
  test/registered/unit/test_glm53_pd_v0522_bundle.py \
  test/registered/unit/test_glm53_pd_apply_stack.py \
  test/registered/unit/test_glm53_pd_bundle.py \
  test/registered/unit/test_glm53_pd_launch.py \
  test/registered/unit/test_glm53_pd_plan.py

python -m pytest -q \
  test/registered/unit/mem_cache/test_storage_prefetch_lifecycle.py
```

缓存的完整用例清单与命令环境记录在部署包的验证附件中。部署包源码重放结果和源码 commit 由最终导出时的 `migration-manifest.json`、`stack.json` 及验证附件精确标识。

## 实际导出补丁的完整重放

已从发布的迁移提交 `71924fcad49c6a6be465ec35e2c3f60fa55a2324` 导出实际补丁，在干净的官方基线文件上执行安装器和 GNU patch，详细结果见 [`SOURCE_REPLAY.json`](SOURCE_REPLAY.json)。该记录验证真实部署差异，不只验证最小测试 fixture。

- Python：53 个变更文件加 `version.py`，共 54 个受管路径，全部最终 SHA-256 和字节与已发布源码相同；语法解析通过。
- 安装器：dry-run 不修改源码，完整安装成功，重复执行为 no-op；刻意制造的未知源码状态被拒绝且没有进一步写入；真实 bash 启动入口的 Python profile / import 检查通过。
- Gateway：7 个 Rust 源码/测试路径的独立补丁零 fuzz 重放成功，最终字节与已发布源码一致。
- 用户提供的 RBG：正向 JSON Patch 与候选完全一致，逆向操作精确恢复输入；6 个字段操作分别为两个镜像、两个 ConfigMap 引用和两份仅设置 tree backend 的 env 列表；其他角色不变。
- 生成的不可变 ConfigMap 包含全部 5 个安装文件，data 为 466,139 字节，低于 1 MiB；所有导出文件 SHA-256 校验通过。

Python 补丁 SHA-256：`8ec58a7ff36c0966f17e0bc001b1f61c92ed2af5382d1413eca22bd68a2b688c`。

Gateway 补丁 SHA-256：`5b7152a965bd27db561bce0eed370806a35bf31d3e6b955a04ce308667aa31a7`。

后续发布记录和导出器附件更新不改变这两份源码差异；最终交付包会再次记录其精确 source revision 和重放结果。上述检查基于官方 Git 字节，不代表已拉取或启动官方容器镜像。

## Rust Gateway

Gateway 使用 release 的 Cargo 定义与 Rust 1.90 验证。HTTP PD 迁移按 release 的双请求结构适配；不存在对新 main decode gate 的依赖。测试所用锁文件随部署包的验证附件保存，不修改上游是否跟踪 `Cargo.lock` 的策略。锁文件 SHA-256 为 `baf76a88b79077de6d9bfb25a7dea4b0d620d85fffb971a41c4b4fc165dea60c`。

库与 HTTP 测试目标实际编译通过。44 项回归全部通过，包括 token bucket、严格并发生命周期、app context、PD 提前拒绝/取消、真实 HTTP 端点和 200,000 轮唤醒压力测试（8.95 秒）。完整命令、日志与源文件哈希在部署包的验证附件中。该验证没有生成适配线上 Router 镜像的 production wheel；构建和安装入口见 [`README.md`](README.md)。

## 仍需 GPU / 镜像验收的项目

1. 在实际选用的 0.5.22 镜像中运行源码预检与 dry-run，确认镜像内源码、Python 环境、CPU 架构和驱动兼容。
2. 模型加载与 GLM-5.3 数值/工具调用/流式输出正确性；1M 上下文及推测窗口边界。
3. TP8/CP8 Prefill、DP8/EP8 Decode、DeepEP、Mooncake L3 和 PD RDMA 的真实通信与取消释放。
4. 冷缓存与热缓存、并发与输入长度组合、EAGLE 6/5、cost-aware 和时间预算的实际 TTFT/TPOT/吞吐；Python TreeCore 的调度 CPU 开销。

本次没有执行线上升级，没有从旧版结果推导新版本性能提升。已有校准参数保留为首轮验收输入，不能当作 0.5.22 上已经重新校准的参数。
