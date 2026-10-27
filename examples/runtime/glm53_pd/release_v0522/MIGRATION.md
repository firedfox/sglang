# 0.5.22 迁移范围与兼容性记录

## 源码边界

| 项目 | 固定来源 |
| --- | --- |
| 新分支 | `firedfox/sglang:feature/glm53-pd-v0.5.22-20261027` |
| 官方基线 | `sgl-project/sglang:release/v0.5.22` / tag `v0.5.22` |
| 基线 commit | `baa9befa381511cd4d56f83980951e5a45fffc6d` |
| 旧优化分支 | `feature/glm53-pd-optimization-20261026` |
| 旧优化 head | `ed565e76dd4aed44ab3f4141fae3450dba3b0bc2` |
| 旧优化共同起点 | `ef11eb3b76354ffbabd9f17a2343e79522e41cf2` |
| 旧线上补丁来源 | 用户提供的 29 个运行时补丁，以及已交付的 0042、0043、0044 |

新分支直接继承官方 release 基线。迁移提取旧分支 14 个定制提交的功能差异，并逐项审查完整 32 个运行时补丁。没有把旧开发分支的其他 main 历史合入 release。编译依赖、模型实现、原生 kernel 和 release 自身的运行时结构继续使用 0.5.22。

本部署 profile 使用 `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`。官方 0.5.22 默认的 Rust tree 不执行 Python 的连续 device-prefix 防御和节点 mutation observer；本次不重编译 Worker Rust 扩展，而是在 P/D 环境中显式选择 Python tree，并在安装入口验证。全局 release 默认不变，CPU 开销和 GPU 吞吐需在目标负载上验收。

全部 Python 差异导出为一个补丁，以消除历史补丁的重复修改和旧版本上下文；Rust Gateway 差异单独导出。分支中的测试、说明、启动器、实验规划器和旧版回移工具一并保留。历史目录 `backport_v0518/` 仅供追溯和旧版部署使用，0.5.22 的入口为本目录。

## 旧优化提交的去向

以下均为已实现功能；新分支采用 release 的接口适配，不保留这些提交的原始父链。

| 来源提交 | 功能 | 0.5.22 处理 |
| --- | --- | --- |
| `1248b9086645` | Gateway 严格并发 permit、等待者上限、zero-refill 唤醒 | 迁移 Rust 中间件与 token bucket；permit 持有到流式响应结束 |
| `3741c269238a` | Router / Prefill / Decode 一致的启动器 | 保留命令构造、验证、dry-run 与测试 |
| `61286c463c28` | Decode 提前拒绝取消 Prefill、保留 HTTP 状态与 Retry-After | 按 release HTTP PD 双请求结构重写，不依赖 main 新增 decode gate |
| `07addc0c6fae` | DP rank admission、ACK、KV 容量账本 | 使用 release 参数组、负载快照和 typed KV ownership |
| `f5b8dde4fcb4` | L3 cost-aware 读取与重算成本模型 | 迁移 cost 模型、参数校验、决策与统计 |
| `f72308a6faa7` | Prefill 时间预算与对齐 | 保留已提交 token/时间预算及首请求进展约束，适配 release 调度器 |
| `b2cf2bea5e7f` | native 压测口径审计 | 保留请求结果、token、usage 与测量校验；未带入 release 不具备的无关 goodput 接口 |
| `67dddaff2570` | KV 实际释放后反馈 admission 容量 | 保留 release 的异步释放所有权，释放完成后发布容量 |
| `8fdf9851f9b1` | Prefill committed diagnostics | 在 release metrics reporter 中记录实际提交批次 |
| `bad1982d03d8` | L3 生命周期与终态决策诊断 | 结合现有存储 ACK、PP drain 和重试机制，避免重复计数 |
| `5b95df88f3fd` | 后续实验规划器 | 保留基于实际报告构造对照实验的工具 |
| `d4b237296c73` | Chat 压测口径审计 | 保留 Chat 使用量、流式与客户端结果校验 |
| `569ec6aecc3e` | 显式 Chat / GSP 规划 | 保留真实输入协议与模型 profile 的显式选择 |
| `ed565e76dd4a` | 旧版补丁安装与部署包 | 保留历史工具，复用经过测试的安装器；新增本 release 导出器 |

## 32 个运行时补丁的覆盖

精确的原补丁文件名、SHA-256、原始受影响文件、适配说明与状态见 [`legacy-patches.json`](legacy-patches.json)。这里按功能列出全部编号；编号有空缺是原输入的实际情况。

| 原补丁 | 功能 | 迁移处理 |
| --- | --- | --- |
| 0003 | Decode cache report / streaming usage 开关 | 保留显式 opt-in 环境变量，默认关闭 |
| 0004 | 推测解码 reasoning usage 在 EOS / stop / 输出上限处截断 | 按最终输出长度统计；finish 状态先于计数 |
| 0005 | 历史 tool arguments 非转义控制字符兼容 | 快速解析失败后使用受限 JSON fallback，保留 object 结构校验 |
| 0006 | JSON 请求开头 UTF-8 BOM | 保留 ASGI 分块请求处理 |
| 0007 | SSE 首包前返回真实 HTTP 错误 | 保留客户端可见状态，不将请求拒绝伪装为 200 |
| 0008 | 缓存命中数为 0 时仍报告 usage 字段 | **release 已覆盖**，不重复修改 |
| 0009、0010 | 内置 web_search、Chat ID / error / tool choice | 迁移 Chat / Responses 合约与原生工具过滤 |
| 0011 | Mooncake bootstrap 分阶段诊断 | 迁移日志和指标 |
| 0012 | 连续传输失败才隔离 session | 迁移计数及成功复位语义 |
| 0013 | Mooncake 后台线程存活与进度 | 兼容 release 角色切换线程 |
| 0014 | 并发清理期间保持本地 metadata 引用 | 适配当前 metadata 对象生命周期 |
| 0015 | host-only 祖先下 device prefix 连续性 | 迁移树路径的防御性约束 |
| 0016 | scheduler 子进程存活指标 | 区分父进程预启动 workers 与 controller，避免重复统计 |
| 0017 | 缺少 `seq_lens_cpu` 的 decode FPM | 在实际 FPM 构造路径提供 fallback |
| 0019 | 已知 context-length 错误映射 413 | 保留明确错误匹配与 HTTP 合约 |
| 0021 | 精确预取容量与 FIFO 重试 | 保留容量等待队列；与 release 的有限存储重试分开 |
| 0024、0025 | 请求 / 操作 / 传输和 SDK 批量 IO 指标 | 迁移并与 release ACK 所有权匹配 |
| 0027 | prefix 漂移与 host mutation 归因 | 迁移 admission observation 与树变更记录 |
| 0028 | device prefix 丢失后的有限 L3 recheck | 保留 release 原生重试上限，补回原观察指标 |
| 0029、0030 | Prefill 无批次诊断、Bootstrap FIFO head 重试 | 保留限流日志与 pending head 进展 |
| 0033 | Chat 流 created timestamp 恒定 | **release 已覆盖**，不重复修改 |
| 0034 | tool stream 仅初始 delta 发送 metadata | 只对 tool delta 使用 exclude-unset，保留普通流默认字段 |
| 0035 | PD handoff reasoning token 与 replay 去重 | **release 已覆盖**，不重复修改 |
| 0036 | 推测解码上下文预留 | 保留一或两个推测窗口，同时保留 release allocator 内存边界裁剪 |
| 0040 | DP admission、cost-aware L3、Prefill 时间预算 | 由相应定制功能迁移覆盖，不套用旧版大补丁 |
| 0041 | 旧版 Req.kv 兼容修复 | 改用 release typed ownership，保留完整 KV / Mamba / backup 所有权判断 |
| 0042 | 实际释放后反馈容量 | 由 `67dddaff2570` 的 release 迁移覆盖 |
| 0043 | Prefill diagnostics | 由 `8fdf9851f9b1` 的 release 迁移覆盖 |
| 0044 | L3 diagnostics | 由 `bad1982d03d8` 的 release 迁移覆盖 |

## 需要语义适配的接口

### DP admission 与释放

0.5.22 将参数拆分为多个 runtime bags。新增字段、解析、resolution hook、互斥校验及 controller 实例初始化全部使用 release 的结构；新增实际执行 `DataParallelController.__init__` 的 CPU 回归，覆盖启用与关闭两种路径。请求创建、session 请求、缺失 session、早拒绝及正常完成都必须绑定正确的账本生命周期。

释放反馈依据实际 KV 所有权，保留异步 Mooncake 释放与 backup 所有权，不能仅凭请求终止便归还容量。取消输出沿 release 的 Python IPC 路径发送；迁移不调用该版本未提供的 Rust `push_abort` 方法。DP admission 的已有约束继续校验，不擅自扩展到尚未验证的嵌入式 Rust、多节点或其他 speculative 组合。

### 缓存与调度

保留 release Unified Cache 的 ACK、PP optional-bool ticket、在途 copy / backup 生命周期。为 cost-aware 的 deadline 决策只补齐必要的小型 timer 与 operation 状态。容量不足时排队和存储读取失败重试具有不同含义，不能共用或绕过 release 重试预算。

Prefill 时间预算统计实际提交 batch 的成本；chunk 对齐、余量和首请求进展仍由调度约束共同决定。观察指标挂在当前 match、admission 和终止路径，避免把新旧回调同时保留而重复计数。

### API、推测解码与进程

HTTP 429 等 4xx/5xx 必须穿过 tokenizer、OpenAI 层和 Gateway。tool/web_search 兼容改动保留 release 新增协议功能。reasoning usage 以实际输出 token 为准，EOS 后被丢弃的已接受 draft 不进入计数。上下文预留和内存分配器容量裁剪同时生效。

0.5.22 的部分 attention-DP workers 由 Engine 父进程提前启动，watchdog 指标按实际进程归属安装；controller 本身不会被误计为一个 GPU scheduler。

## 部署产物边界

- Python 补丁只对固定官方源码字节应用，并验证最终源码哈希。
- 新包在用户提供的 RBG 上替换 P/D 镜像、补丁 ConfigMap 引用，并设置上述 Python tree 环境变量，附带校验条件和精确回滚；其他 env 保留。
- Rust Gateway 需要从同一分支单独构建 wheel 或二进制。Worker 启动脚本不能修改已编译的 Router。
- 旧版性能报告和校准参数作为输入保留，不构成 0.5.22 性能结论。
- 跨重试 admission deadline、DP8 自适应 EAGLE、跨请求 L3 合并、Decode prefix sharing 等路线图尚未实现，不能列为此次迁移成果。

测试与实际部署边界见 [`VALIDATION.md`](VALIDATION.md)。
