# GLM-5.3 PD：0.5.22 镜像与启动补丁

本目录以官方 `release/v0.5.22` 为部署基线：

- 官方分支和 `v0.5.22` tag 的源码提交：`baa9befa381511cd4d56f83980951e5a45fffc6d`。
- 新特性分支：`feature/glm53-pd-v0.5.22-20261027`。
- 迁移来源：旧优化分支 `ed565e76dd4aed44ab3f4141fae3450dba3b0bc2` 的 14 个定制提交，以及 pd1 的完整 32 个运行时补丁所提供的功能。
- 迁移按 0.5.22 的接口重新适配。上游已有的实现保留，重复的回移层合并；逐项记录见 `MIGRATION.md` 和 `legacy-patches.json`。

Worker 运行时使用官方 0.5.22 镜像，在 `sglang serve` 前应用 Python 补丁。源码分支同时包含 Rust Gateway、压测客户端、测试、文档和部署工具。Rust Gateway 需要单独构建；Worker 补丁不能更新已编译的 Router。

**此迁移 profile 显式使用 Python 缓存树**：P/D 的环境变量设为 `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`。0.5.22 官方默认是 Rust tree core，而 0015 的连续 device prefix 修复和 0027 的节点 mutation 归因位于 Python tree core。生成器会设置此变量，启动预检会验证它；官方全局默认保持原样。这是完整保留既有功能的部署要求，其 CPU 开销与最终 GPU 吞吐需在新版本上实测。

## 生成部署包

在新分支的干净 checkout 中运行，生成工具需要 Python 3.11+、Git 和 PyYAML。`--source-ref` 必须是已提交且继承官方基线的 revision；存在未提交的 Python/Gateway 文件时拒绝导出，防止漏掉改动。

```bash
git fetch origin feature/glm53-pd-v0.5.22-20261027
git switch feature/glm53-pd-v0.5.22-20261027

python3 examples/runtime/glm53_pd/release_v0522/build_bundle.py \
  --source-dir . \
  --source-ref HEAD \
  --deployment /path/to/sglang-pd1-glm53.rbg.yaml \
  --image lmsysorg/sglang:v0.5.22 \
  --output-dir /path/to/pd1-v0522
```

`lmsysorg/sglang:v0.5.22` 是官方 release 公告列出的 NVIDIA CUDA 13 镜像。私有镜像或镜像 digest 可以通过 `--image` 显式指定；工具不会拉取镜像，也不会仅凭镜像名称认定其源码兼容。[官方 release 公告](https://github.com/sgl-project/sglang/releases/tag/v0.5.22)

不提供 `--deployment` 和 `--image` 时，可以只生成通用补丁和 ConfigMap：

```bash
python3 examples/runtime/glm53_pd/release_v0522/build_bundle.py \
  --source-dir . \
  --namespace sglang1 \
  --output-dir /path/to/v0522-patches
```

完整的集群 YAML 只由操作者的输入在本地生成，不写入公共仓库。生成器拒绝覆盖已有输出目录。

### 包内文件

| 文件 | 用途 |
| --- | --- |
| `patches/0001-glm53-pd-runtime.patch` | 相对准确的 0.5.22 基线导出的完整 Python 差异，含运行时和压测客户端 |
| `patches/stack.json` | 所有受影响源码与版本文件的前后 SHA-256，以及补丁哈希 |
| `patches/apply-stack.py` | 已验证的整栈安装器，先在临时副本重放，再安装通过哈希验证的文件 |
| `patches/check-source.py` | 检查 Python tree profile 及实际导入的 `sglang` 是否来自待打补丁的源码目录 |
| `patches/fix-patch.sh` | 启动入口；检查或安装失败即返回非零 |
| `configmap.yaml` | 内容寻址、不可变的完整补丁 ConfigMap |
| `gateway.patch` | 独立的 Rust Gateway 源码差异，供构建 Router 使用 |
| `migration-manifest.json` | 官方基线、导出 revision、源码文件数和补丁哈希 |
| `review.json` | 部署候选、输入哈希和实际变更范围 |
| `runtime-upgrade.patch.json` | 提供 RBG 时生成：替换 P/D 镜像、补丁 ConfigMap 引用，并显式设置 Python tree 环境变量 |
| `runtime-upgrade.rollback.json` | 带校验条件的精确逆操作，恢复原镜像、ConfigMap 和环境变量 |
| `runtime-upgrade.rbg.yaml`、`baseline.rbg.yaml` | 候选与输入 RBG，方便审阅 |
| `SHA256SUMS` | 导出文件的校验和 |

部署包将重叠的历史补丁压平为一个完整差异。不要在 0.5.22 镜像上先应用旧 0.5.18 的 29/32 补丁，再叠加此包。

## 镜像内预检与启动

官方镜像的 editable Python 源码通常位于 `/sgl-workspace/sglang`。应在计划使用的实际镜像内执行预检；替换过源码的私有镜像即使版本号同为 0.5.22，也可能被拒绝。

```bash
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python

python3 /sgl-workspace/sglang-fix/check-source.py \
  --source-dir /sgl-workspace/sglang

python3 /sgl-workspace/sglang-fix/apply-stack.py \
  --source-dir /sgl-workspace/sglang \
  --patch-dir /sgl-workspace/sglang-fix \
  --dry-run
```

预检只检查和重放补丁，不启动模型。实际 Worker 保持已有启动顺序：

```bash
set -euo pipefail
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
bash /sgl-workspace/sglang-fix/fix-patch.sh
exec sglang serve <本角色已经审阅的参数>
```

`python3` 和 `sglang` console script 必须使用同一 Python 环境；`check-source.py` 校验的是执行它的 Python。若自定义镜像含多个虚拟环境，应检查 `command -v python3`、`command -v sglang` 及该 console script 的解释器，确保服务使用同一份已打补丁源码。

安装器只接受完整官方源码状态或完整候选状态。`upstream` 与 `baseline` 在本包中都代表相同的官方 0.5.22 字节，全部差异作为 extension 一次安装。它执行补丁哈希检查、零 fuzz 重放、最终源码哈希检查、文件原子替换和异常回滚；重复安装为 no-op。未知、部分修改或混合版本的源码会被拒绝，没有强制跳过开关。

## 应用与回滚

以 `review.json` 中的 namespace、RBG 和 ConfigMap 名称为准。对当前 pd1 输入，操作形式为：

```bash
kubectl apply --server-side --field-manager=sglang-v0522-migration \
  -f /path/to/pd1-v0522/configmap.yaml

kubectl -n sglang1 patch rolebasedgroups.workloads.x-k8s.io sglang-pd1 \
  --type=json --patch-file=/path/to/pd1-v0522/runtime-upgrade.patch.json
```

JSON Patch 在修改前检查资源身份、角色位置、容器名称、完整启动命令、挂载、原镜像、完整 ConfigMap volume 定义及原环境变量。若原来没有 `env` 字段，会校验完整容器，防止覆盖导出之后新增的变量；已有其他环境变量逐项保留。实际输入有 env 列表，因此候选执行六个 replace：两镜像、两 ConfigMap 引用、两份仅调整 tree backend 的 env 列表。P/D 服务参数、资源配置、Router、Mooncake、Service 和副本数由输入 RBG 保留。升级后的 GPU 性能、模型输出、RDMA 传输和集群滚动就绪仍需用实际镜像验证；本次 CPU 测试不能代替这些验收。

回滚恢复原来的两个 Worker 镜像、补丁引用和环境变量，让替换容器从原镜像重新启动：

```bash
kubectl -n sglang1 patch rolebasedgroups.workloads.x-k8s.io sglang-pd1 \
  --type=json --patch-file=/path/to/pd1-v0522/runtime-upgrade.rollback.json
```

应保留原 ConfigMap。回滚不是在运行中的 Python 进程内反向改写源码。目录同时包含候选与回滚文件，应明确指定文件执行。

## 单独构建 Router

迁移保留严格并发 permit、队列上限、zero-refill 唤醒、流式生命周期和 Decode 提前拒绝时取消配对 Prefill 的 Rust 改动。0.5.22 的 HTTP PD 路由仍沿 release 的双请求结构运行，没有引入旧开发基线中额外的 decode gate。

若继续使用 `python3 -m sglang_router.launch_router`，从新分支构建 Python wheel。构建环境需要 Rust 1.90、C/C++ 工具链和 Protobuf compiler，并与 Router 镜像的 CPU 架构及系统 ABI 匹配：

```bash
cd sgl-model-gateway/bindings/python
python3 -m pip install 'maturin>=1,<2'
maturin build --release \
  --features vendored-openssl \
  --out /path/to/router-wheels
```

将明确选定的 wheel 放入单独的 Router 镜像。该 release 的包版本仍是 `sglang-router==0.3.2`，因此替换同版本 wheel 需要 `--force-reinstall`；下例的 `--no-deps` 要求镜像已具备 `pyproject.toml` 的依赖：

```bash
python3 -m pip install --force-reinstall --no-deps /path/to/exact-router-wheel.whl
```

只需要独立二进制时，可以在仓库根目录构建：

```bash
cargo build --release \
  --manifest-path sgl-model-gateway/Cargo.toml \
  --features vendored-openssl \
  --bin sgl-model-gateway
```

`gateway.patch` 也可应用于干净的同一官方基线后构建。这里给出构建入口；CPU 库和 HTTP 回归测试通过不等于已经产出适配线上 Router 镜像的 production wheel。Worker 的 RBG 升级候选没有自动替换 Router 镜像。

## 验证与后续调优

详细测试结果和实际运行限制记录在 `VALIDATION.md`。本次迁移不宣称有新的 GPU 性能收益，也没有执行集群部署。之前在 0.5.18 上测得的吞吐和接受长度不能直接当作 0.5.22 的验收结果。

迁移保持已有 cost-aware L3、Prefill 时间预算、DP admission、EAGLE 配置及诊断功能。后续路线图中的跨重试 admission deadline、DP8 自适应 EAGLE、跨请求 L3 合并、Decode prefix sharing 等尚未实现，不属于此次已完成迁移的功能。
