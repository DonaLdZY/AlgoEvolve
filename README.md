# MLEvolve-Alter

生成方案采用有限、可供其他系统调用的接口协议，详见
[`docs/solution_interface.md`](docs/solution_interface.md)。每个导出的 best/Top-K
方案都会包含机器可读的 `solution_manifest.json`。

MLEvolve-Alter 是 AutoDecision 的方案搜索与代码执行引擎。它读取任务说明、数据和 AutoRealize 生成的结构化上下文，通过多轮 draft、debug 和 improve 搜索可执行方案，并保存最佳代码、指标、模型或求解器 artifact、Top-K 候选和完整搜索记录。

项目支持传统机器学习、深度学习、时序预测、数学优化、组合决策和强化学习。对于优化与 RL，系统允许候选代码自由设计状态、动作、约束、策略和求解流程，但要求最终方案能通过统一 evaluator 得到可比较分数，并提供可复用入口。

## 输入与输出

主要输入：

- 数据目录或 AutoRealize 输出目录。
- `description.md`，或命令行中的简短 `goal`。
- 可选的 `realize_report/automl_context.md`。
- 带注释的 YAML 配置或 OmegaConf 点号覆盖参数。

主要输出：

- 搜索树和 `journal.json`。
- 每个候选节点的代码、执行输出、指标、LLM insight 和程序解析事实。
- `best_solution` 与 `top_solution`。
- 模型、策略、预处理器或求解器 artifact。
- 运行状态、资源用量、LLM token 和简略/详细日志。

## 搜索流程

```text
任务说明 + AutoRealize context + 数据
                  |
                  v
         初始 draft 生成与执行
                  |
                  v
      evaluator 解析、结果反馈、入树
                  |
                  v
     debug / improve / 新分支并行搜索
                  |
                  v
       最佳方案、Top-K 与交付产物
```

DeepSeek 的 `/beta` endpoint 是有意保留的：draft、improve、fusion 等生成代理会把末尾 `assistant_prefix` 作为 Chat Prefix Completion 发送，并设置 `prefix: true`。thinking 默认开启时不发送无效的采样参数，`reasoning_effort` 使用顶层字段；上下文缓存仍由 DeepSeek 根据稳定消息前缀自动命中。

Stepwise 将固定 task/evaluator/workflow 放在会话前缀，后续阶段只追加动态轮次。接近配置的 context headroom 时，较早动态轮次会由 LLM 压缩，最近轮次保留原文；压缩前的精确消息写入 `context_snapshots/`，模型可用 `REQUEST_CONTEXT_SNAPSHOT` 请求宿主回填一次原文。task contract、当前完整代码和最新错误位于受保护的基础上下文，不参与该动态压缩。

### Draft

- `agent.initial_drafts` 决定初始草稿总数，不因预测、优化或 RL 任务类型而改变。
- 每个父节点的第 1-2 个 sibling 使用 simple prompt；simple draft 使用单次生成，第 3-4 个使用 normal prompt，第 5 个起使用 complex prompt。该序号在并发提交时原子保留并写入断点状态。
- 达到 `agent.initial_drafts` 后，新 Draft 以 `agent.search.root_new_draft_probability` 与更深层 UCT 扩展竞争，不再机械填满 root。
- 前端可从 `pending_nodes.json` 显示尚在生成或执行的灰色节点。

### Debug 与 Improve

- 执行失败或代码缺陷进入 debug，输入当前错误、相关上下文和父节点代码。
- 已成功且有可比较分数的节点进入 improve，继续改进指标、稳健性或交付完整性。
- Result Review 一次返回 verdict、reason codes、debug hint、技术总结和前端 insight；只有极端或不确定分数才追加一次裁决调用。

### Prediction 与 Decision

- 问题族只有 Prediction 和 Decision；优化属于 Decision。启发式、局部/元启发式搜索、数学优化、RL 和混合方法是与问题族正交的方法族。
- Prediction Stepwise 为 Data & Validation、Model & Training、Inference & Artifact；Decision Stepwise 为 Problem & Evaluator、Decision Method、Solve/Rollout & Artifact，最后都由 MetaAgent 忠实合并。
- RL 可以从静态优化数据构造环境，定义 state、candidate/action、transition、reward、terminal 和合法动作 mask。
- 声称使用 RL 的节点应实际完成环境交互、policy 训练或配置化策略、rollout 和 artifact 保存，不能只保留未使用的 RL 类。
- 搜索树负责比较不同节点；单个 RL 节点不需要在内部强制生成另一个 baseline 分支。

## 节点与验证语义

- **生成后预检/修复**：运行前检查语法、危险调用和任务族要求的有限接口；修复后仍失败则不执行。
- **Result Review**：结合完整代码、原始输出、metric、运行事实和 artifact/interface 证据判断结果是否可信、是否有 bug。
- 进程异常、缺失/非有限 metric、evaluator mismatch、不可读输出和危险代码仍是确定性否决。artifact 与接口信息写入 manifest/notes，不再形成第三个 `delivery_ready` 资格层。

## 环境要求

- Conda、Miniconda 或 Miniforge
- Python 3.11 或 3.12，推荐 Python 3.12
- 可访问 OpenAI-compatible LLM API
- 足够的 CPU、内存和磁盘空间执行候选代码
- 可选 GPU、NPU 或其他加速卡

生成代码会根据任务动态导入数据处理、机器学习和优化库。基础与机器学习依赖由默认 requirements 提供，特殊领域库按需安装。

## Conda 环境安装

### 在 AutoDecision 主仓库中使用

```bash
cd AutoDecision
conda env create -f environment.yml
conda activate automl
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 独立安装 MLEvolve-Alter

```bash
git clone https://github.com/DonaLdZY/MLEvolve-Alter.git
cd MLEvolve-Alter
conda create -n mlevolve python=3.12 pip -y
conda activate mlevolve
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 会安装基础运行依赖和常用机器学习依赖。需要视觉、NLP、音频、地理、化学等较重的可选库时：

```bash
python -m pip install -r requirements_domain.txt
```

如需 GPU 版 PyTorch，请先使用 PyTorch 或硬件厂商官网给出的匹配命令安装，再安装其余依赖。项目不固定某个 CUDA、ROCm、XPU、MPS 或 Ascend 版本。

验证当前环境：

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

## 配置

默认配置文件是 [`config/config.yaml`](config/config.yaml)，包含中英文注释。配置优先级为：

```text
config/config.yaml
  < MLEVOLVE_CONFIG_PATH 指定的 YAML
  < run.py 的 key=value 点号覆盖
  < 服务请求中的任务临时配置
```

主要配置区：

| 配置区 | 作用 |
| --- | --- |
| `data_dir` / `desc_file` | 数据与任务说明路径 |
| `agent.steps` / `agent.time_limit` | 搜索步数和总时限 |
| `agent.initial_drafts` | 搜索开始时生成的初始草稿数 |
| `agent.code` / `agent.feedback` | 编码模型、评审模型和 API 参数 |
| `agent.output_language` | 模型生成的 plan/review/debug hint/summary/insight 统一语言：`english` 或 `chinese` |
| `agent.draft` | fast first draft、stepwise、review 和重试策略 |
| `agent.search` | 并行搜索、最大草稿数、debug 和 improve 分支预算 |
| `agent.memory_*` | 全局记忆与 Embedding 后端 |
| `resources` | 每任务 CPU、总内存和可见加速卡 |
| `runtime` | 继续任务、journal、状态与产物文件 |
| `logging` | 简略日志、详细日志、控制台和 LLM usage |

一个最小覆盖示例：

```yaml
data_dir: "/path/to/autorealize-output"
desc_file: "/path/to/autorealize-output/description.md"
exp_id: "demo"
exp_name: "demo"
log_dir: "./runs"
workspace_dir: "./runs"

agent:
  steps: 50
  time_limit: 10800
  initial_drafts: 3
  code:
    model: "deepseek-v4-pro"
    base_url: "https://api.deepseek.com/beta"
    api_key: ""
    max_tokens: null
  feedback:
    model: "deepseek-v4-pro"
    base_url: "https://api.deepseek.com/beta"
    api_key: ""
    max_tokens: null
  search:
    parallel_search_num: 4
    num_drafts: 8
    num_improves: 5
```

API Key 优先级：

1. YAML、点号覆盖或服务任务配置中的非空 `api_key`。
2. `MLEVOLVE_CODE_API_KEY` / `MLEVOLVE_FEEDBACK_API_KEY`。
3. `DEEPSEEK_API_KEY`。

Embedding 优先读取配置中的 key，其次读取 `MLEVOLVE_EMBEDDING_API_KEY` 或 `EMBEDDING_API_KEY`。resolved config 和快照不会保留明文密钥。

Linux / macOS 环境变量示例：

```bash
export DEEPSEEK_API_KEY="..."
export MLEVOLVE_CODE_API_KEY="..."
export MLEVOLVE_FEEDBACK_API_KEY="..."
export MLEVOLVE_EMBEDDING_API_KEY="..."
```

Windows PowerShell：

```powershell
$env:DEEPSEEK_API_KEY = "..."
$env:MLEVOLVE_CODE_API_KEY = "..."
$env:MLEVOLVE_FEEDBACK_API_KEY = "..."
$env:MLEVOLVE_EMBEDDING_API_KEY = "..."
```

## 直接运行

`run.py` 使用 OmegaConf 点号参数覆盖配置：

```bash
python run.py --help
```

最小示例：

```bash
python run.py \
  data_dir=/path/to/autorealize-output \
  desc_file=/path/to/autorealize-output/description.md \
  exp_id=demo \
  exp_name=demo \
  log_dir=./runs \
  workspace_dir=./runs \
  agent.steps=50 \
  agent.time_limit=10800 \
  agent.search.parallel_search_num=4
```

Windows PowerShell：

```powershell
python .\run.py `
  data_dir="D:\runs\demo\autorealize" `
  desc_file="D:\runs\demo\autorealize\description.md" `
  exp_id="demo" `
  exp_name="demo" `
  log_dir=".\runs" `
  workspace_dir=".\runs" `
  agent.steps=50 `
  agent.time_limit=10800
```

也可以把完整配置放在其他位置：

```bash
MLEVOLVE_CONFIG_PATH=/path/to/config.yaml python run.py
```

## 继续任务

继续任务必须指向原运行实际使用的日志目录和工作区目录：

```bash
python run.py \
  runtime.resume_run=true \
  log_dir=/path/to/existing/logs \
  workspace_dir=/path/to/existing/workspace \
  data_dir=/path/to/original/data \
  desc_file=/path/to/original/description.md
```

也可以设置：

```bash
MLEVOLVE_RESUME_RUN=1
```

继续时会恢复 journal、最佳节点和已预处理数据，但不会恢复已退出 Python 进程的堆内存、模型实例或缓存，因此继续后的内存占用通常低于停止前。

## 服务模式

```bash
python -m uvicorn service_api:app --host 127.0.0.1 --port 18103
```

常用接口：

- `GET /health`
- `GET /resources/inventory`
- `POST /jobs/start`
- `GET /jobs/{job_id}`
- `POST /jobs/stop`
- `POST /snapshot`

访问 `http://127.0.0.1:18103/docs` 查看 OpenAPI 文档。AutoDecision Gateway 通过服务接口启动独立搜索进程，不在 Gateway 进程内直接执行候选代码。

## 每任务资源限制

```yaml
resources:
  cpu_cores: 4
  memory_limit_gb: 8.0
  accelerator_mode: "selected"  # all | selected | none
  accelerator_device_ids: ["cuda:0"]
  monitor_interval_seconds: 0.5
```

- `cpu_cores` 是整个任务进程树共享的逻辑核心预算。
- `memory_limit_gb` 是控制器和全部子进程共享的总内存目标；`0` 表示不限制。
- `accelerator_mode` 控制任务看到全部、指定或不看到加速卡。

Windows 使用 CPU affinity 与 Job Object；Linux 优先使用 CPU affinity 与 cgroup v2；macOS 使用 worker/线程预算和进程资源限制。CUDA、ROCm、XPU 和 Ascend 通过对应可见性环境变量隔离；Apple MPS 可检测，但不能可靠地按进程隐藏。

## 输出产物

日志目录常见文件：

```text
logs/
|-- journal.json
|-- filtered_journal.json
|-- run_status.json
|-- pending_nodes.json
|-- config.yaml
|-- best_solution.py
|-- MLEvolve.log
|-- MLEvolve.verbose.log
|-- llm_usage.jsonl
|-- llm_usage_summary.json
|-- llm_usage_brief.json
`-- resource_usage.json
```

工作区常见文件：

```text
workspace/
|-- input/
|-- working/
|-- submission/
|-- best_solution/
|   |-- solution.py
|   |-- metric.txt
|   |-- node_id.txt
|   |-- solution_manifest.json
|   `-- <model-or-solver-artifacts>
`-- top_solution/
    |-- top1/
    |-- top2/
    `-- ...
```

生成代码使用固定协议：Prediction 提供 `train(data, artifact_dir)` 与 `predict(model_path, data)`；非 RL Decision 提供 `solve(model_path, data)`；RL/hybrid 提供 `train_policy(data, artifact_dir)` 与 `rollout(model_path, data)`。每个导出目录包含 `solution_manifest.json`。

## 日志与 token 统计

- `MLEvolve.log`：适合前端和快速统计的简略日志。
- `MLEvolve.verbose.log`：包含更完整的调试和第三方库日志。
- `dependency_installations.jsonl`：受控自动补库的逐次明细，包括缺失模块、安装包、当前 Python、命令、耗时、结果和 pip 输出尾部。
- `dependency_installations_summary.json`：本次 run 的补库汇总，可直接查看 `requirements_candidates` 并回填 requirements。

当 `exec.auto_install_missing_dependencies=true` 时，生成代码可把 import 根名和 PyPI distribution 绑定，例如 `# MLEVOLVE_PIP_INSTALL[sklearn]: pip install scikit-learn`。默认 `exec.dependency_install_policy=ai_declared`，AI 可选择任意语法合法的单个 PyPI distribution；脚本需要多个未知包时，每个声明必须绑定自己的 import 根名。运行时出现精确的 `ModuleNotFoundError` 后，MLEvolve 才会套用已有可信版本范围（如有）、使用当前解释器将包安装到任务隔离目录，并立即重跑同一节点；没有声明时才使用 `dependency_import_map` 兜底。AutoDecision 将目录固定为 `runs/<task>/automl/python_packages`，并只给该任务脚本追加 `PYTHONPATH`，不会修改基础 Conda/系统环境。严格部署可把策略切换为 `allowlist`。同一包每次任务只尝试一次，生成代码直接执行 pip/conda/shell 安装仍会被拒绝。
- `llm_usage.jsonl`：逐调用 token、reasoning token、缓存、响应模型和后端 fingerprint。
- `llm_usage_summary.json`：按模型、阶段和调用类型汇总。
- `llm_usage_brief.json`：供前端和成本分析使用的精简汇总，DeepSeek V4 按官方美元单价估算。

## 测试

```bash
conda activate mlevolve
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

在 AutoDecision 根环境中：

```bash
conda activate automl
python -m pytest core/MLEvolve-Alter/tests -q
```

默认单元测试不应调用真实 LLM、长时间搜索或 GPU 训练。

## 常见问题

### Draft 阶段很久没有节点

第一个节点可能仍在等待 LLM 输出、代码 review、请求重试或执行。检查 `pending_nodes.json`、`MLEvolve.log` 和 `llm_usage.jsonl`。启用 `agent.draft.fast_first_draft` 可以缩短首节点等待时间。

### 代码读取了不存在的列或 sheet

确认 `data_dir` 指向完整 AutoRealize 输出目录，而不是只复制了 `description.md`。检查是否存在 `realize_report/automl_context.md`，并确认其中的 Exact Source Schema Contract 与实际数据一致。

### 节点有分数但没有 submission 或 artifact

Result Review 接受的有限分数可以进入搜索和 best/Top-K；缺失 artifact 或配置输出会保留为证据 warning，并写入 solution manifest/metric 元数据，而不是再经过第三层交付资格评审。

### GlobalMemory 保存返回 404

通常表示 Embedding 服务地址、模型或 API Key 不可用。关闭 Embedding memory 不应阻止本地 journal 搜索；需要全局语义记忆时，请检查 Embedding 配置和服务兼容性。

### 达到时限是否算失败

不是。步数用尽和时间预算耗尽都属于正常终止条件。系统应保存已有最佳结果、Top-K、journal 和运行状态，并允许继续任务或生成报告。

### 继续任务后内存下降

继续任务恢复的是持久化状态和工作区，不是旧进程的瞬时内存。旧 worker 退出后，新的搜索进程只加载继续运行所需的状态。

## 上游声明与许可证

MLEvolve-Alter 基于 MLEvolve、AutoMLGen 及相关 agentic machine-learning engineering 工作演进。使用或发布前，请确认本仓库与所有上游项目的许可证、署名和引用要求。

- MLEvolve Project Page: <https://internscience.github.io/MLEvolve/>
- AutoMLGen: <https://arxiv.org/abs/2510.08511>
- InternAgent 1.5: <https://arxiv.org/abs/2602.08990>

在仓库加入明确许可证前，不应视为已经授权自由使用、修改或再分发。
