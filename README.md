# 🏛️ Loco Council Agent

> 这是一个人类做设计，ai负责coding（占比95%），人类review（覆盖90%）兜底的项目。
>
> 面向低算力平台（比如笔记本），但是内存至少需要16G。
>
> 本意是想做成一个面向网络小说，人物传记，历史事件等长文本的小型分析工具，希望它能有一点意思，有一些意义。后来因为一个图片表格识别的任务而意外启动。
>
> 后面优先提高阅读理解方面的体验，在OCR方向的更强功能会更新较慢。
>
> 🔭 **技术观察**：Jev 类决策模型值得关注——本项目的 LLM 打分环节与路线图中的
> "多级 LLM 路由"，都是它的潜在落点。

目前只支持 TXT 和 PDF 两种格式：

- **PDF**：目前写死走图片式表格数据处理逻辑，全程 OCR + 后续复杂处理（每页过 LLM 分块 + tool call），消耗一定 token。
- **TXT**：索引阶段仅在确定章节名格式时可能调 LLM 确定 regex；分块完全不依赖 LLM。

> 目前索引一个3m的txt需要1~2个小时，把embedding_batch_size调整为64可以快很多，但是16G内存的机器会有爆内存的风险。

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Source%20Available-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-178%20passing-brightgreen.svg)](tests/)

![screenshot](docs/screenshot.png)

---

## 💡 项目简介

扫描件PDF本质是图片——没有文字层、没有结构标记、表格连边框都没有。

Loco Council Agent 专为此场景设计：**从 OCR 到回答生成的完整闭环**，核心解决三个问题：

- **无边框表格识别**：PaddleOCR + 坐标重建，把散落的文字重新拼回表格
- **语义级分块**：LLM 理解表格结构后再切块，而不是机械地按字符数截断
- **溯源准确性**：每个回答都附来源引用（文档名 + 页码 + 块编号），拒绝编造

### 总体架构

```
┌─────────────────────────────────────────────────┐
│                Streamlit 前端                      │
│         搜索问答  │  文档管理  │  会话历史           │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│              Controller 调度层                     │
│   SearchController（搜索编排）                      │
│   DocController（索引编排 + 文档管理）               │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│              Pipeline 业务内核                      │
│                                                    │
│  索引管线: PDF → OCR → 表格检测 → LLM分块 → 入库    │
│  检索管线: Query → 混合检索 → 重排序 → LLM打分 → 回答 │
│                                                    │
├──────────┬──────────┬──────────┬─────────────────┤
│ Indexing │Retrieval │   LLM    │    Storage      │
│  OCR+切块 │ 检索+排序 │ 通信代理  │ SQLite+LanceDB │
└──────────┴──────────┴──────────┴─────────────────┘
```

依赖方向严格单向：`UI → Controller → Pipeline → 子模块`

---

## ⚙️ 两条核心管线

### 索引管线：PDF → Chunks

```
PDF 文件
  │
  ▼
OCR 引擎 ──────── PaddleOCR + 坐标重建，逐页输出 Markdown
  │                · 3种提取策略可切换（PaddleOCR + 坐标重建 的效果可堪一用）
  │                · 图像预处理（灰度化/降噪/倾斜校正）
  │                · 无边框表格坐标重建
  ▼
表格 + 表头分析 ── 识别表格区域，提取表头语义
  │
  ▼
LLM 语义分块 ──── DeepSeek 理解表格结构后切块
  │                · 支持跨页表格（segments 数组表达）
  │                · 行覆盖校验：漏行→驳回重做→专项修补
  │                · 三层格式约束：tools → response_format → 兜底解析
  ▼
向量嵌入 ──────── BGE-M3，1024维，分批编码
  │
  ▼
双后端入库 ────── SQLite（元数据）+ LanceDB（向量 + BM25全文检索）
```

### 检索管线：Query → Answer

采用 **4 层漏斗**，逐层收窄，兼顾速度与精度：

```
用户查询
  │
  ▼
第1层 混合检索 ──── 向量 + BM25（RRF融合）→ Top 42
  │                候选为 0？→ 告知用户，可选纯LLM回答或放弃
  ▼
第2层 CrossEncoder ─ BGE-Reranker 本地模型 → 收窄到 Top 15
  │                快、免费、稳定
  ▼
第3层 LLM 分级收网 ─ DeepSeek 对每个候选打分（0-10分）
  │                ≥8过半→全取≥8 │ ≥7过半→全取≥7
  │                ≥6过半→全取≥6 │ 兜底→全取≥5
  │                全部<5？→ 低置信度提示，用户决定是否继续
  ▼
第4层 后处理 ────── 间隙填充 + 邻居扩展，补全叙事上下文
  │                受 gap_fill_token_limit 硬上限保护
  ▼
RAG 生成回答 ───── 拼接上下文 + 用户查询 → LLM 生成
                   三层容量保护：gap(20K) / context(60K) / history(5轮)
                   附来源引用（文档名 + 页码 + chunk编号 + 得分）
```

**为什么四层？** 向量+BM25负责广撒网（互补），CrossEncoder负责快速收窄（免费），LLM负责精准判断（理解复杂意图），间隙填充负责补叙事线——四层各司其职，不可合并。

---

## 🧱 分层设计

### 双后端存储

```
         DocManager（业务层唯一入口）
        ┌─────────┴─────────┐
        │                   │
  _DocMetaStore        _ChunkStore
  (SQLite)              (LanceDB)
        │                   │
  files + chunks        向量索引 + FTS
  (元数据)              (相似度 + 全文)

  HistoryStore          retriever.search()
  (会话历史)             (纯读，直调LanceDB)
```

- **SQLite** 管理元数据（文档列表、chunk归属、启用状态）
- **LanceDB** 管理向量（相似度搜索 + BM25 全文检索）
- 读写分离：DocManager 管写，retriever 管读，独立演进

### Controller 调度层

前端与后端之间引入 Controller 层，作为服务编排的唯一集成点：

```
app.py 启动层组装单例 → 依赖注入 Controller → Controller 编排 Pipeline + HistoryStore
```

Pipeline 和 HistoryStore 互不知情。Controller 是唯一知道"一次搜索需要调哪些服务"的地方。前端只负责渲染，若未来换第二个前端（CLI/API），Controller 零改动复用。

### 双引擎分块

| | FinancialTableChunker | TextChunker |
|---|---|---|
| **输入** | PDF OCR 页列表 | TXT 全文 |
| **策略** | LLM 识别表格 + tool calling | 规则先行(零token) → LLM 兜底判断章节名 |
| **LLM调用** | 每批3页一次 | 至多2次，判断章节名 |
| **降级** | 行覆盖校验→两级回退 | 段落贪心聚合（段落绝不切开） |
| **适用** | 财报/审计报告 | 小说/电子书/长文 |

---

## 📐 设计原则

1. **溯源准确性优先** — 每个回答附来源引用，检索为空时明确告知而非编造
2. **业务错误用返回值，破坏性意外用异常** — OCR失败是业务分支，网络中断是意外，处理方式不同
3. **LLM客户端只做通信代理** — 重试、分批、收发，不做业务决策
4. **规则先行，LLM兜底** — 零成本的规则匹配覆盖90%场景，LLM处理剩余10%
5. **段落绝不从中切开** — 语义完整性 > 尺寸均匀性
6. **三个硬上限保护LLM上下文** — gap(20K) / context(60K) / history(5轮)
7. **本地模型预装是强制步骤** — 数 GB 磁盘与下载流量必须由用户知情后主动承担，不在启动时静默下载；主程序启动即校验模型是否就位，未就绪则暂停全部功能

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Windows / Linux / macOS
- 建议 16GB+ 内存（BGE-M3 启动时加载约 2GB；BGE-Reranker 与 OCR 模型按需加载）
- 约 4GB 可用磁盘空间（存放本地模型）

### 安装

```bash
# 克隆项目
git clone https://github.com/GFBinFace/loco-council-agent.git
cd loco-council-agent

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
# 如需把模型装到非系统盘，同时按注释填好 HF_HOME / PADDLE_PDX_CACHE_HOME
```

### 预下载模型（必需）

首次运行主程序**之前必须**完成这一步。模型体积大（约 3.2GB 文件，加上等量下载流量），
需要你明确知情后主动执行——而不是在启动时悄悄下载。

```bash
python scripts/download_models.py
```

脚本会先列出将下载的模型、哪些已就位、预计开销，**确认后**才开始；可重复执行，
已下载的模型会跳过。`--yes` 可跳过确认（供非交互环境使用）。

包含 BGE-M3（~2GB）、BGE-Reranker（~1GB）、PP-OCRv5 检测与识别模型（~200MB）。

**模型装在哪**：由 `.env` 中的两个环境变量决定，未设置时落在 C 盘用户目录：

| 变量 | 管哪些模型 | 默认值 |
|---|---|---|
| `HF_HOME` | BGE-M3、BGE-Reranker | `%USERPROFILE%\.cache\huggingface` |
| `PADDLE_PDX_CACHE_HOME` | PP-OCRv5 检测与识别 | `%USERPROFILE%\.paddlex` |

建议在 `.env` 中指向空间充足的非系统盘（目录不存在会自动创建）。

未完成预装时，主程序启动会**暂停全部功能**并提示你执行上面的脚本——这是刻意的：
宁可明确拒绝，也不让数 GB 下载在用户不知情时发生。

### 启动

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`，左侧上传PDF索引，右侧搜索问答。

### 常见问题

**启动时提示"本地模型未就绪"**  
说明还没完成模型预装——这一步是必需的，见上方 [预下载模型](#预下载模型必需)。
执行 `python scripts/download_models.py` 即可。
若脚本报「路径无法创建」，检查 `.env` 中 `HF_HOME` / `PADDLE_PDX_CACHE_HOME`
所指的分区是否存在。

**模型下载失败（国内网络）**  
在 `.env` 中设置 `HF_ENDPOINT=https://hf-mirror.com` 使用 HuggingFace 镜像。

**启动后内存不足**  
将 `config.py` 中 `embedding_batch_size` 从 16 调小（如 8），或关闭其他应用释放内存。16GB 机器建议保持默认值。

**DeepSeek API 返回 401**  
检查 `.env` 中 `DEEPSEEK_API_KEY` 是否正确填写，确认 API 账户余额充足。

---

## 🛠️ 技术栈

| 层级 | 选型 | 说明 |
|------|------|------|
| OCR | PaddleOCR + PaddleX 3.0 | 中文识别 + 无边框表格 |
| 嵌入模型 | BAAI/bge-m3 | 1024维，8192 tokens |
| 向量数据库 | LanceDB 0.16 | 嵌入式，零运维 |
| 重排序 | BAAI/bge-reranker-base | 本地运行，免费 |
| 元数据 | SQLite | 标准库，零配置 |
| LLM | DeepSeek (OpenAI兼容协议) | `deepseek-chat` |
| 前端 | Streamlit 1.35+ | 纯Python，快速迭代 |
| 代码质量 | ruff + mypy 严格模式 | 行宽88 |

---

## 📁 项目结构

```
loco-council-agent/
  app.py              ← Streamlit 入口
  config.py           ← 全局配置（唯一配置源）
  controllers/        ← 服务调度层
  services/
    pipeline.py       ← 核心编排器
    indexing/         ← OCR + 表格检测 + 分块
    retrieval/        ← 向量化 + 混合检索 + 重排序
    llm/              ← LLM客户端 + tools + prompts
  storage/            ← DocManager + HistoryStore
  _types/             ← 数据契约
  utils/              ← 通用工具
  ui/                 ← 前端组件
  scripts/            ← CLI 脚本 + 模型下载
  tests/              ← 15个文件，178条用例
  data/               ← 运行时数据
```

---

## ✅ V0.4版开发状态

| 模块 | 状态 |
|------|------|
| OCR 引擎 + 无边框表格重建 | ✅ |
| LLM 语义分块（PDF + TXT 双引擎） | ✅ |
| 向量嵌入 + 混合检索 | ✅ |
| CrossEncoder + LLM 分级收网 | ✅ |
| RAG 生成回答 + 溯源引用 | ✅ |
| 双后端持久化 | ✅ |
| Controller 调度层 | ✅ |
| 知识库概览 + 操作历史面板 | ✅ |
| 历史会话管理 | ✅ |
| 上下文容量保护 | ✅ |
| 端到端测试 | ✅ |

---

## 🗺️ 路线图

### 近期

- [x] **补全端到端测试** — TXT 章节识别、OCR 缓存、错误边界场景。
- [ ] **极简配置** — 抽取最少必要配置项，配好 API Key 即可运行，其余参数走默认值。高级用户可按需深入各环节细调。
- [ ] **开机自检** — 启动时检查 SQLite、LanceDB、本地模型文件、LLM API 连通性等基础配置是否就绪，把故障定位从运行时异常提前到启动阶段。
- [ ] **批量索引** — 一次选择多个 PDF/TXT 文件执行索引（MD5 去重已就绪）。
- [ ] **结构化索引** — 为文本文件的每个章节生成分级摘要嵌入（章级 + chunk 级），以及统计数据（关键词/关键符号密度），增强检索召回与 LLM 上下文感知能力。
- [ ] **CUDA 支持下的 OCR 处理** — 自动检测 CUDA 是否可用，若支持则使用 GPU 加速配置（新增独立参数），利用 GPU 大幅缩短 OCR 耗时；否则回退到 CPU 模式并给出预估处理时间。
- [ ] **PaddleOCR-VL-1.6 GGUF 量化版测试** — 在已有 GPU 上测试 PaddleOCR-VL-1.6 的 GGUF Q4 量化版（同生态 VLM 方案），对比当前传统管线的准确率和表格重建效果，评估升级可行性。

### 中期

- [ ] **章节地图增强** — 这是一个针对pdf的功能，当前默认关闭。已实现 rule_only 规则匹配，但尚未测试。最终目标是让 pdf 分块也像 txt 一样携带章节信息。
- [ ] **SearchResult 状态细化** — 增加 `user_abort` 等专用状态，替代当前统一使用 `success` 标记取消路径。
- [ ] **英文提示词支持** — 某些环节使用英文 prompt 配合特定模型可能获得更好效果（与 LLM 训练语料分布有关）。预留切换能力，遇具体场景再设计实现。
- [ ] **索引过期软提醒** — 在知识库概览面板标记索引天数（如"已索引 90 天"），提醒用户关注可能已过时的文档，不强制重新索引。
- [ ] **CLI / API 模式** — 利用 Controller 层的零改动复用能力，提供不依赖 Streamlit 的命令行和 HTTP API 入口。
- [ ] **多级 LLM 路由** — 复杂提问的自动化拆解与分级调度，成本可控，并提供具有观赏性的处理过程。
- [ ] **探索 PaddleOCR 的 INT8 模型量化** — 收集 PDF 数据样本作为校准数据集，尝试对 PaddleOCR 的检测和识别模型做训练后量化（PTQ），验证在低算力平台上的推理加速效果和坐标精度影响。

### 远期

- [ ] **问答对管理** — 支持永久删除（落库）和临时禁用（前端内存）具体问答的功能，精细控制历史上下文组装。
- [ ] **对话历史 JSONL 迁移** — 对话历史数据从 SQLite JSON 字段迁至 `data/sessions/{id}.jsonl`（append-only 行存储），支持部分加载和并发追加。
- [ ] **体验模式（项目的 B 面）** — 当前实现的是"研究模式"：基于大块材料和多轮问答，提炼关于某个虚拟故事、历史环境或人物的核心数据，进行结构化组织和存储。计划增加"体验模式"：让用户通过对话，亲身体验身处某个历史环境中的经历，或与某个真实/虚拟人物进行交谈。核心挑战在于对象模型与多维特征列表的设计，可能需要参考社科、游戏设计等跨领域的研究成果。
- [ ] **提示词组合系统** — 参考 AutoGPT 的 pipeline + component 模式，提示词模块化、可堆叠组合，由体验模式驱动。
- [ ] **上下文管理重构** — 参考 Hermes 的上下文管理思路，但其设计偏重，计划大幅精简后定向增强。同样由体验模式的长对话场景驱动。
- [ ] **索引过期硬删除** — 在软提醒基础上，支持按策略自动标记或清理过期索引数据。
- [ ] **MinerU OCR 后端** — 可选 OCR 扩展，将 MinerU 包装为独立网络服务供主项目调用，替代 PaddleOCR 处理复杂排版场景。
- [ ] **后台数据管理工具** — 提供查看和管理所有数据的统一入口（SQLite + LanceDB + 文件系统）。
- [ ] **建模数据兼容功能** — 环境/人物建模数据在各版本间的安全迁移与转换，由体验模式驱动。
- [ ] **后台数据校对功能** — 主动检测知识库数据完整性，防止数据破损静默影响检索结果。
- [ ] **生图功能** — 基于环境/人物建模数据，配合生图 LLM 的提示词设计，提供便捷的角色与场景图像生成体验。

---

## 📄 License

非商用自由使用，商用需授权 — 详见 [LICENSE](LICENSE)
