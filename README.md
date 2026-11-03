> **English version below · 中文完整上手指南见下方 👇**

# RAG Knowledge Base QA Agent

**An agentic RAG system: ask questions over your own documents in multi-turn conversation — the LLM decides when to search, and every answer comes with source evidence.**

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Embedding](https://img.shields.io/badge/Embedding-bge--small--zh--v1.5-FFD21E?logo=huggingface&logoColor=black)
![VectorDB](https://img.shields.io/badge/Vector%20Search-FAISS%20%2B%20BM25-5C3EE8)
![LLM](https://img.shields.io/badge/LLM-GLM--4--Flash%20·%20Function%20Calling-brightgreen)
![UI](https://img.shields.io/badge/UI-Gradio-orange?logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

Upload your PDFs or text files, build a persistent vector index, and chat in Chinese.
The LLM calls a `search_knowledge_base` tool via **function calling** whenever it needs evidence,
retrieval fuses **dense (FAISS) + sparse (BM25)** scores behind a **relevance gate**, and the answer
is generated **strictly from retrieved passages** — if nothing relevant exists, it says so instead of making things up.

<!-- TODO: 放一张 Gradio 界面截图，命名 docs/demo.png，然后取消下一行注释 -->
<!-- ![Demo](docs/demo.png) -->

## Why it exists

LLMs don't know your private documents. Fine-tuning is expensive, slow, and goes stale the moment a
document changes. RAG solves this at inference time: retrieve first, then generate. This repo is a
**minimal, fully readable implementation** of that idea — one file, ~400 lines, running entirely on CPU —
pushed one step further: the retrieval step is wrapped as a **tool** the LLM chooses to call, which makes
it a small but real **agent**, not a fixed pipeline.

## Highlights

- **Agentic retrieval (Function Calling)** — retrieval is exposed as a tool; GLM-4-Flash decides *whether*
  and *what* to search each turn (query rewrite included: pronouns like "他" get resolved into standalone
  queries), enabling genuine multi-turn follow-up questions
- **Hybrid retrieval (BM25 + dense)** — dense embeddings miss rare proper nouns and terse follow-ups;
  BM25 rescues them. Fusion: `0.7 × cosine + 0.3 × max-normalized BM25` over the union of both candidate sets
- **Relevance gating** — a candidate enters the generation stage only if `cosine ≥ 0.40` (strong semantic
  match) **or** `normalized BM25 ≥ 0.5` (strong lexical match). On a held-out sample, relevant questions
  scored **0.50–0.57** cosine while irrelevant ones scored **0.19–0.27** and are all blocked — refusal is
  enforced by retrieval, not by pleading in the prompt
- **Graceful degradation** — no API key? It automatically falls back to pure-retrieval mode, so the pipeline
  is still fully demonstrable offline
- **Persistent index** — FAISS index + chunks are saved to `kb_store/`; restarts don't re-index
- **Overlapping chunking** — 300-char windows with 50-char overlap, so key sentences aren't severed mid-thought
- **Local embeddings** — `bge-small-zh-v1.5` (512-dim, Chinese-optimised) runs offline on CPU; your documents
  never leave the machine during indexing

A real rescue case from testing: the follow-up *"那他量化时用的什么工具链？"* scores only **0.347** cosine
(below the gate — dense retrieval fails on terse, pronoun-laden queries) but **1.01 BM25** on the correct
chunk, so the gate's lexical path lets it through and the agent answers "RKNN" correctly instead of hallucinating.

## Quick start

```bash
conda create -n rag python=3.10 -y && conda activate rag
pip install -r requirements.txt

# Put your 智谱 API key in api_key.txt (git-ignored), or set ZHIPU_API_KEY
python app.py     # opens http://127.0.0.1:7860
```

Try it: upload `sample_knowledge.txt` → click **构建知识库** → ask `这个人掌握哪些深度学习框架？`.
Then follow up with `那他量化时用的什么工具链？` — the agent rewrites it into a standalone query, retrieves,
and answers `RKNN`. Then ask something absent (`北京今天天气如何？`) — the relevance gate blocks every
candidate and the agent refuses. **That refusal is the point of the project.**

## Architecture

```
Documents → read_file() → chunk_text(300/50) → SentenceTransformer → normalize → FAISS IndexFlatIP
                                                └→ jieba tokenize → BM25Okapi        (persisted: kb_store/)

User turn → GLM-4-Flash (function calling) ──decides──▶ search_knowledge_base(query)
                                                        ├─ dense: FAISS top-8
                                                        ├─ sparse: BM25 top-8
                                                        ├─ fuse: 0.7·cos + 0.3·bm25_norm
                                                        └─ gate: cos≥0.40 or bm25_norm≥0.5
            ◀── tool result (top-3 chunks or "nothing found") ──┘
Agent loop (≤3 rounds) → final answer + source evidence
```

Vectors are L2-normalised, so inner product == cosine similarity — no extra distance conversion needed.

## Tech decisions worth defending

| Decision | Alternative | Why I chose this |
|---|---|---|
| Function-calling agent loop | Fixed "retrieve→generate" pipeline | Lets the LLM rewrite terse follow-ups into standalone queries and re-search with different keywords — fixed pipelines answer follow-ups with whatever the raw pronoun query retrieves |
| Hybrid BM25 + dense | Dense only | Dense misses rare proper nouns & terse follow-ups (measured: 0.347 cosine on a query BM25 scored 1.01) |
| Relevance gate before generation | Only prompt-level "don't hallucinate" | Enforcement belongs in retrieval; irrelevant contexts never reach the LLM (irrelevant queries measured at 0.19–0.27 cosine, all blocked) |
| `bge-small-zh-v1.5` | OpenAI embeddings | Free, offline, and specifically tuned for Chinese |
| `IndexFlatIP` | HNSW / IVF | Exact search; at this scale brute force is faster and has zero tuning cost |
| 300-char chunks | Whole document | Whole-doc embedding dilutes the signal; too-small chunks lose context |
| `temperature=0.1` | 0.7 | RAG needs faithfulness, not creativity |
| Persist FAISS + chunks | Re-index on every start | Re-indexing wastes ~seconds-to-minutes per restart and hurts demo UX |

## Tuning

| Parameter | Location | Effect of increasing |
|---|---|---|
| `CHUNK_SIZE` | top of `app.py` | More context per hit, lower retrieval precision |
| `CHUNK_OVERLAP` | top of `app.py` | Fewer severed sentences, more redundancy |
| `TOP_K` | top of `app.py` | More evidence for the LLM, more noise too |
| `SIM_THRESHOLD` | top of `app.py` | Stricter semantic gate → fewer irrelevant contexts, more false refusals |
| `BM25_WEIGHT` | top of `app.py` | More weight on lexical matches, less on semantics |
| `temperature` | `call site` | Higher = more creative, less faithful — keep low for RAG |

## What I'd do next

- [x] Hybrid retrieval (BM25 + vector) to fix pure-semantic misses on rare proper nouns
- [x] Persistent index so documents don't need re-indexing on restart
- [x] Relevance gating before generation
- [ ] Reranker (bge-reranker-base) between retrieval and generation
- [ ] Automated eval set (QA pairs + hit@k) to track retrieval quality across changes
- [ ] Swap FAISS for **Qdrant / Weaviate** to test a production vector database

---

<sub>Built by <a href="https://github.com/lannawhite">@lannawhite</a> · AI undergraduate, Beibu Gulf University · Open to LLM application / Agent development internships</sub>

---
---

<!-- ===================== 以下是中文完整版 ===================== -->
# RAG 本地知识库问答 Agent — 快速上手指南

> 技术栈：Python + bge-small-zh 嵌入模型 + FAISS/BM25 混合检索 + GLM-4-Flash（免费，Function Calling）+ Gradio 多轮对话

---

## 一、这是什么

大模型不知道你的私有文档内容。本项目的思路：**把"检索知识库"封装成工具**，让大模型在对话中自主决定
"是否检索、检索什么"（Function Calling），检索采用**向量 + BM25 混合**并用**相关度门控**过滤，
最终只依据检索到的内容回答——检索不到就明确拒答，不编造。

```
用户提问（支持多轮追问）
   │
   ├─ LLM 判断 → 调用工具 search_knowledge_base(query)
   │                ├─ 向量检索（FAISS，语义相似）
   │                ├─ BM25 检索（词面匹配，兜住专有名词/口语化追问）
   │                ├─ 融合排序 → 相关度门控（不相关的片段不进入生成）
   │                └─ 返回 top-3 片段
   ├─ 大模型"开卷考试"，生成有依据的回答
   └─ 界面同步展示原文依据，可人工核对
```

---

## 二、环境准备（约 20 分钟）

```bash
# 1. 用你的 anaconda 建独立环境（Python 3.10）
conda create -n rag python=3.10 -y
conda activate rag

# 2. 装依赖（清华镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 让嵌入模型从国内镜像下载（Windows CMD/PowerShell）
set HF_ENDPOINT=https://hf-mirror.com        # CMD
$env:HF_ENDPOINT="https://hf-mirror.com"     # PowerShell
# 注意：这一步要在运行 python app.py 的同一个终端里设置

# 4. 申请免费大模型 API Key
#    打开 https://bigmodel.cn （智谱）→ 注册 → 右上角"API Keys"→ 创建一个
#    GLM-4-Flash 模型免费，学生够用
#    （备选：DeepSeek https://platform.deepseek.com 充 10 块钱能用很久）

# 5. 配置 Key（二选一）
#    方式 A（推荐，本项目默认）：在项目根目录下新建 api_key.txt，把 Key 粘贴进去保存即可
#    （api_key.txt 已被 .gitignore 忽略，不会被上传到 GitHub）
#    方式 B：设置环境变量 ZHIPU_API_KEY
set ZHIPU_API_KEY=你复制的Key      # CMD
$env:ZHIPU_API_KEY="你复制的Key"   # PowerShell

# 6. 准备嵌入模型（本地部署方式）
#    从 https://hf-mirror.com/BAAI/bge-small-zh-v1.5 下载整个模型文件夹
#    放到项目根目录的 bge-small-zh-v1.5/ 下（代码会优先用本地权重，没有则自动按模型名下载）
#    若下载的文件夹缺少 1_Pooling 子目录，请新建 bge-small-zh-v1.5\1_Pooling\config.json，
#    内容见文末附录。模型权重较大（约 100MB），已通过 .gitignore 排除，不入库。
```

## 三、运行（约 5 分钟）

```bash
python app.py
# 自动打开 http://127.0.0.1:7860
```

**验证流程**（用自带的示例文档练手）：
1. 上传 `sample_knowledge.txt` → 点「构建知识库」（会持久化到 `kb_store/`，重启程序无需重建）
2. 问：`这个人掌握哪些深度学习框架？` → 应答：PyTorch、TensorFlow/Keras
3. **多轮追问**：`那他量化时用的什么工具链？` → Agent 会把追问改写成独立查询再检索 → 应答：RKNN 工具链、FP16/INT8
4. 问一个知识库里没有的（如`北京今天天气如何`）→ 相关度门控拦截全部候选，应答"没有找到相关信息"——**这一步一定要演示，这是本项目防幻觉的核心卖点**
5. 换成你自己找的任何 PDF/TXT（论文、产品文档、课程笔记）再试

未配置 API Key 时自动降级为纯检索模式：不生成回答，直接返回最相关的原文片段。

## 四、参数怎么调

| 参数 | 位置 | 作用 | 调大的影响 |
|---|---|---|---|
| `CHUNK_SIZE` | app.py 顶部 | 文本块长度（300 字符） | 块更大→上下文更完整但检索精度下降 |
| `CHUNK_OVERLAP` | app.py 顶部 | 块间重叠（50 字符） | 防止关键句被切断，过大则冗余 |
| `TOP_K` | app.py 顶部 | 召回块数（3） | 提供给模型更多依据，但可能引入噪音 |
| `SIM_THRESHOLD` | app.py 顶部 | 语义门控阈值（0.40） | 更严格→无关内容更难进入生成，但误拒答变多 |
| `BM25_WEIGHT` | app.py 顶部 | 混合检索中 BM25 权重（0.3） | 词面匹配权重更大，语义权重更小 |
| `temperature` | call_llm 里 | 生成随机度（0.1） | 越低越严谨，RAG 场景要低 |

## 五、常见坑

- **模型不调用工具直接回答**：GLM-4-Flash 对 system prompt 措辞敏感，本项目已调出稳定写法
  （分步"工作流程"式指令）。换模型后若复发，参考 `SYSTEM_PROMPT` 的措辞重新调试
- **模型下载慢/失败**：确认 `HF_ENDPOINT` 在运行前已设置；或手动从 hf-mirror.com 下载 `BAAI/bge-small-zh-v1.5` 整个文件夹放到本地
- **zhipuai 报错**：`pip install zhipuai --upgrade`；Key 报错就检查环境变量是否真的设置（`echo %ZHIPU_API_KEY%`）
- **PDF 提取出乱码/空**：扫描版 PDF 没有文字层，换文字版 PDF 或 txt
- **faiss 装不上**：确认 python 3.10，用 `pip install faiss-cpu` 而不是 faiss

## 附录：手动补齐 1_Pooling/config.json

部分镜像源下载的模型包不含 `1_Pooling` 目录，会导致新版 sentence-transformers 报
`Pooling.__init__() missing 'embedding_dimension'`。新建 `bge-small-zh-v1.5\1_Pooling\config.json`：

```json
{
  "word_embedding_dimension": 512,
  "pooling_mode_cls_token": true,
  "pooling_mode_mean_tokens": false,
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false,
  "pooling_mode_weightedmean_tokens": false,
  "pooling_mode_lasttoken": false,
  "include_prompt": true
}
```
