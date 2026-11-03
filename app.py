# -*- coding: utf-8 -*-
"""
RAG 本地知识库问答 Agent
================================================
完整链路：
  上传文档 → 文本切分 → 向量化(Embedding) → FAISS + BM25 混合检索
  → LLM Function Calling 自主决策是否检索 → 基于检索结果生成回答
  → Gradio 多轮对话界面（附原文依据，防幻觉）

核心特性：
  1. Agent 化：知识库检索封装为工具，GLM-4-Flash 通过 Function Calling
     自主决定"是否检索、检索什么"，支持多轮对话追问
  2. 混合检索：BM25(词面匹配) + 向量(语义匹配) 融合排序，缓解纯语义
     检索对专有名词/罕见词的漏召
  3. 相似度门控：检索得分低于阈值的片段不进入生成环节，从检索层抑制幻觉
  4. 索引持久化：构建一次知识库，重启后无需重新索引
  5. 优雅降级：未配置 API Key 时自动退化为纯检索模式

运行方式：
  1. 设置智谱 API Key（见 README.md，不设置也能用，退化为纯检索模式）
  2. python app.py
  3. 浏览器打开 http://127.0.0.1:7860
"""
import json
import os

import faiss
import gradio as gr
import jieba
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ============ 1. 全局配置 ============
def _load_api_key() -> str:
    """优先读环境变量 ZHIPU_API_KEY；没有则读本地 api_key.txt（该文件已被 .gitignore 忽略，不会上传）"""
    key = os.environ.get("ZHIPU_API_KEY", "")
    if not key:
        key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")
        if os.path.exists(key_file):
            with open(key_file, "r", encoding="utf-8") as f:
                key = f.read().strip()
    return key

API_KEY = _load_api_key()
LLM_MODEL = "glm-4-flash"       # 智谱免费模型
CHUNK_SIZE = 300                # 每个文本块最大字符数
CHUNK_OVERLAP = 50              # 相邻块重叠字符数（保证语义不断裂）
TOP_K = 3                       # 融合排序后最终交给 LLM 的片段数
CANDIDATE_K = 8                 # 每路检索（向量 / BM25）先各召回的候选数
SIM_THRESHOLD = 0.40            # 向量余弦相似度门控：低于该值的候选不进入生成环节
BM25_WEIGHT = 0.3               # 混合得分 = (1-BM25_WEIGHT)*向量相似度 + BM25_WEIGHT*BM25归一化分
VECTOR_WEIGHT = 1.0 - BM25_WEIGHT
MAX_TOOL_ROUNDS = 3             # Agent 单次回答中最多允许的工具调用轮数

# 嵌入模型：优先用仓库内本地权重（离线可用），没有则按模型名自动下载
_LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bge-small-zh-v1.5")
MODEL_SOURCE = _LOCAL_MODEL_DIR if os.path.isdir(_LOCAL_MODEL_DIR) else "BAAI/bge-small-zh-v1.5"
print(f"正在加载嵌入模型 {MODEL_SOURCE} ...（首次运行需要下载）")
embedder = SentenceTransformer(MODEL_SOURCE)

# 知识库持久化目录
STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb_store")
INDEX_PATH = os.path.join(STORE_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(STORE_DIR, "chunks.json")

# 全局知识库状态
chunks = []          # 所有文本块（字符串列表）
index = None         # FAISS 向量索引
bm25 = None          # BM25 稀疏检索器


# ============ 2. 文档读取与切分 ============
def read_file(path: str) -> str:
    """读取 txt / md / pdf 文件为纯文本"""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    # txt / md 统一按 utf-8 读
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str) -> list:
    """把长文本切成带重叠的小块——RAG 最核心的预处理步骤"""
    text = text.strip()
    result = []
    start = 0
    while start < len(text):
        piece = text[start: start + CHUNK_SIZE]
        if len(piece) > 30:  # 过滤太短的碎片
            result.append(piece)
        start += CHUNK_SIZE - CHUNK_OVERLAP  # 步进 = 块长 - 重叠
    return result


def _tokenize(text: str) -> list:
    """jieba 搜索引擎模式分词，供 BM25 使用"""
    return list(jieba.cut_for_search(text))


# ============ 3. 建向量索引 + BM25 + 持久化 ============
def _rebuild_bm25() -> None:
    """根据当前 chunks 重建 BM25 索引（BM25 体积小，从文本重建代价可忽略）"""
    global bm25
    if chunks:
        bm25 = BM25Okapi([_tokenize(c) for c in chunks])
    else:
        bm25 = None


def save_store() -> None:
    """持久化：FAISS 索引 + 文本块，重启后无需重新索引"""
    os.makedirs(STORE_DIR, exist_ok=True)
    if index is not None:
        faiss.write_index(index, INDEX_PATH)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)


def load_store() -> str:
    """启动时加载已持久化的知识库（若有）"""
    global index, chunks
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
        return ""
    try:
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        _rebuild_bm25()
        return f"已从本地加载知识库：{len(chunks)} 个文本块（无需重新索引）"
    except Exception:
        return ""


def clear_store() -> str:
    """清空知识库（内存 + 磁盘）"""
    global index, chunks, bm25
    index, chunks, bm25 = None, [], None
    for p in (INDEX_PATH, CHUNKS_PATH):
        if os.path.exists(p):
            os.remove(p)
    return "知识库已清空"


def build_index(files):
    """上传文件后重建知识库：切分 → 向量化 → FAISS + BM25 → 持久化"""
    global chunks, index
    if not files:
        return "请先上传至少一个文档（支持 txt / md / pdf）"
    chunks = []
    for f in files:
        try:
            text = read_file(f.name)
            chunks.extend(chunk_text(text))
        except Exception as e:
            return f"读取 {os.path.basename(f.name)} 失败：{e}"

    # 向量化并归一化（归一化后内积 = 余弦相似度）
    vectors = embedder.encode(chunks, normalize_embeddings=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)          # 内积索引（精确检索，小库够用）
    index.add(np.array(vectors, dtype="float32"))
    _rebuild_bm25()
    save_store()
    return f"知识库构建完成：共 {len(chunks)} 个文本块（来自 {len(files)} 个文件），已持久化到 kb_store/"


# ============ 4. 混合检索（向量 + BM25 融合 + 相似度门控） ============
def retrieve(question: str) -> list:
    """
    混合检索：两路召回 → 分数融合 → 门控过滤
    门控规则：向量相似度 >= SIM_THRESHOLD（语义强匹配）
              或 BM25 归一化分 >= 0.5（词面强匹配，兜住口语化追问/专有名词）
    返回 [(chunk, fused_score, vec_score)]，按融合分降序，最多 TOP_K 条
    """
    if index is None or not chunks or bm25 is None:
        return []

    # ---- 第一路：向量语义检索 ----
    q_vec = embedder.encode([question], normalize_embeddings=True)
    vec_scores, vec_ids = index.search(np.array(q_vec, dtype="float32"), min(CANDIDATE_K, len(chunks)))

    # ---- 第二路：BM25 词面检索 ----
    bm_scores = bm25.get_scores(_tokenize(question))
    bm_ids = np.argsort(bm_scores)[::-1][:CANDIDATE_K]

    # ---- 融合：取两路并集，BM25 分在候选集内做 max 归一化后加权 ----
    cand = {}
    for i, s in zip(vec_ids[0], vec_scores[0]):
        if i >= 0:
            cand[int(i)] = [float(s), 0.0]
    for i in bm_ids:
        if bm_scores[i] > 0:
            cand.setdefault(int(i), [0.0, float(bm_scores[i])])[1] = float(bm_scores[i])
    if not cand:
        return []
    bm_max = max(v[1] for v in cand.values()) or 1.0
    scored = [
        (chunks[i], VECTOR_WEIGHT * v[0] + BM25_WEIGHT * v[1] / bm_max, v[0], v[1] / bm_max)
        for i, v in cand.items()
    ]

    # ---- 门控：语义强匹配 或 词面强匹配，二者其一才进入生成环节 ----
    BM25_GATE = 0.5
    scored = [t for t in scored if t[2] >= SIM_THRESHOLD or t[3] >= BM25_GATE]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(c, s, v) for c, s, v, _ in scored[:TOP_K]]


# ============ 5. Agent：Function Calling 工具调用 + 多轮对话 ============
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "在用户上传的本地文档知识库中检索与查询相关的内容。"
                           "每一轮用户提问都必须先调用本工具检索后再回答，追问也不例外。"
                           "一次检索不够时可以用不同关键词多次检索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询。必须是脱离对话上下文也能独立成立的完整问题，"
                                       "要把代词（如'他''它'）替换为具体对象，不要照抄口语化短句",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

SYSTEM_PROMPT = (
    "你是本地知识库问答助手。工作流程：第一步，针对用户问题调用工具 search_knowledge_base 检索；"
    "第二步，仅根据工具返回的资料组织答案，不要编造；禁止跳过第一步直接回答。回答使用中文，简洁、准确。"
)


def _format_tool_result(hits) -> str:
    """把检索结果格式化为喂回给 LLM 的工具返回内容"""
    if not hits:
        return "（知识库中没有检索到相关内容）"
    return "\n---\n".join(f"[片段{i + 1}｜相关度 {s:.3f}]\n{c}" for i, (c, s, _) in enumerate(hits))


def _format_sources(hits) -> str:
    """检索依据展示（给用户核对，防幻觉）"""
    if not hits:
        return "（本次未检索到与问题相关的内容）"
    return "\n\n".join(
        f"【相关片段 {i + 1}｜融合分 {s:.3f}｜向量相似度 {v:.3f}】\n{c[:120]}..."
        for i, (c, s, v) in enumerate(hits)
    )


def _agent_reply(history: list, question: str) -> tuple:
    """
    Agent 主循环：
      messages 组装 → LLM 决策 → 若发起工具调用则执行检索并回填 → 循环直到给出最终回答
    返回 (回答文本, 最后一轮检索依据)
    """
    from zhipuai import ZhipuAI  # 延迟导入，没装 SDK 也不影响纯检索模式
    client = ZhipuAI(api_key=API_KEY)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    last_hits = []
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.1,  # 低温度 = 更稳定、少编造
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return (msg.content or "").strip(), _format_sources(last_hits)

        # 模型发起工具调用：执行检索，把结果以 role=tool 回填
        # （SDK 要求 messages 为纯 dict，需把 pydantic 对象显式转回 dict）
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            if tc.function.name == "search_knowledge_base":
                try:
                    query = json.loads(tc.function.arguments or "{}").get("query", question)
                except json.JSONDecodeError:
                    query = question
                last_hits = retrieve(query)
                tool_content = _format_tool_result(last_hits)
            else:
                tool_content = f"未知工具：{tc.function.name}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_content})

    return "（工具调用轮次已达上限，请换个问法重试）", _format_sources(last_hits)


def _pure_retrieve_reply(question: str) -> tuple:
    """未配置 API Key 时的降级模式：只检索，把最相关的原文片段直接返回"""
    hits = retrieve(question)
    if not hits:
        return "知识库中没有检索到相关内容（当前为纯检索模式，未配置 API Key）。", _format_sources(hits)
    body = "\n---\n".join(f"[片段{i + 1}｜相关度 {s:.3f}]\n{c}" for i, (c, s, _) in enumerate(hits))
    return (
        "（未配置 ZHIPU_API_KEY，当前为纯检索模式。以下是与问题最相关的原文片段：）\n\n" + body,
        _format_sources(hits),
    )


# ============ 6. Gradio 多轮对话界面 ============
def chat_respond(message: str, history: list):
    """Gradio 回调：维护多轮对话历史，调用 Agent（或降级纯检索）"""
    message = (message or "").strip()
    if not message:
        return "", history
    if index is None or not chunks:
        history = history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": "知识库为空，请先上传文档并点击「构建知识库」。"}]
        return "", history
    try:
        if API_KEY:
            reply, sources = _agent_reply(history, message)
        else:
            reply, sources = _pure_retrieve_reply(message)
        history = history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": reply}]
        return "", history, sources
    except Exception as e:
        # 出错时把真实原因显示在界面上，方便定位（比如 Key 无效 / SDK 版本 / 网络）
        import traceback
        history = history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": f"出错了：{type(e).__name__}: {e}"}]
        return "", history, traceback.format_exc()


with gr.Blocks(title="RAG 知识库问答 Agent") as demo:
    gr.Markdown(
        "# RAG 本地知识库问答 Agent\n"
        "上传文档 → 自动切分与向量化 → **混合检索(向量+BM25)** + **相似度门控** → "
        "**Function Calling 自主检索** → 多轮对话问答"
    )
    with gr.Row():
        files = gr.File(label="上传知识库文档（txt / md / pdf，可多选）", file_count="multiple")
        with gr.Column():
            build_btn = gr.Button("构建知识库", variant="primary")
            clear_btn = gr.Button("清空知识库", variant="stop")
    build_status = gr.Textbox(label="知识库状态", interactive=False)
    build_btn.click(build_index, inputs=files, outputs=build_status)
    clear_btn.click(clear_store, inputs=None, outputs=build_status)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话", height=420)
            question = gr.Textbox(label="你的问题（支持多轮追问）",
                                  placeholder="例如：这个人掌握哪些深度学习框架？")
        with gr.Column(scale=2):
            sources_box = gr.Textbox(label="检索到的原文依据（可核对，防幻觉）", lines=24, interactive=False)
    question.submit(chat_respond, [question, chatbot], [question, chatbot, sources_box])

if __name__ == "__main__":
    _boot_msg = load_store()
    if _boot_msg:
        print(_boot_msg)
    demo.launch()  # 自动打开 http://127.0.0.1:7860
