# -*- coding: utf-8 -*-
"""
RAG 本地知识库问答系统（快速上手版）
================================================
完整链路：
  上传文档 → 文本切分 → 向量化(Embedding) → FAISS 向量检索
  → 拼装 Prompt（问题 + 检索到的相关片段）→ LLM 生成回答 → Gradio Web 界面

运行方式：
  1. 设置智谱 API Key（见 README.md，不设置也能用，只是只检索不生成回答）
  2. python app.py
  3. 浏览器打开 http://127.0.0.1:7860

与课程代码的对应关系：
  - 老师代码里的 AutoModel/AutoTokenizer（llm/pre_training.py）
    → 本文件里的 SentenceTransformer("BAAI/bge-small-zh-v1.5")，同一族接口
  - 垃圾短信项目里手写的 Word2sequence（分词/序列化）
    → RAG 里换成"预训练嵌入模型"直接把文本变成向量，效果更好
  - classification/train.py 里训练模型
    → RAG 不训练模型，做的是"检索 + 拼上下文"，这是它快的原因
"""
import os
import numpy as np
import faiss
import gradio as gr
from pypdf import PdfReader
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
LLM_MODEL = "glm-4-flash"   # 智谱免费模型；用 DeepSeek 则改成 "deepseek-chat" 并换 SDK
CHUNK_SIZE = 300            # 每个文本块最大字符数
CHUNK_OVERLAP = 50          # 相邻块重叠字符数（保证语义不断裂）
TOP_K = 3                   # 每次检索召回的文本块数量

# 嵌入模型：把中文文本编码成 512 维向量。首次运行自动下载（约 100MB），CPU 即可
print("正在加载嵌入模型 bge-small-zh-v1.5 ...（首次运行需要下载）")
embedder = SentenceTransformer("D://rag-kb-qa//bge-small-zh-v1.5")

# 全局知识库状态
chunks = []          # 所有文本块（字符串列表）
index = None         # FAISS 向量索引


# ============ 2. 文档读取与切分 ============
def read_file(path: str) -> str:
    """读取 txt / md / pdf 文件为纯文本"""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    # txt / md 统一按 utf-8 读
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str):
    """把长文本切成带重叠的小块——RAG 最核心的预处理步骤"""
    text = text.strip()
    result = []
    start = 0
    while start < len(text):
        piece = text[start : start + CHUNK_SIZE]
        if len(piece) > 30:  # 过滤太短的碎片
            result.append(piece)
        start += CHUNK_SIZE - CHUNK_OVERLAP  # 步进 = 块长 - 重叠
    return result


# ============ 3. 建向量索引 ============
def build_index(files):
    """上传文件后重建知识库：切分 → 向量化 → 存入 FAISS"""
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
    return f"知识库构建完成：共 {len(chunks)} 个文本块（来自 {len(files)} 个文件）"


# ============ 4. 检索 + LLM 生成 ============
def retrieve(question: str):
    """向量检索：问题也变成向量，找知识库里最相似的 TOP_K 块"""
    if index is None or len(chunks) == 0:
        return []
    q_vec = embedder.encode([question], normalize_embeddings=True)
    scores, ids = index.search(np.array(q_vec, dtype="float32"), TOP_K)
    return [(chunks[i], float(scores[0][rank])) for rank, i in enumerate(ids[0])]


def call_llm(question: str, context: str) -> str:
    """调用大模型生成回答。没配 API Key 时返回提示，系统降级为纯检索模式"""
    prompt = (
        "你是一个严谨的知识库问答助手。请只根据下面提供的参考资料回答问题，"
        "如果资料中没有相关内容，请直接说'知识库中没有找到相关信息'，不要编造。\n\n"
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{question}"
    )
    if not API_KEY:
        return "（未配置 ZHIPU_API_KEY，当前为纯检索模式。以下是与问题最相关的原文片段：）\n\n" + context
    from zhipuai import ZhipuAI  # 延迟导入，没装 SDK 也不影响检索功能
    client = ZhipuAI(api_key=API_KEY)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # 低温度 = 更稳定、少编造
    )
    return resp.choices[0].message.content


def answer(question: str):
    """完整 RAG 问答：检索 → 拼上下文 → 生成"""
    if not question.strip():
        return "请输入问题", ""
    try:
        hits = retrieve(question)
        if not hits:
            return "知识库为空，请先上传文档", ""
        context = "\n---\n".join(f"[片段{i+1}] {c}" for i, (c, s) in enumerate(hits))
        reply = call_llm(question, context)
        sources = "\n\n".join(
            f"【相关片段 {i+1}｜相似度 {s:.3f}】\n{c[:120]}..." for i, (c, s) in enumerate(hits)
        )
        return reply, sources
    except Exception as e:
        # 出错时把真实原因显示在界面上，方便定位（比如 Key 无效 / SDK 版本 / 网络）
        import traceback
        return f"出错了：{type(e).__name__}: {e}", traceback.format_exc()


# ============ 5. Gradio Web 界面 ============
with gr.Blocks(title="RAG 本地知识库问答") as demo:
    gr.Markdown("# RAG 本地知识库问答系统\n上传文档 → 自动切分与向量化 → 检索增强问答")
    with gr.Row():
        files = gr.File(label="上传知识库文档（txt / md / pdf，可多选）", file_count="multiple")
        build_btn = gr.Button("构建知识库", variant="primary")
    build_status = gr.Textbox(label="知识库状态", interactive=False)
    build_btn.click(build_index, inputs=files, outputs=build_status)

    question = gr.Textbox(label="你的问题", placeholder="例如：这个人掌握哪些深度学习框架？")
    ask_btn = gr.Button("提问", variant="primary")
    answer_box = gr.Textbox(label="回答", lines=6, interactive=False)
    sources_box = gr.Textbox(label="检索到的原文依据（可核对，防幻觉）", lines=6, interactive=False)
    ask_btn.click(answer, inputs=question, outputs=[answer_box, sources_box])

if __name__ == "__main__":
    demo.launch()  # 自动打开 http://127.0.0.1:7860
