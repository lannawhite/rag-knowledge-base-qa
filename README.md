# RAG 本地知识库问答系统 — 快速上手指南

> 技术栈：Python + bge-small-zh 嵌入模型 + FAISS 向量检索 + GLM-4-Flash（免费）+ Gradio 界面

---

## 一、RAG 是什么

大模型不知道你的私有文档内容。RAG（检索增强生成）的思路：

```
用户提问
   │
   ├─① 问题向量化 ──► ② 在你的文档向量库里找最相关的 3 段 ──► ③ 把这 3 段 + 问题
   │                                                            一起塞给大模型
   └──────────────────────── ④ 大模型"开卷考试"，生成有依据的回答
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
#    方式 A（推荐，本项目默认）：在 D:\rag-kb-qa 下新建 api_key.txt，把 Key 粘贴进去保存即可
#    （api_key.txt 已被 .gitignore 忽略，不会被上传到 GitHub）
#    方式 B：设置环境变量 ZHIPU_API_KEY
set ZHIPU_API_KEY=你复制的Key      # CMD
$env:ZHIPU_API_KEY="你复制的Key"   # PowerShell

# 6. 准备嵌入模型（本地部署方式）
#    从 https://hf-mirror.com/BAAI/bge-small-zh-v1.5 下载整个模型文件夹
#    （或 hf.co 官方源），放到 D:\rag-kb-qa\bge-small-zh-v1.5
#    若下载的文件夹缺少 1_Pooling 子目录，请新建 bge-small-zh-v1.5\1_Pooling\config.json，
#    内容见文末附录。模型权重较大（约 100MB），已通过 .gitignore 排除，不入库。
```

## 三、运行（约 5 分钟）

```bash
cd D:\rag-kb-qa
python app.py
# 自动打开 http://127.0.0.1:7860
```

**验证流程**（用自带的示例文档练手）：
1. 上传 `sample_knowledge.txt`→ 点「构建知识库」
2. 问：`这个人掌握哪些深度学习框架？` → 应答：PyTorch、TensorFlow/Keras
3. 问：`他实习时用什么模型做量化？` → 应答：RKNN 工具链、FP16/INT8
4. 问一个知识库里没有的（如`北京今天天气如何`）→ 应答"知识库中没有找到相关信息"——**这一步一定要演示，这是 RAG 防幻觉的核心卖点**
5. 换成你自己找的任何 PDF/TXT（论文、产品文档、课程笔记）再试

首次运行会自动下载 bge-small-zh-v1.5（约 100MB），之后就是纯本地检索 + API 调用，CPU 电脑也流畅。

## 四、参数怎么调

| 参数 | 位置 | 作用 | 调大的影响 |
|---|---|---|---|
| `CHUNK_SIZE` | app.py 顶部 | 文本块长度（300 字符） | 块更大→上下文更完整但检索精度下降 |
| `CHUNK_OVERLAP` | app.py 顶部 | 块间重叠（50 字符） | 防止关键句被切断，过大则冗余 |
| `TOP_K` | app.py 顶部 | 召回块数（3） | 提供给模型更多依据，但可能引入噪音 |
| `temperature` | call_llm 里 | 生成随机度（0.1） | 越低越严谨，RAG 场景要低 |


## 五、常见坑

- **模型下载慢/失败**：确认 `HF_ENDPOINT` 在运行前已设置；或手动从 hf-mirror.com 下载 `BAAI/bge-small-zh-v1.5` 整个文件夹放到本地，把代码里模型名改成路径
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
