# 文档知识管理系统 (Document Knowledge Management System)

一个基于 Python 的文档结构化知识抽取、建模和 RAG 问答系统。

## 🎯 核心功能

- **文档处理**: 支持 PDF（文字型和图片型）的解析和处理
- **知识抽取**: 使用 NLP 技术抽取实体、关系、摘要等结构化知识
- **知识建模**: 构建知识图谱和本体表示
- **RAG 问答**: 基于检索增强的生成式问答系统

## 🛠️ 技术栈

- **后端**: Python 3.11+
- **LLM 框架**: LangChain
- **向量存储**: FAISS (本地) / SQLite
- **文本处理**: spaCy, NLTK, Hugging Face Transformers
- **PDF 处理**: PyPDF2, pdfplumber, Pillow (图片 PDF)
- **框架**: FastAPI (API 服务)

## 📁 项目结构

```
doc-knowledge-system/
├── src/
│   ├── __init__.py
│   ├── config.py              # 配置文件
│   ├── document_processor/    # 文档处理模块
│   │   ├── __init__.py
│   │   ├── pdf_parser.py      # PDF 解析（文字+图片）
│   │   └── text_splitter.py   # 文本分割
│   ├── knowledge_extraction/  # 知识抽取模块
│   │   ├── __init__.py
│   │   ├── ner.py             # 命名实体识别
│   │   ├── relation_extractor.py  # 关系抽取
│   │   └── summarizer.py      # 文本摘要
│   ├── knowledge_modeling/    # 知识建模模块
│   │   ├── __init__.py
│   │   ├── graph_builder.py   # 知识图谱构建
│   │   └── ontology.py        # 本体管理
│   ├── rag/                   # RAG 问答模块
│   │   ├── __init__.py
│   │   ├── retriever.py       # 向量检索
│   │   ├── generator.py       # 生成式回答
│   │   └── pipeline.py        # RAG 管道
│   └── storage/               # 存储模块
│       ├── __init__.py
│       ├── vector_db.py       # FAISS 管理
│       └── sqlite_db.py       # SQLite 管理
├── api/
│   ├── __init__.py
│   ├── main.py                # FastAPI 应用入口
│   └── routes/                # 各模块 API 路由
├── tests/                     # 测试文件
├── data/                      # 示例数据
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件配置 API key
```

### 3. 初始化向量数据库

```bash
python -m src.storage.vector_db --init
```

### 4. 启动 API 服务

```bash
uvicorn api.main:app --reload
```

## 📖 使用示例

```python
from src.document_processor import PDFParser
from src.knowledge_extraction import KnowledgeExtractor
from src.rag import RAGPipeline

# 1. 处理 PDF 文档
parser = PDFParser("document.pdf")
text = parser.extract_text()

# 2. 抽取知识
extractor = KnowledgeExtractor()
entities = extractor.extract_entities(text)
relations = extractor.extract_relations(text)

# 3. 建立 RAG 管道
rag = RAGPipeline()
rag.add_documents([text])
answer = rag.query("问题？")
```

## 📝 许可证

MIT

