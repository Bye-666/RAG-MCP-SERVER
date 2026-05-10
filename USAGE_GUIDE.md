# RAG-MCP-SERVER 快速使用指南

## 📋 目录
1. [必需配置](#必需配置)
2. [快速开始](#快速开始)
3. [使用流程](#使用流程)
4. [常见问题](#常见问题)

---

## 🔧 必需配置

### 1. API密钥配置（必需）

项目需要配置以下API密钥才能正常运行：

#### 方式一：通过Dashboard配置（推荐）

1. 启动Dashboard：`streamlit run src/observability/dashboard/app.py`
2. 访问 http://localhost:8501
3. 点击侧边栏 **⚙️ 系统设置**
4. 配置以下必需项：

**LLM配置**（必需）：
- 提供商：`openai` / `azure` / `ollama` / `deepseek`
- API密钥：你的API Key
- 模型：`gpt-4` / `gpt-3.5-turbo` 等
- API地址：如使用Azure需填写

**Embedding配置**（必需）：
- 提供商：`openai` / `azure` / `ollama`
- API密钥：你的API Key
- 模型：`text-embedding-3-small` 等
- 向量维度：1536（OpenAI默认）

**向量存储配置**（可选，有默认值）：
- 提供商：`chroma`（默认）/ `qdrant`
- 集合名称：`rag_collection`（默认）
- 持久化目录：`./data/chroma`（默认）

5. 点击 **💾 保存配置**
6. 重启Dashboard使配置生效

#### 方式二：手动编辑配置文件

编辑 `config/settings.yaml`：

```yaml
llm:
  provider: openai
  api_key: sk-your-openai-api-key-here
  model: gpt-4
  api_base: https://api.openai.com/v1
  temperature: 0.7
  max_tokens: 2000

embedding:
  provider: openai
  api_key: sk-your-openai-api-key-here
  model: text-embedding-3-small
  api_base: https://api.openai.com/v1
  dimensions: 1536

vector_store:
  provider: chroma
  collection_name: rag_collection
  persist_directory: ./data/chroma

retrieval:
  dense_top_k: 10
  sparse_top_k: 10
  final_top_k: 5

reranker:
  enabled: true
  provider: none
  model: ""
  top_k: 3
```

### 2. 依赖安装（必需）

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

---

## 🚀 快速开始

### 步骤1：启动Dashboard

```bash
# 激活虚拟环境
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 启动Dashboard
streamlit run src/observability/dashboard/app.py
```

访问：http://localhost:8501

### 步骤2：配置API密钥

在Dashboard的 **⚙️ 系统设置** 页面配置LLM和Embedding的API密钥。

### 步骤3：上传文档

1. 在Dashboard点击 **📥 摄取管理**
2. 点击 **选择 PDF 文件** 上传文档
3. 点击 **🚀 开始摄取**
4. 等待摄取完成（可以看到进度条）

### 步骤4：测试查询

1. 在Dashboard点击 **🔍 查询测试**
2. 输入查询文本
3. 调整参数（可选）
4. 点击 **🚀 执行查询**
5. 查看检索结果

---

## 📖 使用流程

### 完整工作流程

```
1. 配置API密钥
   ↓
2. 上传PDF文档（摄取管理）
   ↓
3. 查看摄取结果（数据浏览）
   ↓
4. 测试查询（查询测试）
   ↓
5. 查看追踪信息（追踪总览/查询追踪）
   ↓
6. 运行评估（评估面板）
```

### 各页面功能详解

#### 🏠 系统总览
- 查看当前配置（LLM、Embedding、向量存储等）
- 查看系统状态
- 无需配置，自动读取

#### 📊 追踪总览
- 查看所有追踪记录（摄取+查询）
- 按类型筛选
- 搜索特定追踪
- 无需配置

#### 📚 数据浏览
- 浏览已摄取的文档
- 查看文档详情（Chunk、图片）
- 搜索文档
- 需要先摄取文档

#### 📥 摄取管理
- **上传PDF文档**
- 配置摄取参数
- 查看摄取进度
- 删除已摄取文档
- **这是使用的第一步**

#### 📈 摄取追踪
- 查看摄取历史
- 查看各阶段耗时
- 分析性能瓶颈
- 自动记录，无需配置

#### 🔍 查询测试
- **交互式查询测试**
- 调整检索参数（top_k、权重等）
- 查看检索结果和分数
- 需要先摄取文档

#### 📉 查询追踪
- 查看查询历史
- Dense vs Sparse对比
- Rerank前后排名变化
- 自动记录，无需配置

#### 📊 评估面板
- 运行RAG评估
- 查看评估指标
- 使用黄金测试集
- 需要配置评估器和测试集

#### ⚙️ 系统设置
- **配置API密钥**
- 修改系统参数
- 保存/重新加载配置
- **必需配置**

---

## 🎯 典型使用场景

### 场景1：第一次使用

```bash
1. 启动Dashboard
2. 进入"系统设置"配置API密钥
3. 重启Dashboard
4. 进入"摄取管理"上传PDF文档
5. 进入"查询测试"测试检索效果
```

### 场景2：日常使用

```bash
1. 启动Dashboard
2. 进入"摄取管理"上传新文档
3. 进入"查询测试"进行查询
4. 进入"查询追踪"查看详细分析
```

### 场景3：性能调优

```bash
1. 进入"查询测试"测试不同参数
2. 进入"查询追踪"分析耗时
3. 进入"系统设置"调整配置
4. 进入"评估面板"运行评估
```

### 场景4：MCP集成

```bash
1. 配置MCP Server（见下文）
2. 在Claude/Copilot中调用工具
3. 在Dashboard查看追踪记录
```

---

## 🔌 MCP配置（可选）

如果要在Claude Desktop或GitHub Copilot中使用，需要配置MCP。

### Claude Desktop配置

编辑配置文件：
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rag-knowledge-hub": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "D:\\Dev\\Workspace\\AI\\RAG-MCP-SERVER",
      "env": {
        "PYTHONPATH": "D:\\Dev\\Workspace\\AI\\RAG-MCP-SERVER"
      }
    }
  }
}
```

### GitHub Copilot配置

在项目根目录创建 `.vscode/mcp.json`：

```json
{
  "mcpServers": {
    "rag-knowledge-hub": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    }
  }
}
```

### 可用工具

- `query_knowledge_hub`: 查询知识库
- `list_collections`: 列出所有集合
- `get_document_summary`: 获取文档摘要

---

## ❓ 常见问题

### Q1: 启动Dashboard后无法访问？

**A**: 检查端口8501是否被占用：
```bash
# Windows
netstat -ano | findstr :8501

# 如果被占用，可以指定其他端口
streamlit run src/observability/dashboard/app.py --server.port 8502
```

### Q2: 上传文档后摄取失败？

**A**: 检查以下几点：
1. API密钥是否正确配置
2. 网络是否能访问API服务
3. 查看错误信息，可能是API额度不足
4. 检查PDF文件是否损坏

### Q3: 查询没有返回结果？

**A**: 可能原因：
1. 还没有摄取任何文档
2. 查询关键词与文档内容不匹配
3. top_k设置过小
4. 向量数据库未正确初始化

### Q4: 如何使用本地模型（Ollama）？

**A**: 在系统设置中：
1. LLM提供商选择 `ollama`
2. API地址填写 `http://localhost:11434`
3. 模型填写已安装的模型名（如 `llama2`）
4. Embedding同样配置

### Q5: 配置修改后不生效？

**A**: 需要重启Dashboard：
1. 停止当前Dashboard（Ctrl+C）
2. 重新运行 `streamlit run src/observability/dashboard/app.py`

### Q6: 如何查看详细的错误日志？

**A**: 
1. Dashboard页面会显示错误信息
2. 查看终端输出
3. 查看 `logs/traces.jsonl` 文件

### Q7: 向量数据库存储在哪里？

**A**: 
- 默认位置：`./data/chroma/`
- 可在系统设置中修改
- 删除此目录会清空所有数据

### Q8: 如何删除已摄取的文档？

**A**: 
1. 进入"摄取管理"页面
2. 找到要删除的文档
3. 点击"🗑️ 删除"按钮
4. 确认删除

---

## 📝 配置检查清单

使用前请确认以下配置已完成：

- [ ] 虚拟环境已创建并激活
- [ ] 依赖已安装（`pip install -r requirements.txt`）
- [ ] LLM API密钥已配置
- [ ] Embedding API密钥已配置
- [ ] Dashboard可以正常启动
- [ ] 已上传至少一个测试文档
- [ ] 查询测试可以返回结果

---

## 🎓 学习路径

### 初学者路径

1. **第1天**：配置环境，启动Dashboard，熟悉界面
2. **第2天**：上传文档，测试查询，理解检索流程
3. **第3天**：调整参数，查看追踪，理解性能指标
4. **第4天**：配置MCP，在Claude/Copilot中使用
5. **第5天**：运行评估，理解评估指标

### 进阶路径

1. 研究代码架构（可插拔设计）
2. 自定义Loader/Splitter/Reranker
3. 添加新的评估指标
4. 扩展Dashboard功能
5. 集成到自己的项目

---

## 📞 获取帮助

- 查看 `README.md` 了解项目详情
- 查看 `DEV_SPEC.md` 了解架构设计
- 查看 `DASHBOARD_REVIEW.md` 了解Dashboard功能
- 查看 `tests/e2e/E2E_VERIFICATION_REPORT.md` 了解测试覆盖

---

**祝使用愉快！** 🎉
