# 清雨问影

> 自研「九洲灵寰」修仙世界观 · NPC 智能对话游戏
> 与雨司堂执事**墨清影**谈笑间，问尽清雨宗门规与江湖旧事。

---

## 一、项目简介

《清雨问影》是一套基于 RAG（检索增强生成）的修仙世界观 NPC 文字交互游戏。

项目自创「九洲灵寰」世界观，将世界设定、宗门门规、人物志等内容整理为 PDF 设定集，通过向量检索注入大模型上下文，让玩家以清雨宗新晋外门弟子的身份，与雨司堂常驻执事**墨清影**（金丹中期）自由对话。

她说话温柔平缓、耐心细致，熟知宗门典籍；面对浪费灵材、欺压低阶修士等违规试探，她会温柔规劝「不太合规矩呢……」；遇到典籍中不曾记载的事，她会坦然道一句「此事我不甚知晓」，绝不编造设定。

整条链路完全本地化运行：**知识库、向量库、大模型全部部署在本机**，无需任何云端 API Key，数据不出门。

---

## 二、技术架构

| 层级 | 技术选型 | 说明 |
|---|---|---|
| 前端交互 | Unity 2D + TextMeshPro | `UnityWebRequest` POST 对话，UI 实时展示 |
| 服务接口 | FastAPI + Uvicorn | `POST /chat`，JSON 进出，统一异常兜底 |
| 对话生成 | Ollama（qwen2.5:7b） | 本地大模型，LangChain `ChatOllama` 接入 |
| 检索增强 | LangChain + Chroma | k=3 相似度检索，持久化向量库 |
| 文本嵌入 | sentence-transformers | `BAAI/bge-small-zh-v1.5` 中文语义模型 |
| 知识入库 | pypdf + RecursiveCharacterTextSplitter | PDF 解析，chunk_size=450 / overlap=60 |

```
┌─────────────┐   POST /chat    ┌──────────────────────────────┐
│  Unity 2D   │ ──────────────▶ │  FastAPI (main.py)           │
│  前端界面    │ ◀────────────── │  参数校验 / 异常兜底 / 日志    │
└─────────────┘   {"answer"}    └──────────────┬───────────────┘
                                               │
                              ┌────────────────▼────────────────┐
                              │  LangChain 对话链                │
                              │  墨清影人设 Prompt + 检索上下文   │
                              └───────┬─────────────────┬───────┘
                                      │                 │
                        ┌─────────────▼───┐   ┌─────────▼───────────┐
                        │ Ollama 本地大模型 │   │ Chroma 向量库        │
                        │ qwen2.5:7b      │   │ ./chroma_db 持久化   │
                        └─────────────────┘   └─────────┬───────────┘
                                                        │
                                          ┌─────────────▼─────────────┐
                                          │ PDF 设定集 → 切分 → 嵌入   │
                                          │ world_data/*.pdf          │
                                          │ bge-small-zh-v1.5         │
                                          └───────────────────────────┘
```

---

## 三、项目亮点

1. **私有知识库 + 强 Prompt 双重约束，根治人设崩坏与幻觉**
   - 知识库侧：回答内容强制锚定检索到的设定集原文，超纲问题触发「此事我不甚知晓」兜底，杜绝编造；
   - Prompt 侧：系统提示词约束身份（金丹执事）、语气（温柔平缓）、口头禅（「不急，慢慢说。」等三句）、行为边界（违规温柔规劝）、禁项（不自称 AI、不出现现代词汇）。
2. **全链路本地化，零云端依赖**：嵌入模型、向量库、大模型均在本机运行，无 API Key、无数据外传、无调用费用。
3. **可复现的人设评估体系**：配套 `test_record.md` 五类测试用例（合规提问 / 日常闲聊 / 违规试探 / 超纲提问 / 性格测试），量化检验人设稳定性。
4. **前后端分离，接口即游戏服务**：FastAPI 统一出口，命令行、Unity、未来的其他客户端均可复用同一套 RAG 链路。
5. **工程化细节完备**：向量库覆盖重建、PDF 读取异常捕获、空输入校验、网络超时兜底、人设口吻的友好错误文案。

---

## 四、环境部署步骤

### 4.1 基础环境

- Windows 10/11（Linux/macOS 同理）
- Python 3.12
- Ollama（本机大模型运行时）

### 4.2 创建 Python 虚拟环境并安装依赖

```powershell
cd <项目根目录>

# 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 升级 pip
python -m pip install --upgrade pip setuptools wheel

# 一次性安装全部依赖
python -m pip install langchain langchain-community langchain-chroma pypdf sentence-transformers fastapi uvicorn langchain-ollama
```

### 4.3 安装 Ollama 并拉取对话模型

1. 下载安装：https://ollama.com/download/windows （或 `winget install Ollama.Ollama`）
2. 拉取模型（首次约数分钟）：

```powershell
ollama pull qwen2.5:7b
```

### 4.4 构建向量知识库

将世界观设定集放入 `world_data/` 目录（当前已有 `九洲灵寰_墨清影设定集.pdf`），然后执行：

```powershell
python build_vector.py
```

- 自动读取 `world_data/` 下全部 PDF，切分后嵌入入库；
- 持久化目录为 `./chroma_db`，**已存在时自动删除重建**；
- 嵌入模型 `BAAI/bge-small-zh-v1.5` 首次运行自动下载（代码内置 HF 镜像加速，国内网络可直接使用）。

---

## 五、项目启动完整方法

### 方式一：命令行对话（快速体验）

```powershell
.\.venv\Scripts\Activate.ps1
python qingyu_chat.py
```

- 直接与墨清影多轮对话，保留最近 10 轮上下文；
- **输入空格后回车**即离开对话。

### 方式二：启动 HTTP 服务（供 Unity 等前端调用）

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn main:app --host 127.0.0.1 --port 8000
```

接口调用（PowerShell）：

```powershell
$body = @{ question = "清雨宗的门规有哪些？" } | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat" `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

返回：`{"answer": "墨清影的回答"}`；浏览器打开 `http://127.0.0.1:8000/docs` 可查看在线接口文档。

### 方式三：Unity 2D 前端

1. 新建 Unity 2022 工程，将 `QingYuChat.cs` 放入 `Assets/Scripts/`；
2. Canvas 中创建 `TMP_InputField`（输入框）、`TMP_Text`（对话显示）、`Button`（发送）、`TMP_Text`（加载提示，可选）；
3. 空物体挂载 `QingYuChat`，在 Inspector 中拖入以上 UI 引用；
4. 确认后端已启动，点击 Play 即可在游戏界面与墨清影对话。

---

## 六、演示效果说明

| 场景 | 表现 |
|---|---|
| 请教门规 | 检索设定集原文，以长辈口吻耐心转述，如「不急，慢慢说。」 |
| 闲聊风景 | 世界观内自然应答，无出戏的现代词汇 |
| 试探「怎么浪费灵材最省事？」 | 温柔规劝并说出「不太合规矩呢……」，绝不提供方法 |
| 询问「域外魔界功法」 | 典籍无载，回复「此事我不甚知晓」，零编造 |
| 尖锐挑衅 | 情绪稳定、不卑不亢，保持金丹执事气度 |

完整对话实测样例与人设评估记录见 [`test_record.md`](test_record.md)。五类用例覆盖知识命中、闲聊、违规诱导、超纲防幻觉与性格压力测试，可复现地验证「人设不崩、设定不编」两大核心目标。

---

## 七、简历项目描述

> **清雨问影 —— 基于 RAG 的修仙 NPC 智能对话游戏**（个人独立开发）
>
> 自研「九洲灵寰」修仙世界观，构建以设定集 PDF 为私有知识库的 NPC 文字交互游戏。技术栈：Python + LangChain + Chroma + sentence-transformers + FastAPI + Unity 2D + Ollama 本地大模型。通过「向量检索锚定原文 + 强人设系统提示词」双重约束，系统性解决大模型在角色扮演场景下的人设崩坏与幻觉编造问题，并设计五类测试用例量化验证人设稳定性；实现 PDF 解析切分、向量库覆盖重建、k=3 相似度检索、FastAPI 统一接口与 Unity 前端对接，全链路本地化部署，零云端依赖、零调用成本。

---

## 附：项目结构

```
QingYu_Game/
├── world_data/                 # 世界观设定集 PDF（私有知识库源文件）
├── chroma_db/                  # Chroma 持久化向量库（build_vector.py 生成）
├── load_world.py               # PDF 加载与文本切分预览
├── build_vector.py             # 知识库构建：切分 → 嵌入 → 入库
├── qingyu_chat.py              # 墨清影 RAG 对话核心（人设 Prompt / 检索 / 生成）
├── main.py                     # FastAPI 接口封装（POST /chat）
├── QingYuChat.cs               # Unity 2D 前端对话脚本
├── test_record.md              # 人设测试记录（五类用例评估）
└── README.md
```
