# PhysMaster 飞书机器人

将 PhysMaster 的物理求解能力接入飞书，用户在群聊/私聊中发送物理题，Bot 异步执行求解并推送结果。

## 快速开始

### 1. 创建飞书应用

1. 打开 [飞书开放平台](https://open.feishu.cn/app)，创建一个**企业自建应用**
2. 记录 **App ID** 和 **App Secret**

### 2. 配置权限

在应用的「权限管理」页面，申请以下权限：

| 权限 | 说明 |
|------|------|
| `im:message` | 获取与发送单聊、群组消息 |
| `im:message:send_as_bot` | 以应用身份发送消息 |
| `im:message.group_at_msg` | 接收群聊中 @ 机器人的消息 |
| `im:message.p2p_msg` | 接收用户发给机器人的单聊消息 |

### 3. 配置事件订阅

1. 进入「事件订阅」页面
2. 设置请求地址（Request URL）为：`https://<your-domain>/webhook/event`
3. 添加事件：`im.message.receive_v1`（接收消息）
4. 记录页面上的 **Verification Token**

### 4. 配置 Bot

```bash
cd feishu
cp config_example.yaml config.yaml
```

编辑 `config.yaml`，填入：

```yaml
feishu:
  app_id: "cli_xxxx"          # 替换为你的 App ID
  app_secret: "xxxx"          # 替换为你的 App Secret
  verification_token: "xxxx"  # 替换为事件订阅的 Verification Token
  encrypt_key: ""             # 如需加密可填入 Encrypt Key

server:
  host: "0.0.0.0"
  port: 9000
  max_workers: 2              # 并发求解任务数

physmaster:
  config_path: "config.yaml"  # 指向主项目 config.yaml
```

### 5. 安装依赖

```bash
pip install fastapi uvicorn requests pyyaml
```

### 6. 启动服务

```bash
cd feishu
python bot.py
```

服务默认监听 `0.0.0.0:9000`。

### 7. 暴露公网地址（开发环境）

如果在本地开发，可使用 ngrok 暴露端口：

```bash
ngrok http 9000
```

将 ngrok 给出的 HTTPS 地址填入飞书开放平台的「事件订阅 → 请求地址」，例如：

```
https://abcd1234.ngrok.io/webhook/event
```

### 8. 启用机器人

在飞书开放平台「应用发布」中发布应用版本，或在测试企业中直接启用。

## 使用方式

- **私聊**：直接给 Bot 发送物理题文字
- **群聊**：@Bot 后跟物理题文字

Bot 会立即回复「正在求解…」，求解完成后（通常 5-30 分钟）主动推送 summary 结果。

## 架构说明

```
用户发消息 ──→ 飞书服务器 ──→ POST /webhook/event
                                    │
                              bot.py (FastAPI)
                                    │
                      ┌─────────────┴──────────────┐
                      │  立即回复 "正在求解…"         │
                      │  提交到 ThreadPoolExecutor   │
                      └─────────────┬──────────────┘
                                    │ (后台线程)
                              worker.py
                                    │
                      Clarifier → MCTS → Summarizer
                                    │
                              结果推送回飞书
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `bot.py` | FastAPI 服务，接收 webhook、发送消息 |
| `worker.py` | 调用 PhysMaster pipeline 的封装 |
| `config_example.yaml` | 配置模板 |
| `config.yaml` | 实际配置（不提交到 git） |

## 健康检查

```bash
curl http://localhost:9000/health
```

## 注意事项

- `config.yaml` 包含密钥，请勿提交到版本控制（已在 `.gitignore` 中排除）
- `max_workers` 控制并发求解数，每个任务消耗较多内存和 GPU/API 资源，建议设为 1-2
- 飞书消息有长度限制，超长 summary 会自动截断，完整结果请查看服务器上的 `summary.md`
- Bot 内置消息去重机制，飞书重试不会导致重复求解
