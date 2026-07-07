# HebutGuide Docker 部署说明

## 部署前检查

确认 Docker Desktop 已启动，并且当前使用的是 Linux containers。

确认项目根目录存在 `.env`，并至少配置好模型 API 相关变量，例如：

```env
ANTHROPIC_API_KEY=你的key
ANTHROPIC_MODEL=qwen-max
ANTHROPIC_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode
```

确认 Chroma ONNX 模型包已经放在项目内：

```text
model-cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz
```

Dockerfile 会把这个本地模型包复制进镜像，并解压到 Chroma 需要的缓存目录，因此构建镜像时不需要再访问国外 S3 下载模型。

## 一键全栈部署

在项目根目录执行：

```powershell
docker compose up -d --build
```

这个命令会启动：

```text
Redis
ChromaDB
Prometheus
HebutGuide Python 后端
后端 Nginx 反向代理
HebutGuide 前端页面
```

部署成功后访问：

```text
前端页面 Docker:        http://localhost:5174
前端页面 Vite 开发:     http://localhost:5173
后端入口 FastAPI:       http://localhost:8000
Nginx 入口:             http://localhost
Swagger 文档:           http://localhost:8000/docs
ReDoc 文档:             http://localhost:8000/redoc
OpenAPI JSON:           http://localhost:8000/openapi.json
健康检查:               http://localhost/health
后端健康检查:           http://localhost:8000/health
Skills 状态:            http://localhost:8000/skills
ChromaDB:               http://localhost:8001
ChromaDB 心跳:          http://localhost:8001/api/v1/heartbeat
Prometheus:             http://localhost:9090
Prometheus 健康检查:    http://localhost:9090/-/healthy
Redis:                  localhost:6379
```

## Windows PowerShell 常用命令

查看服务状态：

```powershell
docker compose ps
```

查看后端日志：

```powershell
docker compose logs -f hebutguide
```

热加载招生咨询 Skills：

```powershell
curl.exe -X POST http://localhost:8000/skills/reload
curl.exe http://localhost:8000/skills
```

查看前端日志：

```powershell
docker compose logs -f frontend
```

只重建后端：

```powershell
docker compose up -d --build hebutguide
```

只重建前端：

```powershell
docker compose up -d --build frontend
```

停止服务但保留数据卷：

```powershell
docker compose down
```

停止服务并删除数据卷：

```powershell
docker compose down -v
```

## 清空历史记忆后重新部署

最省心的全量重置方式：

```powershell
docker compose down -v
docker compose up -d --build
```

说明：

- `docker compose down -v` 会删除本项目 compose 中声明的数据卷，包括 `redis-data` 和 `chromadb-data`，同时也会删除 Prometheus、Nginx 日志卷。
- 重启后 ChromaDB 为空，后端启动时会重新导入代码中的默认知识库，并自动导入 `data/demo_docs/*.txt`。
- 如果曾经在 ChromaDB 服务不可用时使用过本地嵌入式 Chroma，还可以额外清理项目目录中的备用数据：

```powershell
Remove-Item -Recurse -Force .\data\chroma\*
```

只想定向删除 Redis 和 ChromaDB 卷时，先停服务：

```powershell
docker compose down
docker volume ls
docker volume rm hebutguide_redis-data hebutguide_chromadb-data
docker compose up -d --build
```

如果 `docker volume rm` 提示卷名不存在，以 `docker volume ls` 里实际显示的 `redis-data`、`chromadb-data` 结尾的卷名为准。

重置后建议先用一句纯寒暄测试：

```powershell
curl.exe -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\":\"你好\",\"user_id\":\"test_reset\"}"
```

预期回复应是简短问候，不应再自动带出省份、科类、位次、专业偏好或中外合作项目。

## 国内镜像源

后端镜像构建默认使用国内源：

```text
apt:           https://mirrors.aliyun.com/debian
apt security:  https://mirrors.aliyun.com/debian-security
pip:           https://pypi.tuna.tsinghua.edu.cn/simple
```

前端镜像构建默认使用 npm 国内源：

```text
npm: https://registry.npmmirror.com
```

一般情况下直接执行下面命令即可：

```powershell
docker compose up -d --build
```

## 单独构建和运行后端镜像

只构建后端应用镜像：

```powershell
.\build-image.ps1 -Command build-prod -ImageName hebutguide -Version latest
```

直接运行已经构建好的后端镜像：

```powershell
.\run-image.ps1 -ImageName hebutguide -Version latest -Detach
```

等价的原生 `docker run` 命令：

```powershell
docker run -d --name hebutguide-app --restart unless-stopped --env-file .env -p 8000:8000 -v ${PWD}\data:/app/data -v ${PWD}\logs:/app/logs hebutguide:latest
```

注意：`docker run` 命令最后必须带镜像名，例如 `hebutguide:latest`，否则 Docker 不知道要启动哪个镜像。

## 前端说明

前端代码位于：

```text
HebutGuideFrontend
```

根目录的 `docker-compose.yml` 已经集成前端服务，正常情况下不需要进入前端目录单独操作。

如果只想单独启动前端：

```powershell
cd HebutGuideFrontend
docker compose up -d --build
```

单独启动前端时，请确保后端已经在宿主机的 `8000` 端口运行。
