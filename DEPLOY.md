# EchoMind Docker 部署说明

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
EchoMind Python 后端
后端 Nginx 反向代理
EchoMind 前端页面
```

部署成功后访问：

```text
前端页面:     http://localhost:5174
后端入口:     http://localhost:8000
Nginx 入口:   http://localhost
Swagger 文档: http://localhost:8000/docs
健康检查:     http://localhost/health
ChromaDB:     http://localhost:8001
Prometheus:  http://localhost:9090
```

## Windows PowerShell 常用命令

查看服务状态：

```powershell
docker compose ps
```

查看后端日志：

```powershell
docker compose logs -f echomind
```

查看前端日志：

```powershell
docker compose logs -f frontend
```

只重建后端：

```powershell
docker compose up -d --build echomind
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
.\build-image.ps1 -Command build-prod -ImageName echomind -Version latest
```

直接运行已经构建好的后端镜像：

```powershell
.\run-image.ps1 -ImageName echomind -Version latest -Detach
```

等价的原生 `docker run` 命令：

```powershell
docker run -d --name echomind-app --restart unless-stopped --env-file .env -p 8000:8000 -v ${PWD}\data:/app/data -v ${PWD}\logs:/app/logs echomind:latest
```

注意：`docker run` 命令最后必须带镜像名，例如 `echomind:latest`，否则 Docker 不知道要启动哪个镜像。

## 前端说明

前端代码位于：

```text
EchoMindFrontend
```

根目录的 `docker-compose.yml` 已经集成前端服务，正常情况下不需要进入前端目录单独操作。

如果只想单独启动前端：

```powershell
cd EchoMindFrontend
docker compose up -d --build
```

单独启动前端时，请确保后端已经在宿主机的 `8000` 端口运行。
