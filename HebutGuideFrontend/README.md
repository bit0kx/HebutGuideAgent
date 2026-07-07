# HebutGuide 前端部署说明

这是 HebutGuide 的 Vue + Vite 前端项目，可连接 Python 后端和 Java 后端。当前全栈 Docker 部署默认连接 Python 后端。

## 本地开发

进入前端目录：

```powershell
cd HebutGuideFrontend
```

安装依赖：

```powershell
npm install
```

启动开发服务：

```powershell
npm run dev
```

访问：

```text
http://localhost:5173
```

开发模式下，Vite 会把接口代理到：

```text
/api/python -> http://localhost:8000
/api/java   -> http://localhost:8080
```

## 单独部署前端

如果后端已经在宿主机运行，可以只启动前端容器：

```powershell
cd HebutGuideFrontend
docker compose up -d --build
```

访问：

```text
http://localhost:5174
```

注意：前端 Dockerfile 已经会在镜像构建阶段自动执行 `npm ci` 和 `npm run build`，不需要手动先执行 `npm run build`。

## 全栈部署

推荐在项目根目录使用根 compose 一次启动后端、依赖服务和前端：

```powershell
cd D:\桌面\HebutGuide
docker compose up -d --build
```

访问地址：

```text
前端页面:     http://localhost:5174
后端入口:     http://localhost:8000
Nginx 入口:   http://localhost
接口文档:     http://localhost:8000/docs
Prometheus:  http://localhost:9090
```

全栈部署时，前端容器会通过 Docker Desktop 的宿主机地址访问 Python 后端：

```text
http://host.docker.internal:8000
```

## 国内源

前端镜像构建默认使用 npm 国内源：

```text
https://registry.npmmirror.com
```

如需覆盖：

```powershell
$env:NPM_REGISTRY="https://registry.npmmirror.com"
docker compose up -d --build frontend
```

## 常用命令

查看前端容器状态：

```powershell
docker compose ps frontend
```

查看前端日志：

```powershell
docker compose logs -f frontend
```

只重建前端：

```powershell
docker compose up -d --build frontend
```

停止全栈服务：

```powershell
docker compose down
```

