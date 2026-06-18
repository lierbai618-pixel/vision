# 部署指南

## 概述

本文档介绍如何部署智能视觉系统到不同环境。

## 本地部署

### 1. 安装依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/vision-system.git
cd vision-system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.yaml` 文件：

```yaml
model:
  detection_model: 'yolov8n.pt'
  face_confidence: 0.5

api:
  host: '0.0.0.0'
  port: 8000

web:
  host: '0.0.0.0'
  port: 8501
```

### 3. 启动服务

```bash
# 启动API服务
uvicorn api:app --host 0.0.0.0 --port 8000

# 启动Web界面
streamlit run app.py --server.port 8501
```

## Docker部署

### 1. 构建镜像

```bash
docker build -t vision-system .
```

### 2. 运行容器

```bash
# 单容器运行
docker run -d \
  -p 8000:8000 \
  -p 8501:8501 \
  -v ./data:/app/data \
  -v ./models:/app/models \
  --name vision-system \
  vision-system
```

### 3. 使用Docker Compose

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## 云服务器部署

### 1. 阿里云ECS

```bash
# 登录服务器
ssh root@your-server-ip

# 安装Docker
curl -fsSL https://get.docker.com | sh

# 克隆项目
git clone https://github.com/yourusername/vision-system.git
cd vision-system

# 构建并启动
docker-compose up -d
```

### 2. 腾讯云CVM

```bash
# 登录服务器
ssh root@your-server-ip

# 安装Docker
yum install -y docker

# 启动Docker
systemctl start docker
systemctl enable docker

# 克隆项目
git clone https://github.com/yourusername/vision-system.git
cd vision-system

# 构建并启动
docker-compose up -d
```

## Nginx配置

### 1. 安装Nginx

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nginx

# CentOS
sudo yum install nginx
```

### 2. 配置Nginx

编辑 `/etc/nginx/sites-available/vision-system`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # API代理
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        client_max_body_size 100M;
    }

    # Web界面代理
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 3. 启用配置

```bash
sudo ln -s /etc/nginx/sites-available/vision-system /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## HTTPS配置

### 1. 使用Let's Encrypt

```bash
# 安装Certbot
sudo apt install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d your-domain.com

# 自动续期
sudo certbot renew --dry-run
```

### 2. 手动配置

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # ... 其他配置
}
```

## 监控配置

### 1. 使用Prometheus

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vision-system'
    static_configs:
      - targets: ['localhost:8000']
```

### 2. 使用Grafana

```bash
# 安装Grafana
docker run -d \
  -p 3000:3000 \
  --name=grafana \
  grafana/grafana
```

## 性能优化

### 1. GPU加速

```bash
# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 重启Docker
sudo systemctl restart docker

# 运行GPU容器
docker run --gpus all -p 8000:8000 vision-system
```

### 2. 内存优化

```yaml
# config.yaml
model:
  detection_model: 'yolov8n.pt'  # 使用更小的模型

api:
  workers: 2  # 减少worker数量
```

## 故障排除

### 1. 端口被占用

```bash
# 查看端口占用
lsof -i :8000

# 杀死进程
kill -9 <PID>
```

### 2. 内存不足

```bash
# 查看内存使用
free -h

# 增加swap空间
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 3. 模型下载失败

```bash
# 手动下载模型
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt -O models/yolov8n.pt
```

## 安全建议

1. 使用HTTPS
2. 配置防火墙
3. 定期更新依赖
4. 使用强密码
5. 启用日志记录
6. 定期备份数据
