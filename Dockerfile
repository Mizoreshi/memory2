# 用 Python 精简镜像（体积小，部署快）
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存加速
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 端口（Render 等平台会通过 PORT 环境变量分配）
# 默认 8000
ENV PORT=8000

# 启动网关
CMD ["python", "main.py"]
