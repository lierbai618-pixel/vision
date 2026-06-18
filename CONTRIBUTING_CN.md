# 贡献指南

感谢您对本项目的关注！我们欢迎任何形式的贡献。

## 如何贡献

### 报告Bug

1. 使用GitHub Issues报告Bug
2. 描述问题的症状
3. 提供复现步骤
4. 提供系统环境信息

### 提交功能请求

1. 使用GitHub Issues提交功能请求
2. 描述您希望添加的功能
3. 说明使用场景

### 提交代码

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 开发规范

### 代码风格

- 遵循PEP 8规范
- 使用Black格式化代码
- 使用isort排序导入

```bash
# 格式化代码
black src/

# 排序导入
isort src/

# 检查代码
flake8 src/
```

### 提交规范

使用语义化提交信息：

```
<type>(<scope>): <subject>

<body>

<footer>
```

类型：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `perf`: 性能优化
- `test`: 测试
- `chore`: 构建/工具

### 测试

```bash
# 运行测试
pytest tests/ -v

# 运行带覆盖率的测试
pytest tests/ -v --cov=src/ --cov-report=html
```

## 联系方式

- GitHub Issues
- Email: your.email@example.com
