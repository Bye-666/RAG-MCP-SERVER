# 清理临时文件路径记录

## 问题说明

在修复文件上传逻辑之前，Dashboard 上传的文件被保存到系统临时目录（如 `C:\Users\...\AppData\Local\Temp\`），这些临时路径被存储到数据库中。虽然文件在摄取后被删除，但记录仍然存在，导致：

1. 数据浏览页面显示"未找到文档"错误
2. 无法正常删除这些记录

## 解决方案

### 方案 1：使用清理脚本（推荐）

运行清理脚本批量删除所有临时路径的记录：

```bash
# Windows PowerShell
python scripts/cleanup_temp_records.py

# 或使用虚拟环境
.venv/Scripts/python scripts/cleanup_temp_records.py
```

脚本会：
1. 扫描所有文档记录
2. 识别临时路径的记录
3. 显示待删除的记录列表
4. 确认后批量删除

### 方案 2：在 Dashboard 中手动删除

修复后的删除逻辑现在可以处理临时路径记录：

1. 进入 **Ingestion 管理** 页面
2. 找到临时路径的文档（路径包含 `\Temp\` 或 `AppData\Local\Temp`）
3. 点击 **删除** 按钮
4. 确认删除

即使没有找到 chunks，系统也会尝试从 integrity checker 中删除记录。

## 修复内容

### 1. 改进的删除逻辑

`src/ingestion/document_manager.py` 中的 `delete_document()` 方法现在：
- 即使没有找到 chunks，也会尝试删除 integrity 记录
- 从 integrity checker 中查找 file_hash
- 只要删除了任何数据（chunks、images 或 integrity 记录），就视为成功

### 2. 新的文件存储方式

从现在开始，上传的文件会：
- 保存到 `./data/uploads/`（可配置）
- 使用原始文件名
- 持久化存储，不会被自动删除
- 路径清晰可见，便于管理

## 预防措施

为避免将来出现类似问题：

1. **配置上传目录**：在 `config/settings.yaml` 中设置合适的上传目录
   ```yaml
   config:
     storage:
       upload_directory: ./data/uploads  # 或其他持久化路径
   ```

2. **定期备份**：定期备份 `data/db/` 目录中的数据库文件

3. **监控磁盘空间**：确保上传目录有足够的磁盘空间

## 常见问题

**Q: 清理脚本会删除什么？**
A: 只删除路径包含临时目录标识（如 `\Temp\`、`AppData\Local\Temp`）的记录。正常路径的文档不会受影响。

**Q: 删除后能恢复吗？**
A: 无法恢复。建议在删除前确认列表中的文档确实是不需要的。

**Q: 如果我想保留某些临时路径的文档怎么办？**
A: 在运行清理脚本前，先在 Dashboard 中查看这些文档，如果需要保留，请重新上传原始文件。

**Q: 新上传的文件会保存在哪里？**
A: 默认保存在 `./data/uploads/`，可通过配置文件修改。
