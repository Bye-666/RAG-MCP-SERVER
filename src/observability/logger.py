"""
结构化日志记录，使用 JSON Lines 格式进行追踪持久化。

提供 JSON 格式的日志记录和追踪持久化到 logs/traces.jsonl。
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """自定义格式化器，将日志记录输出为 JSON 格式。"""

    def format(self, record: logging.LogRecord) -> str:
        """
        将日志记录格式化为 JSON 字符串。

        Args:
            record: 要格式化的日志记录

        Returns:
            日志记录的 JSON 字符串表示
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 如果存在异常信息，添加到日志中
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # 从记录中添加额外字段
        if hasattr(record, "trace_id"):
            log_data["trace_id"] = record.trace_id
        if hasattr(record, "trace_type"):
            log_data["trace_type"] = record.trace_type

        return json.dumps(log_data, ensure_ascii=False)


def get_trace_logger(name: str = "trace", log_dir: str = "logs") -> logging.Logger:
    """
    获取配置为 JSON Lines 输出的日志记录器。

    Args:
        name: 日志记录器名称
        log_dir: 日志文件目录

    Returns:
        配置好的日志记录器实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # 如果日志目录不存在，创建它
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 添加带有 JSON 格式化器的文件处理器
    file_handler = logging.FileHandler(log_path / "traces.jsonl", encoding="utf-8")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    # 防止传播到根日志记录器
    logger.propagate = False

    return logger


def write_trace(trace_dict: Dict[str, Any], log_dir: str = "logs") -> None:
    """
    将追踪字典写入 logs/traces.jsonl。

    Args:
        trace_dict: 来自 TraceContext.to_dict() 的追踪字典
        log_dir: 日志文件目录
    """
    # 确保日志目录存在
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 将追踪作为 JSON 行追加到文件
    trace_file = log_path / "traces.jsonl"
    with open(trace_file, "a", encoding="utf-8") as f:
        json.dump(trace_dict, f, ensure_ascii=False)
        f.write("\n")
