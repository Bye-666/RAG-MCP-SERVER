"""
TraceCollector: 收集和持久化追踪数据。

负责收集已完成的追踪并触发持久化。
"""

from typing import Optional
from .trace_context import TraceContext
from src.observability.logger import write_trace


class TraceCollector:
    """
    收集追踪数据并触发持久化到 JSON Lines。
    """

    def __init__(self, log_dir: str = "logs"):
        """
        初始化追踪收集器。

        参数:
            log_dir: 写入追踪日志的目录（默认："logs"）
        """
        self.collected_traces = []
        self.log_dir = log_dir

    def collect(self, trace: TraceContext) -> None:
        """
        收集追踪并持久化到 logs/traces.jsonl。

        参数:
            trace: 要收集的 TraceContext 实例
        """
        # 存储追踪数据
        trace_dict = trace.to_dict()
        self.collected_traces.append(trace_dict)

        # 持久化到 JSON Lines 文件
        write_trace(trace_dict, self.log_dir)

    def get_traces(self):
        """
        获取所有收集的追踪。

        返回:
            追踪字典列表
        """
        return self.collected_traces

    def clear(self):
        """清除所有收集的追踪。"""
        self.collected_traces.clear()
