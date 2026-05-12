"""
TraceContext: 用于追踪流水线阶段的最小实现。

这是 Phase C 的占位符实现。将在 Phase F 中增强，
包括结构化日志、持久化和详细指标。
"""

import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class StageRecord:
    """单个流水线阶段执行的记录"""
    stage_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        """计算阶段持续时间（毫秒）"""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000


class TraceContext:
    """
    用于追踪流水线执行的最小追踪上下文。

    提供:
    - 唯一 trace_id 生成
    - 带计时的阶段记录
    - 基本元数据存储

    Phase F 将添加:
    - JSON Lines 持久化
    - 结构化日志
    - 详细指标和瀑布图可视化
    """

    def __init__(self, trace_id: Optional[str] = None, trace_type: str = "query"):
        """
        初始化追踪上下文。

        参数:
            trace_id: 可选的追踪 ID。如果未提供，生成新的 UUID。
            trace_type: 追踪类型 - "query" 或 "ingestion"
        """
        self.trace_id = trace_id or str(uuid.uuid4())
        self.trace_type = trace_type
        self.stages: List[StageRecord] = []
        self.metadata: Dict[str, Any] = {}
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None

    def record_stage(self, stage_name: str, metadata: Optional[Dict[str, Any]] = None) -> StageRecord:
        """
        记录流水线阶段的开始。

        参数:
            stage_name: 阶段名称（例如 "chunk_refiner"、"dense_encoder"）
            metadata: 此阶段的可选元数据

        返回:
            可用于标记完成的 StageRecord 对象
        """
        record = StageRecord(
            stage_name=stage_name,
            start_time=datetime.now(),
            metadata=metadata or {}
        )
        self.stages.append(record)
        return record

    def finish_stage(self, record: StageRecord, metadata: Optional[Dict[str, Any]] = None):
        """
        标记阶段为已完成。

        参数:
            record: record_stage 返回的 StageRecord
            metadata: 要合并的可选附加元数据
        """
        record.end_time = datetime.now()
        if metadata:
            record.metadata.update(metadata)

    def finish(self, metadata: Optional[Dict[str, Any]] = None):
        """
        标记整个追踪为已完成。

        参数:
            metadata: 整个追踪的可选元数据
        """
        self.end_time = datetime.now()
        if metadata:
            self.metadata.update(metadata)

    def log(self, component: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """
        记录追踪日志消息。

        参数:
            component: 组件名称（例如 "azure_vision_llm"）
            message: 日志消息
            metadata: 可选的附加元数据
        """
        # 简单实现：将日志存储在元数据中
        if "logs" not in self.metadata:
            self.metadata["logs"] = []

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "message": message
        }
        if metadata:
            log_entry["metadata"] = metadata

        self.metadata["logs"].append(log_entry)

    @property
    def total_duration_ms(self) -> Optional[float]:
        """计算总追踪持续时间（毫秒）"""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    def elapsed_ms(self, stage_name: Optional[str] = None) -> float:
        """
        获取已用时间（毫秒）。

        参数:
            stage_name: 可选的阶段名称。如果提供，返回该阶段的持续时间。
                       如果为 None，返回总已用时间。

        返回:
            已用时间（毫秒），如果未找到阶段或未完成则返回 0
        """
        if stage_name is None:
            # 返回总已用时间
            if self.end_time:
                delta = self.end_time - self.start_time
            else:
                delta = datetime.now() - self.start_time
            return delta.total_seconds() * 1000

        # 查找阶段并返回其持续时间
        for stage in self.stages:
            if stage.stage_name == stage_name:
                if stage.duration_ms is not None:
                    return stage.duration_ms
                # 阶段尚未完成，计算当前已用时间
                delta = datetime.now() - stage.start_time
                return delta.total_seconds() * 1000

        return 0.0  # 未找到阶段

    def to_dict(self) -> Dict[str, Any]:
        """
        将追踪转换为字典以便序列化。

        返回:
            包含 trace_type、时间戳和阶段的追踪字典表示
        """
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.start_time.isoformat(),
            "finished_at": self.end_time.isoformat() if self.end_time else None,
            "total_elapsed_ms": self.total_duration_ms,
            "metadata": self.metadata,
            "stages": [
                {
                    "stage_name": stage.stage_name,
                    "start_time": stage.start_time.isoformat(),
                    "end_time": stage.end_time.isoformat() if stage.end_time else None,
                    "duration_ms": stage.duration_ms,
                    "metadata": stage.metadata
                }
                for stage in self.stages
            ]
        }
