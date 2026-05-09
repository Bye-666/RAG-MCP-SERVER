"""
追踪服务层，解析和查询 traces.jsonl。

为 Dashboard 提供追踪数据访问接口。
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class TraceService:
    """追踪服务，读取和解析 traces.jsonl"""

    def __init__(self, log_dir: str = "logs"):
        """
        初始化追踪服务。

        Args:
            log_dir: 日志目录路径
        """
        self.log_dir = Path(log_dir)
        self.traces_file = self.log_dir / "traces.jsonl"

    def list_traces(
        self,
        trace_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        列出所有追踪记录。

        Args:
            trace_type: 可选的追踪类型过滤（"ingestion" 或 "query"）
            limit: 可选的返回数量限制

        Returns:
            追踪记录列表，按时间倒序排列
        """
        if not self.traces_file.exists():
            return []

        traces = []
        try:
            with open(self.traces_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        trace = json.loads(line)

                        # 过滤追踪类型
                        if trace_type and trace.get("trace_type") != trace_type:
                            continue

                        traces.append(trace)
                    except json.JSONDecodeError:
                        continue  # 跳过无效行

        except Exception:
            return []

        # 按时间倒序排列
        traces.sort(key=lambda t: t.get("started_at", ""), reverse=True)

        # 限制返回数量
        if limit:
            traces = traces[:limit]

        return traces

    def get_trace_by_id(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 trace_id 获取追踪记录。

        Args:
            trace_id: 追踪 ID

        Returns:
            追踪记录字典，如果未找到则返回 None
        """
        if not self.traces_file.exists():
            return None

        try:
            with open(self.traces_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        trace = json.loads(line)
                        if trace.get("trace_id") == trace_id:
                            return trace
                    except json.JSONDecodeError:
                        continue

        except Exception:
            return None

        return None

    def get_ingestion_traces(self, limit: Optional[int] = 50) -> List[Dict[str, Any]]:
        """
        获取 Ingestion 追踪记录。

        Args:
            limit: 返回数量限制

        Returns:
            Ingestion 追踪记录列表
        """
        return self.list_traces(trace_type="ingestion", limit=limit)

    def get_query_traces(self, limit: Optional[int] = 50) -> List[Dict[str, Any]]:
        """
        获取 Query 追踪记录。

        Args:
            limit: 返回数量限制

        Returns:
            Query 追踪记录列表
        """
        return self.list_traces(trace_type="query", limit=limit)

    def search_traces(
        self,
        keyword: str,
        trace_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索追踪记录。

        Args:
            keyword: 搜索关键词
            trace_type: 可选的追踪类型过滤

        Returns:
            匹配的追踪记录列表
        """
        all_traces = self.list_traces(trace_type=trace_type)

        if not keyword:
            return all_traces

        keyword_lower = keyword.lower()
        matched = []

        for trace in all_traces:
            # 搜索 trace_id
            if keyword_lower in trace.get("trace_id", "").lower():
                matched.append(trace)
                continue

            # 搜索 metadata
            metadata = trace.get("metadata", {})
            metadata_str = json.dumps(metadata).lower()
            if keyword_lower in metadata_str:
                matched.append(trace)
                continue

        return matched

    def get_trace_stats(self) -> Dict[str, Any]:
        """
        获取追踪统计信息。

        Returns:
            统计信息字典
        """
        all_traces = self.list_traces()

        ingestion_count = sum(1 for t in all_traces if t.get("trace_type") == "ingestion")
        query_count = sum(1 for t in all_traces if t.get("trace_type") == "query")

        return {
            "total_traces": len(all_traces),
            "ingestion_traces": ingestion_count,
            "query_traces": query_count
        }
