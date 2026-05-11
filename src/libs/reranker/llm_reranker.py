import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from .base_reranker import BaseReranker
from ...core.trace import TraceContext


class LLMReranker(BaseReranker):
    """基于 LLM 的重排序器，使用结构化输出"""

    def __init__(self, llm, prompt_path: Optional[str] = None, prompt_text: Optional[str] = None):
        """
        初始化 LLM 重排序器

        参数:
            llm: 具有 chat() 方法的 LLM 实例
            prompt_path: 提示模板文件路径（默认: config/prompts/rerank.txt）
            prompt_text: 可选的提示文本覆盖（用于测试）
        """
        self.llm = llm

        if prompt_text is not None:
            self.prompt_template = prompt_text
        else:
            if prompt_path is None:
                prompt_path = "config/prompts/rerank.txt"

            prompt_file = Path(prompt_path)
            if not prompt_file.exists():
                raise FileNotFoundError(f"重排序提示文件未找到: {prompt_path}")

            self.prompt_template = prompt_file.read_text(encoding='utf-8')

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[TraceContext] = None
    ) -> List[Dict[str, Any]]:
        """
        使用 LLM 对候选项进行重排序

        参数:
            query: 用户查询
            candidates: 候选项字典列表（必须有 'id' 和 'text' 字段）
            trace: 可选的跟踪上下文

        返回:
            带有更新分数的重排序候选项列表

        异常:
            ValueError: 如果候选项格式无效或 LLM 输出不匹配架构
            RuntimeError: 如果 LLM 调用失败（允许回退到原始排序）
        """
        if not candidates:
            return []

        if trace:
            trace.log("llm_reranker", f"重排序 {len(candidates)} 个候选项")

        # 验证候选项具有必需字段
        for i, cand in enumerate(candidates):
            if 'id' not in cand:
                raise ValueError(f"候选项 {i} 缺少 'id' 字段")
            if 'text' not in cand:
                raise ValueError(f"候选项 {i} 缺少 'text' 字段")

        # 为提示构建候选项列表
        candidate_lines = []
        for cand in candidates:
            candidate_lines.append(f"ID: {cand['id']}\nText: {cand['text']}\n")

        candidates_text = "\n".join(candidate_lines)

        # 格式化提示
        prompt = self.prompt_template.format(
            query=query,
            candidates=candidates_text
        )

        # 调用 LLM
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat(messages)

            if trace:
                trace.log("llm_reranker", f"LLM 响应: {response[:200]}...")

        except Exception as e:
            raise RuntimeError(f"LLM 重排序失败: {str(e)}") from e

        # 解析结构化输出
        try:
            # 从响应中提取 JSON（处理 markdown 代码块）
            response_clean = response.strip()
            if response_clean.startswith("```"):
                # 移除 markdown 代码块标记
                lines = response_clean.split("\n")
                response_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else response_clean

            result = json.loads(response_clean)

            if "ranked_ids" not in result:
                raise ValueError("LLM 输出缺少 'ranked_ids' 字段")

            ranked_ids = result["ranked_ids"]

            if not isinstance(ranked_ids, list):
                raise ValueError(f"'ranked_ids' 必须是列表，得到 {type(ranked_ids).__name__}")

            # 验证所有 ID 都存在
            candidate_ids = {cand['id'] for cand in candidates}
            ranked_id_set = set(ranked_ids)

            if ranked_id_set != candidate_ids:
                missing = candidate_ids - ranked_id_set
                extra = ranked_id_set - candidate_ids
                error_parts = []
                if missing:
                    error_parts.append(f"缺少的 ID: {missing}")
                if extra:
                    error_parts.append(f"额外的 ID: {extra}")
                raise ValueError(f"排序的 ID 与候选项不匹配 ({', '.join(error_parts)})")

        except json.JSONDecodeError as e:
            raise ValueError(f"LLM 输出不是有效的 JSON: {str(e)}") from e
        except (KeyError, ValueError) as e:
            raise ValueError(f"LLM 输出不匹配预期架构: {str(e)}") from e

        # 构建重排序结果
        id_to_candidate = {cand['id']: cand for cand in candidates}
        reranked = []

        for rank, cand_id in enumerate(ranked_ids):
            cand = id_to_candidate[cand_id].copy()
            cand['rerank_score'] = 1.0 / (rank + 1)  # 倒数排名作为分数
            reranked.append(cand)

        if trace:
            trace.log("llm_reranker", f"重排序为: {[c['id'] for c in reranked]}")

        return reranked
