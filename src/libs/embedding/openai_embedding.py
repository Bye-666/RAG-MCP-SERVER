from typing import List
from .base_embedding import BaseEmbedding
from openai import OpenAI
import time

class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", api_base: str = None, **kwargs):
        client_kwargs = {
            "api_key": api_key,
            "timeout": kwargs.get("timeout", 60.0),  # 默认 60 秒超时
            "max_retries": kwargs.get("max_retries", 3)  # 默认重试 3 次
        }
        if api_base:
            client_kwargs["base_url"] = api_base
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.batch_size = kwargs.get("batch_size", 100)  # OpenAI 推荐最大为 2048，但小批量更稳定

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        embeddings = []

        # 按批次处理
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # OpenAI API 要求非空字符串，过滤掉空的
            filtered_batch = [(idx, text) for idx, text in enumerate(batch) if text.strip()]
            if not filtered_batch:
                # 如果整个批次都是空的，添加零向量
                embeddings.extend([[0.0] * 1536 for _ in range(len(batch))])
                continue

            filtered_texts = [item[1] for item in filtered_batch]

            try:
                response = self.client.embeddings.create(
                    input=filtered_texts,
                    model=self.model
                )

                # 将结果映射回原始位置
                batch_embeddings = [[float(x) for x in item.embedding] for item in response.data]

                # 为原始批次中的每个文本分配 embedding（空文本使用零向量）
                embedding_idx = 0
                for text in batch:
                    if text.strip():  # 非空文本
                        embeddings.append(batch_embeddings[embedding_idx])
                        embedding_idx += 1
                    else:  # 空文本
                        # 根据模型决定向量维数
                        dim = 1536 if "3-small" in self.model else 3072 if "3-large" in self.model else 1536
                        embeddings.append([0.0] * dim)
            except Exception as e:
                # 如果 API 调用失败，为当前批次的所有文本返回零向量
                error_type = type(e).__name__
                print(f"⚠️ 调用 Embedding API 时出错 ({error_type}): {str(e)}")
                print(f"   批次大小: {len(batch)}, 模型: {self.model}")
                print(f"   将使用零向量作为后备方案")
                dim = 1536 if "3-small" in self.model else 3072 if "3-large" in self.model else 1536
                embeddings.extend([[0.0] * dim for _ in range(len(batch))])

        return embeddings