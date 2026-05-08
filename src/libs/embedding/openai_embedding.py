from typing import List
from .base_embedding import BaseEmbedding
from openai import OpenAI

class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", **kwargs):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = kwargs.get("batch_size", 100)  # OpenAI推荐最大为2048，但小批量更稳定

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        embeddings = []

        # 按批次处理
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # OpenAI API要求非空字符串，过滤掉空的
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

                # 为原始批次中的每个文本分配embedding（空文本使用零向量）
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
                # 如果API调用失败，为当前批次的所有文本返回零向量
                print(f"Error calling OpenAI API: {str(e)}")
                dim = 1536 if "3-small" in self.model else 3072 if "3-large" in self.model else 1536
                embeddings.extend([[0.0] * dim for _ in range(len(batch))])

        return embeddings