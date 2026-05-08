from typing import List
from .base_embedding import BaseEmbedding
from openai import AzureOpenAI

class AzureEmbedding(BaseEmbedding):
    def __init__(self, api_key: str, azure_endpoint: str, deployment_name: str, api_version: str = "2024-02-01", **kwargs):
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_deployment=deployment_name
        )
        self.deployment_name = deployment_name
        self.batch_size = kwargs.get("batch_size", 100)

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        embeddings = []

        # 按批次处理
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Azure OpenAI要求非空字符串，过滤掉空的
            filtered_batch = [(idx, text) for idx, text in enumerate(batch) if text.strip()]
            if not filtered_batch:
                # 如果整个批次都是空的，添加零向量
                embeddings.extend([[0.0] * 1536 for _ in range(len(batch))])
                continue

            filtered_texts = [item[1] for item in filtered_batch]

            try:
                response = self.client.embeddings.create(
                    input=filtered_texts,
                    model=self.deployment_name  # 在Azure中，model参数是部署名称
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
                        # Azure默认维度通常是1536，但可以配置为不同的模型
                        embeddings.append([0.0] * 1536)
            except Exception as e:
                # 如果API调用失败，为当前批次的所有文本返回零向量
                print(f"Error calling Azure OpenAI API: {str(e)}")
                embeddings.extend([[0.0] * 1536 for _ in range(len(batch))])

        return embeddings