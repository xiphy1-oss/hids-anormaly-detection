"""
BERT 임베딩 생성 모듈
"""
import torch
from transformers import pipeline


class BERTEmbedder:
    """BERT 임베딩 생성 클래스"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Args:
            model_name: BERT 모델 이름
        """
        self.model_name = model_name
        self.extractor = pipeline(
            "feature-extraction",
            model=model_name,
            tokenizer=model_name,
            return_tensors=True,
        )
    
    @staticmethod
    def mean_pool(token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        평균 풀링을 수행하여 문장 임베딩 생성
        
        Args:
            token_embeddings: (batch, seq_len, hidden_size) 형태의 토큰 임베딩
        
        Returns:
            (batch, hidden_size) 형태의 문장 임베딩
        """
        return token_embeddings.mean(dim=1)
    
    def embed(self, text: str) -> torch.Tensor:
        """
        단일 텍스트에 대한 임베딩 생성
        
        Args:
            text: 입력 텍스트
        
        Returns:
            (hidden_size,) 형태의 임베딩 벡터
        """
        outputs = self.extractor(text, padding=True, truncation=True)
        pooled = self.mean_pool(outputs)
        return pooled.squeeze(0)
    
    def embed_batch(self, texts: list) -> torch.Tensor:
        """
        배치 텍스트에 대한 임베딩 생성
        
        Args:
            texts: 텍스트 리스트
        
        Returns:
            (batch_size, hidden_size) 형태의 임베딩 텐서
        """
        embeddings_list = []
        for text in texts:
            embedding = self.embed(text)
            embeddings_list.append(embedding)
        return torch.stack(embeddings_list)
