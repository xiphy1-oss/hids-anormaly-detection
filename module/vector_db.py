"""
FAISS Vector Database 모듈

임베딩을 FAISS를 사용하여 저장하고 검색하는 클래스
"""
import faiss
import numpy as np
import json
import os
import torch
from typing import List, Dict, Optional, Union


class EmbeddingVectorDB:
    """
    FAISS를 사용하여 임베딩을 Vector DB에 저장하고 검색하는 클래스
    """
    
    def __init__(
        self, 
        collection_name: str = "hids_embeddings", 
        persist_directory: str = "./faiss_db", 
        index_type: str = "flat"
    ):
        """
        Args:
            collection_name: 컬렉션 이름
            persist_directory: 데이터베이스 저장 디렉토리
            index_type: 인덱스 타입 ("flat", "ivf", "cosine")
                - "flat": 완전 검색 인덱스 (L2 거리 사용, 정확하지만 느림)
                - "ivf": IVF 인덱스 (L2 거리 사용, 빠르지만 근사 검색)
                - "cosine": 코사인 유사도 인덱스 (내적 사용, 벡터 정규화)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.index_type = index_type
        self.index = None
        self.metadata = []
        self.ids = []
        self.embedding_dim = None
        
        # 저장 디렉토리 생성
        os.makedirs(persist_directory, exist_ok=True)
        
        # 파일 경로 설정
        self.index_path = os.path.join(persist_directory, f"{collection_name}.index")
        self.metadata_path = os.path.join(persist_directory, f"{collection_name}_metadata.json")
        self.ids_path = os.path.join(persist_directory, f"{collection_name}_ids.json")
        
        # 기존 인덱스가 있으면 로드
        if os.path.exists(self.index_path):
            self._load_index()
            print(f"기존 인덱스 '{collection_name}'을 불러왔습니다. (임베딩 수: {self.index.ntotal})")
        else:
            print(f"새 인덱스 '{collection_name}'을 생성합니다.")
    
    def _load_index(self):
        """기존 인덱스 로드"""
        self.index = faiss.read_index(self.index_path)
        self.embedding_dim = self.index.d
        
        # 메타데이터와 ID 로드
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        if os.path.exists(self.ids_path):
            with open(self.ids_path, 'r', encoding='utf-8') as f:
                self.ids = json.load(f)
    
    def _create_index(self, embedding_dim: int):
        """FAISS 인덱스 생성"""
        self.embedding_dim = embedding_dim
        
        if self.index_type == "flat":
            # 완전 검색 인덱스 (L2 거리 사용, 정확하지만 느림)
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif self.index_type == "ivf":
            # IVF 인덱스 (L2 거리 사용, 빠르지만 근사 검색)
            nlist = 100  # 클러스터 수
            quantizer = faiss.IndexFlatL2(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        elif self.index_type == "cosine":
            # 코사인 유사도 인덱스 (내적 사용, 벡터 정규화 필요)
            # IndexFlatIP는 내적을 사용하므로, 정규화된 벡터의 내적 = 코사인 유사도
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            raise ValueError(
                f"지원하지 않는 인덱스 타입: {self.index_type}. "
                f"사용 가능한 타입: 'flat', 'ivf', 'cosine'"
            )
    
    def save_embeddings(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        metadata_list: List[Dict],
        ids: Optional[List[str]] = None
    ):
        """
        임베딩을 Vector DB에 저장
        
        Args:
            embeddings: 임베딩 배열 또는 텐서 (n_samples, embedding_dim)
            metadata_list: 각 임베딩에 대한 메타데이터 리스트
            ids: 각 임베딩의 고유 ID 리스트 (None이면 자동 생성)
        """
        # 텐서를 numpy 배열로 변환
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy().astype('float32')
        else:
            embeddings_np = np.array(embeddings, dtype='float32')
        
        n_samples, embedding_dim = embeddings_np.shape
        
        # ID가 없으면 자동 생성
        if ids is None:
            start_idx = len(self.ids)
            ids = [f"{self.collection_name}_{start_idx + i}" for i in range(n_samples)]
        
        # 인덱스가 없으면 생성
        if self.index is None:
            self._create_index(embedding_dim)
        elif self.index.d != embedding_dim:
            raise ValueError(
                f"임베딩 차원이 일치하지 않습니다. "
                f"기존: {self.index.d}, 새로운: {embedding_dim}"
            )
        
        # 코사인 유사도 인덱스인 경우 벡터 정규화 필요
        if self.index_type == "cosine":
            # L2 정규화: 각 벡터를 단위 벡터로 변환
            faiss.normalize_L2(embeddings_np)
        
        # IVF 인덱스인 경우 훈련 필요
        if self.index_type == "ivf" and not self.index.is_trained:
            print("IVF 인덱스 훈련 중...")
            self.index.train(embeddings_np)
        
        # 임베딩 추가
        self.index.add(embeddings_np)
        
        # 메타데이터와 ID 저장
        processed_metadata = []
        for meta in metadata_list:
            processed_meta = {}
            for key, value in meta.items():
                # JSON 직렬화 가능한 타입만 저장
                if isinstance(value, (str, int, float, bool, type(None))):
                    processed_meta[key] = value
                else:
                    # 복잡한 객체는 JSON 문자열로 변환
                    processed_meta[key] = json.dumps(value, ensure_ascii=False)
            processed_metadata.append(processed_meta)
        
        self.metadata.extend(processed_metadata)
        self.ids.extend(ids)
        
        # 인덱스와 메타데이터 저장
        self._save_index()
        
        print(f"총 {n_samples}개의 임베딩이 저장되었습니다. (전체: {self.index.ntotal}개)")
    
    def _save_index(self):
        """인덱스와 메타데이터를 파일로 저장"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        with open(self.ids_path, 'w', encoding='utf-8') as f:
            json.dump(self.ids, f, ensure_ascii=False, indent=2)
    
    def search_similar(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        n_results: int = 10,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        유사한 임베딩 검색
        
        Args:
            query_embedding: 검색할 임베딩 벡터
            n_results: 반환할 결과 개수
            where: 메타데이터 필터 조건 (FAISS는 기본적으로 필터링 미지원, 후처리로 처리)
        
        Returns:
            검색 결과 딕셔너리
            {
                'ids': [[id1, id2, ...]],  # 각 쿼리별 결과 ID 리스트
                'distances': [[dist1, dist2, ...]],  # 각 쿼리별 거리/유사도 리스트
                    # index_type="flat" 또는 "ivf": L2 거리 (작을수록 유사, 0에 가까울수록 유사)
                    # index_type="cosine": 코사인 유사도 (클수록 유사, 1에 가까울수록 유사, 범위: -1 ~ 1)
                'metadatas': [[meta1, meta2, ...]]  # 각 쿼리별 메타데이터 리스트
            }
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("인덱스가 비어있습니다. 먼저 임베딩을 저장해주세요.")
        
        # 텐서를 numpy 배열로 변환
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu().numpy().astype('float32')
        else:
            query_embedding = np.array(query_embedding, dtype='float32')
        
        # 단일 벡터인 경우 2D로 변환
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # 코사인 유사도 인덱스인 경우 쿼리 벡터도 정규화 필요
        if self.index_type == "cosine":
            faiss.normalize_L2(query_embedding)
        
        # 검색 (더 많은 결과를 가져와서 필터링 가능하도록)
        k = min(n_results * 3 if where else n_results, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        # 코사인 유사도인 경우: 내적 값을 코사인 거리로 변환 (1 - 코사인 유사도)
        # 내적이 클수록 유사하므로, 거리로 변환하려면 1 - 내적 사용
        # 하지만 일반적으로 코사인 유사도 자체를 사용하므로, 여기서는 내적 값을 그대로 사용
        # 필요시 distances를 1 - distances로 변환 가능
        
        # 결과 처리
        result_ids = []
        result_distances = []
        result_metadatas = []
        
        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            ids_row = []
            dists_row = []
            metadatas_row = []
            
            for dist, idx in zip(dist_row, idx_row):
                if idx == -1:  # FAISS에서 유효하지 않은 인덱스
                    continue
                
                # 메타데이터 필터링
                if where:
                    metadata = self.metadata[idx]
                    match = True
                    for key, value in where.items():
                        if metadata.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue
                
                ids_row.append(self.ids[idx])
                dists_row.append(float(dist))
                metadatas_row.append(self.metadata[idx])
                
                if len(ids_row) >= n_results:
                    break
            
            result_ids.append(ids_row)
            result_distances.append(dists_row)
            result_metadatas.append(metadatas_row)
        
        return {
            'ids': result_ids,
            'distances': result_distances,
            'metadatas': result_metadatas
        }
    
    def get_all_embeddings(self) -> Dict:
        """
        모든 임베딩 가져오기 (FAISS는 벡터만 저장하므로 메타데이터만 반환)
        
        Returns:
            모든 메타데이터와 ID
        """
        return {
            'ids': self.ids,
            'metadatas': self.metadata,
            'total': len(self.ids)
        }
    
    def get_all_embeddings_vectors(self) -> np.ndarray:
        """
        Vector DB에서 모든 임베딩 벡터를 가져오기
        
        Returns:
            모든 임베딩 벡터 (n_samples, embedding_dim)
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("인덱스가 비어있습니다. 먼저 임베딩을 저장해주세요.")
        
        # FAISS에서 모든 벡터 재구성
        total = self.index.ntotal
        all_vectors = []
        
        for i in range(total):
            vector = self.index.reconstruct(i)
            all_vectors.append(vector)
        
        return np.array(all_vectors, dtype='float32')
    
    def delete_collection(self):
        """컬렉션 삭제"""
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        if os.path.exists(self.ids_path):
            os.remove(self.ids_path)
        
        self.index = None
        self.metadata = []
        self.ids = []
        
        print(f"컬렉션 '{self.collection_name}'이 삭제되었습니다.")
    
    def get_collection_info(self) -> Dict:
        """컬렉션 정보 반환"""
        total = self.index.ntotal if self.index else 0
        return {
            "collection_name": self.collection_name,
            "total_embeddings": total,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "persist_directory": self.persist_directory
        }
