"""
시퀀스 및 N-gram 처리 모듈
"""
import json
import os
import numpy as np
from collections import Counter
from typing import Optional, Tuple, List, Dict, Any


class SequenceMatcher:
    """원본 시퀀스와 클러스터 레이블을 매칭하는 클래스"""
    
    @staticmethod
    def match_sequence_with_clusters(
        original_sequence: List[Any],
        cluster_labels: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        원본 시퀀스와 클러스터링 결과를 매칭하여 각 항목에 클러스터 ID를 추가
        
        Args:
            original_sequence: 원본 데이터 시퀀스 (리스트)
            cluster_labels: 클러스터링 결과 레이블 (numpy array)
        
        Returns:
            각 항목이 {'data': 원본데이터, 'cluster_id': 클러스터ID, 'index': 인덱스} 형태인 리스트
        """
        if len(original_sequence) != len(cluster_labels):
            raise ValueError(
                f"원본 시퀀스 길이({len(original_sequence)})와 클러스터 레이블 길이({len(cluster_labels)})가 일치하지 않습니다."
            )
        
        if not isinstance(cluster_labels, np.ndarray):
            cluster_labels = np.array(cluster_labels)
        
        matched_sequence = []
        for idx, (data, cluster_id) in enumerate(zip(original_sequence, cluster_labels)):
            matched_item = {
                'index': idx,
                'cluster_id': int(cluster_id),
                'data': data
            }
            matched_sequence.append(matched_item)
        
        return matched_sequence
    
    @staticmethod
    def load_original_sequence_from_file(data_file: str = 'data/tr_normal_tracee.json') -> List[Dict]:
        """
        원본 JSON Lines 파일에서 원본 시퀀스를 로드
        
        Args:
            data_file: 원본 데이터 파일 경로
        
        Returns:
            원본 데이터 시퀀스
        """
        original_sequence = []
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        json_obj = json.loads(line)
                        original_sequence.append(json_obj)
                    except json.JSONDecodeError:
                        continue
        
        return original_sequence
    
    @staticmethod
    def load_cluster_labels_from_mapping(
        mapping_file: str = 'cluster_mapping.json',
        convert_noise_to_cluster: bool = True
    ) -> np.ndarray:
        """
        cluster_mapping.json에서 클러스터 레이블을 재구성
        
        Args:
            mapping_file: cluster_mapping.json 파일 경로
            convert_noise_to_cluster: True이면 노이즈(-1)를 별도의 클러스터 ID로 변환
        
        Returns:
            인덱스 순서대로 정렬된 클러스터 레이블
        """
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        index_to_cluster = {}
        cluster_ids_set = set()
        
        for cluster_id, cluster_info in mapping.get('clusters', {}).items():
            cluster_id_int = int(cluster_id) if cluster_id != '-1' else -1
            
            for item in cluster_info.get('data', []):
                idx = item.get('index')
                if idx is not None:
                    index_to_cluster[idx] = cluster_id_int
                    if cluster_id_int != -1:
                        cluster_ids_set.add(cluster_id_int)
        
        if not index_to_cluster:
            raise ValueError("클러스터 매핑에 데이터가 없습니다.")
        
        max_index = max(index_to_cluster.keys())
        
        if convert_noise_to_cluster:
            if cluster_ids_set:
                max_cluster_id = max(cluster_ids_set)
                next_cluster_id = max_cluster_id + 1
            else:
                next_cluster_id = 0
            
            noise_indices = [idx for idx, cid in index_to_cluster.items() if cid == -1]
            noise_count = len(noise_indices)
            
            if noise_count > 0:
                for i, idx in enumerate(noise_indices):
                    index_to_cluster[idx] = next_cluster_id + i
                
                if noise_count == 1:
                    print(f"   노이즈 포인트(-1) 1개를 클러스터 ID {next_cluster_id}로 변환했습니다.")
                else:
                    print(f"   노이즈 포인트(-1) {noise_count}개를 클러스터 ID {next_cluster_id}~{next_cluster_id + noise_count - 1}로 변환했습니다.")
            else:
                print("   노이즈 포인트가 없습니다.")
        
        cluster_labels = np.zeros(max_index + 1, dtype=int)
        for idx, cluster_id in index_to_cluster.items():
            cluster_labels[idx] = cluster_id
        
        return cluster_labels
    
    @staticmethod
    def save_matched_sequence(matched_sequence: List[Dict], output_file: str = 'matched_sequence.json'):
        """
        matched_sequence를 JSON 파일로 저장
        
        Args:
            matched_sequence: match_sequence_with_clusters 함수의 결과
            output_file: 저장할 파일 경로
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(matched_sequence, f, ensure_ascii=False, indent=2)
        
        print(f"matched_sequence가 {output_file}에 저장되었습니다.")
        print(f"총 {len(matched_sequence)}개의 항목이 저장되었습니다.")


class NGramGenerator:
    """N-gram 생성 및 분석 클래스"""
    
    def __init__(self, n: int = 2):
        """
        Args:
            n: n-gram의 n 값 (예: 2=bigram, 3=trigram, 5=5-gram)
        """
        self.n = n
        self.ngrams_ = None
        self.cluster_sequence_ = None
    
    def create_from_file(self, matched_sequence_file: str = 'matched_sequence.json') -> Tuple[List[Tuple], List[int]]:
        """
        matched_sequence.json에서 cluster_id를 추출하여 n-gram 시퀀스를 생성
        
        Args:
            matched_sequence_file: matched_sequence.json 파일 경로
        
        Returns:
            ngrams: n-gram 시퀀스 리스트 (각 항목은 n개의 cluster_id 튜플)
            cluster_sequence: 원본 클러스터 시퀀스
        """
        with open(matched_sequence_file, 'r', encoding='utf-8') as f:
            matched_sequence = json.load(f)
        
        cluster_sequence = []
        for item in sorted(matched_sequence, key=lambda x: x['index']):
            cluster_sequence.append(item['cluster_id'])
        
        ngrams = []
        for i in range(len(cluster_sequence) - self.n + 1):
            ngram = tuple(cluster_sequence[i:i+self.n])
            ngrams.append(ngram)
        
        self.ngrams_ = ngrams
        self.cluster_sequence_ = cluster_sequence
        
        return ngrams, cluster_sequence
    
    def create_from_sequence(self, cluster_sequence: List[int]) -> List[Tuple]:
        """
        클러스터 시퀀스로부터 n-gram 생성
        
        Args:
            cluster_sequence: 클러스터 ID 시퀀스
        
        Returns:
            n-gram 시퀀스 리스트
        """
        ngrams = []
        for i in range(len(cluster_sequence) - self.n + 1):
            ngram = tuple(cluster_sequence[i:i+self.n])
            ngrams.append(ngram)
        
        self.ngrams_ = ngrams
        self.cluster_sequence_ = cluster_sequence
        
        return ngrams
    
    def analyze(self, ngrams: Optional[List[Tuple]] = None, top_k: int = 10) -> Dict[str, Any]:
        """
        n-gram 통계 분석
        
        Args:
            ngrams: n-gram 리스트 (None이면 self.ngrams_ 사용)
            top_k: 출력할 상위 k개 n-gram
        
        Returns:
            통계 정보 딕셔너리
        """
        if ngrams is None:
            ngrams = self.ngrams_
        
        if ngrams is None:
            raise ValueError("ngrams가 없습니다. 먼저 create_from_file 또는 create_from_sequence를 실행하세요.")
        
        ngram_counts = Counter(ngrams)
        
        stats = {
            'total_ngrams': len(ngrams),
            'unique_ngrams': len(ngram_counts),
            'top_ngrams': ngram_counts.most_common(top_k),
            'ngram_counts': dict(ngram_counts)
        }
        
        return stats
    
    def print_statistics(self, ngrams: Optional[List[Tuple]] = None, top_k: int = 10):
        """
        n-gram 통계 정보 출력
        
        Args:
            ngrams: n-gram 리스트 (None이면 self.ngrams_ 사용)
            top_k: 출력할 상위 k개 n-gram
        """
        if ngrams is None:
            ngrams = self.ngrams_
        
        if ngrams is None:
            raise ValueError("ngrams가 없습니다. 먼저 create_from_file 또는 create_from_sequence를 실행하세요.")
        
        stats = self.analyze(ngrams, top_k)
        
        print(f"고유 {self.n}-gram 수: {stats['unique_ngrams']}")
        print(f"\n상위 {top_k}개 빈도 {self.n}-gram:")
        for i, (ngram, count) in enumerate(stats['top_ngrams'], 1):
            print(f"  {i:2d}. {ngram}: {count:4d}회 ({count/len(ngrams)*100:.2f}%)")
