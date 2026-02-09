"""
학습된 모델을 사용하여 공격 데이터를 검증하는 모듈

학습된 Bloom Filter와 Clustering 모델을 로드하여
공격 데이터에 대한 이상 탐지를 수행합니다.
"""

import os
import sys
import json
import pickle
import argparse
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader import TraceeDataset
from utility import Config, TextFilter
from embedder import BERTEmbedder
from clustering import Clusterer
from sequence import NGramGenerator
from bloomfilter import NGramBloomFilter
from vector_db import EmbeddingVectorDB

    # 생성된 embedding 들을 class로 분류하는 함수 작성
def classify_embeddings_by_class(
    similar_embeddings_list: List[List[dict]],
    model_data_dir: str = "model_data",
    config: Optional[Config] = None
) -> List[dict]:
    """
    생성된 embedding 들을 class로 분류하는 함수
    
    각 embedding에 대해:
    1. 가장 유사한 vector를 vector db에서 검색 (top-1)
    2. top-1의 class의 index를 기준으로 cluster_centroid, cluster_max_distance 확인
    3. distance가 max_distance 내에 있으면 top-1의 class로 분류
    4. max_distance 바깥이라면 CLASS_THRESHOLD 값과 비교하여 unknown 처리
    
    Args:
        similar_embeddings_list: 각 공격 임베딩에 대한 유사한 임베딩 리스트
        model_data_dir: 모델 데이터 디렉토리 경로
        config: 설정 객체 (None이면 새로 생성)
    
    Returns:
        분류된 결과 리스트
    """
    # Config 로드
    if config is None:
        config = Config()
    
    # 클러스터 레이블 로드
    cluster_labels_path = os.path.join(model_data_dir, "cluster_labels.npy")
    if not os.path.exists(cluster_labels_path):
        raise FileNotFoundError(f"클러스터 레이블 파일을 찾을 수 없습니다: {cluster_labels_path}")
    
    cluster_labels = np.load(cluster_labels_path)
    print(f"✓ 클러스터 레이블 로드 완료: {len(cluster_labels)}개")
    
    # 클러스터 정보 로드
    cluster_info_path = os.path.join(model_data_dir, "cluster_info.json")
    if not os.path.exists(cluster_info_path):
        raise FileNotFoundError(f"클러스터 정보 파일을 찾을 수 없습니다: {cluster_info_path}")
    
    with open(cluster_info_path, 'r', encoding='utf-8') as f:
        cluster_info = json.load(f)
    
    # cluster_centroids와 cluster_max_distances 추출
    cluster_centroids = {}
    cluster_max_distances = {}
    
    if 'cluster_centroids' in cluster_info:
        cluster_centroids = {
            int(k): np.array(v) for k, v in cluster_info['cluster_centroids'].items()
        }
    
    if 'cluster_max_distances' in cluster_info:
        cluster_max_distances = {
            int(k): float(v) for k, v in cluster_info['cluster_max_distances'].items()
        }
    
    if not cluster_centroids or not cluster_max_distances:
        raise ValueError("cluster_info.json에 cluster_centroids 또는 cluster_max_distances가 없습니다.")
    
    print(f"✓ 클러스터 정보 로드 완료: {len(cluster_centroids)}개 클러스터")
    print(f"  - CLASS_THRESHOLD: {config.CLASS_THRESHOLD}")
    
    # 분류 결과 리스트
    classified_results = []
    
    # 각 공격 임베딩에 대해 분류 수행
    for idx, similar_embeddings in enumerate(similar_embeddings_list):
        if not similar_embeddings:
            # 검색 결과가 없는 경우
            classified_results.append({
                'top1_result': None,
                'cluster_id': None,
                'class': 'unknown',
                'distance': None,
                'max_distance': None,
                'within_max_distance': False
            })
            continue
        
        # Top-1 결과 가져오기
        top1_result = similar_embeddings[0]
        top1_similarity = top1_result['distance']  # 코사인 유사도 (1에 가까울수록 유사)
        top1_metadata = top1_result['metadata']
        
        # 코사인 유사도를 코사인 거리로 변환 (코사인 거리 = 1 - 코사인 유사도)
        # 코사인 유사도 범위: -1 ~ 1, 코사인 거리 범위: 0 ~ 2
        top1_distance = 1.0 - top1_similarity
        
        # metadata에서 index 가져오기
        top1_index = top1_metadata.get('index')
        if top1_index is None:
            # index가 없는 경우 unknown 처리
            classified_results.append({
                'top1_result': top1_result,
                'cluster_id': None,
                'class': 'unknown',
                'distance': top1_distance,
                'similarity': top1_similarity,
                'max_distance': None,
                'within_max_distance': False
            })
            continue
        
        # cluster_labels에서 해당 index의 클러스터 ID 찾기
        if top1_index >= len(cluster_labels):
            # 인덱스가 범위를 벗어난 경우
            classified_results.append({
                'top1_result': top1_result,
                'cluster_id': None,
                'class': 'unknown',
                'distance': top1_distance,
                'similarity': top1_similarity,
                'max_distance': None,
                'within_max_distance': False
            })
            continue
        
        cluster_id = int(cluster_labels[top1_index])
        
        # 노이즈 포인트(-1)도 하나의 클러스터로 처리
        # 해당 클러스터의 max_distance 가져오기
        if cluster_id not in cluster_max_distances:
            # 클러스터 정보가 없는 경우
            classified_results.append({
                'top1_result': top1_result,
                'cluster_id': cluster_id,
                'class': 'unknown',
                'distance': top1_distance,
                'similarity': top1_similarity,
                'max_distance': None,
                'within_max_distance': False
            })
            continue
        
        max_distance = cluster_max_distances[cluster_id]
        
        # 코사인 거리가 max_distance 내에 있는지 확인 (거리가 작을수록 유사)
        within_max_distance = top1_distance <= max_distance
        
        if within_max_distance:
            # max_distance 내에 있으면 해당 클러스터로 분류
            classified_class = cluster_id
        else:
            # max_distance 바깥이면 CLASS_THRESHOLD와 비교
            # 코사인 거리가 CLASS_THRESHOLD보다 크면 unknown 처리
            if top1_distance > config.CLASS_THRESHOLD:
                classified_class = 'unknown'
            else:
                # CLASS_THRESHOLD 이내이면 해당 클러스터로 분류 (경고 수준)
                classified_class = cluster_id
        
        classified_results.append({
            'top1_result': top1_result,
            'cluster_id': cluster_id,
            'class': classified_class,
            'distance': top1_distance,  # 코사인 거리
            'similarity': top1_similarity,  # 코사인 유사도 (참고용)
            'max_distance': max_distance,
            'within_max_distance': within_max_distance
        })
    
    print(f"\n✓ 분류 완료: {len(classified_results)}개 샘플")
    
    # 분류 결과 통계 출력
    class_counts = {}
    for result in classified_results:
        class_val = result['class']
        class_counts[class_val] = class_counts.get(class_val, 0) + 1
    
    print("\n분류 결과 통계:")
    for class_val, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  클래스 {class_val}: {count}개")
    
    return classified_results


def generate_ngram_and_check_bloom_filter(
    classified_results: List[Dict[str, Any]],
    trained_models: Dict[str, Any],
    model_data_dir: str = "model_data",
    config: Optional[Config] = None
) -> Dict[str, Any]:
    """
    분류된 class를 이용해 n-gram sequence를 생성하고 Bloom Filter에 멤버인지 확인하는 함수
    
    Args:
        classified_results: 분류된 결과 리스트 (각 항목은 'class' 필드를 포함)
        trained_models: 학습된 모델 딕셔너리 (bloom_filter 포함)
        model_data_dir: 모델 데이터 디렉토리 경로
        config: 설정 객체 (None이면 새로 생성)
    
    Returns:
        결과 딕셔너리:
        {
            'cluster_sequence': [class1, class2, ...],  # 클러스터 ID 시퀀스
            'ngrams': [(class1, class2), (class2, class3), ...],  # N-gram 시퀀스
            'bloom_filter_results': [True, False, ...],  # 각 N-gram의 Bloom Filter 멤버십
            'total_ngrams': int,  # 총 N-gram 수
            'matched_ngrams': int,  # Bloom Filter에 매칭된 N-gram 수
            'unmatched_ngrams': int,  # Bloom Filter에 매칭되지 않은 N-gram 수
            'anomaly_indices': [idx1, idx2, ...]  # 이상 탐지된 인덱스 리스트
        }
    """
    # Config 로드
    if config is None:
        config = Config()
    
    # Bloom Filter 가져오기
    bloom_filter = trained_models.get('bloom_filter')
    if bloom_filter is None:
        # trained_models에 없으면 직접 로드 시도
        bloom_filter_path = os.path.join(model_data_dir, "bloom_filter.pkl")
        if os.path.exists(bloom_filter_path):
            with open(bloom_filter_path, 'rb') as f:
                bloom_filter = pickle.load(f)
            print(f"  ✓ Bloom Filter 로드 완료: {bloom_filter_path}")
        else:
            raise FileNotFoundError(f"Bloom Filter 파일을 찾을 수 없습니다: {bloom_filter_path}")
    
    if bloom_filter is None:
        raise ValueError("Bloom Filter를 로드할 수 없습니다.")
    
    print(f"  ✓ Bloom Filter 준비 완료")
    
    # 분류된 class 시퀀스 추출
    cluster_sequence = []
    for result in classified_results:
        class_val = result.get('class')
        
        # 'unknown'인 경우 처리 방법 결정
        # 옵션 1: 'unknown'을 -999로 변환 (특별한 클러스터 ID)
        # 옵션 2: 'unknown'을 None으로 처리하고 제외
        # 옵션 3: 'unknown'을 -1로 변환 (노이즈 포인트로 처리)
        if class_val == 'unknown':
            # 노이즈 포인트로 처리 (-1)
            cluster_sequence.append(-1)
        elif class_val is None:
            # None인 경우도 -1로 처리
            cluster_sequence.append(-1)
        else:
            # 정수 클러스터 ID로 변환
            try:
                cluster_id = int(class_val)
                cluster_sequence.append(cluster_id)
            except (ValueError, TypeError):
                # 변환 실패 시 -1로 처리
                cluster_sequence.append(-1)
    
    print(f"  ✓ 클러스터 시퀀스 생성 완료: {len(cluster_sequence)}개 샘플")
    
    # N-gram 생성
    n_gram_size = config.N_GRAM_SIZE
    ngram_generator = NGramGenerator(n=n_gram_size)
    ngrams = ngram_generator.create_from_sequence(cluster_sequence)
    
    print(f"  ✓ N-gram 생성 완료: {len(ngrams)}개 (N={n_gram_size})")
    
    # Bloom Filter에서 각 N-gram 확인
    print(f"  ✓ Bloom Filter 확인 중...")
    bloom_filter_results = []
    
    for ngram in tqdm(ngrams, desc="Bloom Filter 확인", leave=False):
        # NGramBloomFilter의 check 메서드 사용
        is_member = bloom_filter.check(ngram)
        bloom_filter_results.append(is_member)
    
    # 통계 계산
    total_ngrams = len(ngrams)
    matched_ngrams = sum(bloom_filter_results)
    unmatched_ngrams = total_ngrams - matched_ngrams
    
    # 이상 탐지 인덱스 찾기 (Bloom Filter에 없는 N-gram의 시작 인덱스)
    anomaly_indices = []
    for i, is_member in enumerate(bloom_filter_results):
        if not is_member:
            # N-gram의 시작 인덱스 추가
            anomaly_indices.append(i)
    
    print(f"  ✓ Bloom Filter 확인 완료")
    print(f"    - 총 N-gram: {total_ngrams}개")
    print(f"    - 매칭된 N-gram: {matched_ngrams}개 ({matched_ngrams/total_ngrams*100:.2f}%)" if total_ngrams > 0 else "    - 매칭된 N-gram: 0개")
    print(f"    - 미매칭 N-gram: {unmatched_ngrams}개 ({unmatched_ngrams/total_ngrams*100:.2f}%)" if total_ngrams > 0 else "    - 미매칭 N-gram: 0개")
    print(f"    - 이상 탐지된 인덱스: {len(anomaly_indices)}개")
    
    # 샘플 결과 출력 (처음 10개만)
    if len(ngrams) > 0:
        print(f"\n  샘플 N-gram 결과 (처음 10개):")
        for i in range(min(10, len(ngrams))):
            ngram = ngrams[i]
            is_member = bloom_filter_results[i]
            status = "✓ 매칭" if is_member else "✗ 미매칭"
            print(f"    {i+1}. N-gram {ngram}: {status}")
    
    return {
        'cluster_sequence': cluster_sequence,
        'ngrams': ngrams,
        'bloom_filter_results': bloom_filter_results,
        'total_ngrams': total_ngrams,
        'matched_ngrams': matched_ngrams,
        'unmatched_ngrams': unmatched_ngrams,
        'anomaly_indices': anomaly_indices,
        'n_gram_size': n_gram_size
    }


def create_attack_data_loader(config: Config, text_filter: TextFilter, attack_file_path: Optional[str] = None):
    """
    공격 데이터 로더를 생성하는 함수
    
    Args:
        config: 설정 객체
        text_filter: 텍스트 필터 객체
        attack_file_path: 공격 데이터 파일 경로 (None이면 기본 경로 사용)
    
    Returns:
        attack_loader: 공격 데이터 로더
    """
    print("=" * 80)
    print("공격 데이터 로더 생성 시작")
    print("=" * 80)
    
    filter_func = lambda x: text_filter.filter(x)
    
    # 공격 데이터 파일 경로 설정
    if attack_file_path is None:
        # 기본 경로 사용
        data_dir = "../data"
        attack_file = os.path.join(data_dir, "attack_tracee.json")
    else:
        # 사용자가 제공한 경로 사용
        attack_file = attack_file_path
    
    print(f"\n사용할 데이터 파일: {attack_file}")
    
    # 데이터 로더 생성
    print("\n[공격 데이터 로더 생성 중...]")
    try:
        if not os.path.exists(attack_file):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {attack_file}")
        
        attack_dataset = TraceeDataset(attack_file, filter_func=filter_func)
        attack_loader = DataLoader(
            attack_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,  # 추론 시에는 셔플하지 않음
            num_workers=0,
            collate_fn=lambda x: x
        )
        print(f"✓ 공격 데이터 로더 생성 완료: {len(attack_dataset)}개 샘플")
    except FileNotFoundError as e:
        print(f"✗ 오류: {e}")
        attack_loader = None
    
    print("\n" + "=" * 80)
    print("데이터 로더 생성 완료")
    print("=" * 80)
    
    return attack_loader


def load_trained_models(model_data_dir: str = "model_data"):
    """
    학습된 모델들을 로드하는 함수
    
    Args:
        model_data_dir: 모델 데이터 디렉토리 경로
    
    Returns:
        dict: 로드된 모델들 (bloom_filter, clusterer, config 등)
    """
    print("=" * 80)
    print("학습된 모델 로드 중...")
    print("=" * 80)
    
    models = {}
    
    # 1. 설정 정보 로드
    config_path = os.path.join(model_data_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        print(f"✓ 설정 정보 로드: {config_path}")
        models['config_dict'] = config_dict
    else:
        print(f"⚠ 경고: 설정 파일을 찾을 수 없습니다: {config_path}")
    
    # 2. Bloom Filter 로드
    bloom_filter_path = os.path.join(model_data_dir, "bloom_filter.pkl")
    if os.path.exists(bloom_filter_path):
        with open(bloom_filter_path, 'rb') as f:
            bloom_filter = pickle.load(f)
        print(f"✓ Bloom Filter 로드: {bloom_filter_path}")
        models['bloom_filter'] = bloom_filter
    else:
        print(f"⚠ 경고: Bloom Filter 파일을 찾을 수 없습니다: {bloom_filter_path}")
        models['bloom_filter'] = None
    
    # 3. 클러스터링 모델 로드
    clusterer_path = os.path.join(model_data_dir, "clusterer.pkl")
    if os.path.exists(clusterer_path):
        with open(clusterer_path, 'rb') as f:
            clusterer = pickle.load(f)
        print(f"✓ 클러스터링 모델 로드: {clusterer_path}")
        models['clusterer'] = clusterer
    else:
        print(f"⚠ 경고: 클러스터링 모델 파일을 찾을 수 없습니다: {clusterer_path}")
        models['clusterer'] = None
    
    # 4. 클러스터 레이블 로드
    cluster_labels_path = os.path.join(model_data_dir, "cluster_labels.npy")
    if os.path.exists(cluster_labels_path):
        cluster_labels = np.load(cluster_labels_path)
        print(f"✓ 클러스터 레이블 로드: {cluster_labels_path}")
        models['cluster_labels'] = cluster_labels
    else:
        print(f"⚠ 경고: 클러스터 레이블 파일을 찾을 수 없습니다: {cluster_labels_path}")
        models['cluster_labels'] = None
    
    # 5. N-gram 통계 로드
    ngram_stats_path = os.path.join(model_data_dir, "ngram_stats.json")
    if os.path.exists(ngram_stats_path):
        with open(ngram_stats_path, 'r', encoding='utf-8') as f:
            ngram_stats = json.load(f)
        print(f"✓ N-gram 통계 로드: {ngram_stats_path}")
        models['ngram_stats'] = ngram_stats
    else:
        print(f"⚠ 경고: N-gram 통계 파일을 찾을 수 없습니다: {ngram_stats_path}")
        models['ngram_stats'] = None
    
    print("\n" + "=" * 80)
    print("모델 로드 완료")
    print("=" * 80)
    
    return models


def generate_embeddings_for_attack_data(
    attack_loader: DataLoader,
    config: Config
) -> Tuple[Optional[torch.Tensor], Optional[BERTEmbedder]]:
    """
    공격 데이터에 대한 임베딩을 생성하는 함수
    
    Args:
        attack_loader: 공격 데이터 로더
        config: 설정 객체
    
    Returns:
        attack_embeddings: 생성된 임베딩 텐서 (n_samples, embedding_dim)
        embedder: BERT 임베딩 생성기 객체
    """
    print("=" * 80)
    print("공격 데이터 임베딩 생성 시작")
    print("=" * 80)
    
    try:
        # BERT 임베딩 생성기 초기화
        embedder = BERTEmbedder(model_name=config.BERT_MODEL_NAME)
        print(f"✓ BERT 임베딩 생성기 초기화 완료 (모델: {config.BERT_MODEL_NAME})")
        
        # 모든 배치에 대해 임베딩 생성
        all_embeddings = []
        total_samples = len(attack_loader.dataset)
        
        print(f"총 {total_samples}개 샘플에 대한 임베딩 생성 시작...")
        print("(이 작업은 시간이 걸릴 수 있습니다)")
        
        for batch_idx, batch_texts in enumerate(tqdm(attack_loader, desc="임베딩 생성")):
            # 배치별로 임베딩 생성
            batch_embeddings = embedder.embed_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        # 모든 임베딩을 하나의 텐서로 결합
        attack_embeddings = torch.cat(all_embeddings, dim=0)
        
        print(f"\n✓ 임베딩 생성 완료!")
        print(f"  - 총 샘플 수: {attack_embeddings.shape[0]}")
        print(f"  - 임베딩 차원: {attack_embeddings.shape[1]}")
        print(f"  - 텐서 형태: {attack_embeddings.shape}")
        
        return attack_embeddings, embedder
        
    except Exception as e:
        print(f"\n✗ 임베딩 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def search_similar_embeddings_from_vector_db(
    attack_embeddings: torch.Tensor,
    top_k: int = 3,
    vector_db_path: str = "./faiss_db",
    collection_name: str = "normal_embeddings"
) -> List[List[dict]]:
    """
    공격 데이터 임베딩에 대해 학습된 Vector DB에서 유사한 임베딩을 검색하는 함수
    
    Args:
        attack_embeddings: 공격 데이터 임베딩 텐서 (n_samples, embedding_dim)
        top_k: 검색할 상위 k개 (기본값: 3)
        vector_db_path: Vector DB 저장 경로
        collection_name: 컬렉션 이름
    
    Returns:
        similar_embeddings_list: 각 공격 임베딩에 대한 유사한 임베딩 리스트
            [
                [  # 첫 번째 공격 임베딩에 대한 결과
                    {'id': '...', 'distance': 0.123, 'metadata': {...}},
                    {'id': '...', 'distance': 0.456, 'metadata': {...}},
                    {'id': '...', 'distance': 0.789, 'metadata': {...}}
                ],
                [  # 두 번째 공격 임베딩에 대한 결과
                    ...
                ],
                ...
            ]
    """
    print("=" * 80)
    print(f"Vector DB에서 유사한 임베딩 검색 시작 (Top-{top_k})")
    print("=" * 80)
    
    # Vector DB 로드
    print(f"\nVector DB 로드 중... (경로: {vector_db_path}, 컬렉션: {collection_name})")
    try:
        vector_db = EmbeddingVectorDB(
            collection_name=collection_name,
            persist_directory=vector_db_path,
            index_type="cosine"  # 코사인 유사도 사용
        )
        
        if vector_db.index is None or vector_db.index.ntotal == 0:
            raise ValueError("Vector DB가 비어있습니다. 학습 데이터가 저장되어 있는지 확인해주세요.")
        
        print(f"✓ Vector DB 로드 완료 (임베딩 수: {vector_db.index.ntotal})")
    except Exception as e:
        print(f"✗ Vector DB 로드 실패: {e}")
        raise
    
    # 각 공격 임베딩에 대해 유사한 임베딩 검색
    n_samples = attack_embeddings.shape[0]
    print(f"\n총 {n_samples}개 공격 임베딩에 대해 검색 시작...")
    
    similar_embeddings_list = []
    
    for idx in tqdm(range(n_samples), desc="유사 임베딩 검색"):
        # 단일 임베딩 추출
        query_embedding = attack_embeddings[idx]
        
        # Vector DB에서 검색
        search_results = vector_db.search_similar(
            query_embedding=query_embedding,
            n_results=top_k
        )
        
        # 결과를 딕셔너리 리스트로 변환
        results_for_sample = []
        if search_results['ids'] and len(search_results['ids']) > 0:
            ids = search_results['ids'][0]
            distances = search_results['distances'][0]
            metadatas = search_results['metadatas'][0]
            
            for i in range(min(len(ids), top_k)):
                result_item = {
                    'id': ids[i],
                    'distance': float(distances[i]),
                    'metadata': metadatas[i]
                }
                results_for_sample.append(result_item)
        
        similar_embeddings_list.append(results_for_sample)
    
    print(f"\n✓ 검색 완료!")
    





    print(f"  - 총 {n_samples}개 샘플에 대해 검색 수행")
    print(f"  - 각 샘플당 상위 {top_k}개 결과 반환")
    
    # 샘플 결과 출력 (처음 3개만)
    print(f"\n샘플 검색 결과 (처음 3개):")
    for sample_idx in range(min(3, len(similar_embeddings_list))):
        print(f"\n  샘플 {sample_idx + 1}:")
        results = similar_embeddings_list[sample_idx]
        if results:
            for rank, result in enumerate(results, 1):
                metadata = result['metadata']
                print(f"    {rank}. 거리: {result['distance']:.4f}, "
                      f"프로세스: {metadata.get('processName', 'N/A')}, "
                      f"이벤트: {metadata.get('eventName', 'N/A')}")
        else:
            print(f"    검색 결과 없음")
    
    return similar_embeddings_list


def main(attack_file_path: Optional[str] = None):
    """
    메인 함수
    
    학습된 모델을 로드하고 공격 데이터를 검증합니다.
    
    Args:
        attack_file_path: 공격 데이터 파일 경로 (None이면 기본 경로 사용)
    """
    print("=" * 80)
    print("HIDS 이상 탐지 시스템 - 공격 데이터 검증 시작")
    print("=" * 80)
    
    # 설정 로드
    print("\n[1단계] 설정 로드 중...")
    config = Config()
    config.print_config()
    
    # 텍스트 필터 초기화
    print("\n[2단계] 텍스트 필터 초기화 중...")
    text_filter = TextFilter(keep_prefixes=config.KEEP_PREFIXES)
    print(f"✓ 텍스트 필터 초기화 완료 (Keep Prefixes: {config.KEEP_PREFIXES})")
    
    # 학습된 모델 로드
    print("\n[3단계] 학습된 모델 로드 중...")
    model_data_dir = "model_data"
    trained_models = load_trained_models(model_data_dir)
    
    # 공격 데이터 로더 생성
    print("\n[4단계] 공격 데이터 로더 생성 중...")
    attack_loader = create_attack_data_loader(config, text_filter, attack_file_path)
    
    if attack_loader is None:
        print("\n✗ 오류: 공격 데이터 로더를 생성할 수 없습니다.")
        if attack_file_path:
            print(f"제공된 파일 경로를 확인해주세요: {attack_file_path}")
        else:
            print("data/attack_tracee.json 파일이 존재하는지 확인해주세요.")
        return None
    
    # 샘플 데이터 확인
    print("\n[5단계] 샘플 데이터 확인 중...")
    sample_batch = next(iter(attack_loader))
    print(f"✓ 공격 데이터 배치 크기: {len(sample_batch)}")
    print(f"✓ 첫 번째 샘플 (처음 100자): {sample_batch[0][:100]}...")
    
    # 공격 데이터 임베딩 생성
    print("\n[6단계] 공격 데이터 임베딩 생성 중...")
    attack_embeddings = None
    embedder = None
    try:
        attack_embeddings, embedder = generate_embeddings_for_attack_data(
            attack_loader, config
        )
    except Exception as e:
        print(f"\n✗ 임베딩 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    # Vector DB에서 유사한 임베딩 검색
    similar_embeddings_list = None
    if attack_embeddings is not None:
        print("\n[7단계] Vector DB에서 유사한 임베딩 검색 중...")
        try:
            similar_embeddings_list = search_similar_embeddings_from_vector_db(
                attack_embeddings, top_k=3
            )
        except Exception as e:
            print(f"\n✗ 유사 임베딩 검색 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    # 생성된 embedding 들을 class로 분류
    classified_results = None
    if similar_embeddings_list is not None:
        print("\n[8단계] 임베딩을 클래스로 분류 중...")
        try:
            if not similar_embeddings_list:
                print("⚠ 경고: 분류할 유사 임베딩이 없습니다.")
            else:
                classified_results = classify_embeddings_by_class(
                    similar_embeddings_list, model_data_dir, config
                )
                if classified_results:
                    print(f"\n✓ 분류 완료: {len(classified_results)}개 샘플 분류됨")
                    
                    # 분류 결과 요약 출력
                    unknown_count = sum(1 for r in classified_results if r.get('class') == 'unknown')
                    classified_count = len(classified_results) - unknown_count
                    
                    print(f"\n분류 결과 요약:")
                    print(f"  - 정상 분류: {classified_count}개")
                    print(f"  - Unknown: {unknown_count}개")
                    
                    # 클러스터별 분류 결과 통계
                    cluster_counts = {}
                    for result in classified_results:
                        cluster_id = result.get('cluster_id')
                        if cluster_id is not None and cluster_id != -1:
                            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
                    
                    if cluster_counts:
                        print(f"\n클러스터별 분류 결과 (상위 10개):")
                        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
                        for cluster_id, count in sorted_clusters[:10]:
                            print(f"  - 클러스터 {cluster_id}: {count}개")
                else:
                    print("⚠ 경고: 분류 결과가 비어있습니다.")
        except FileNotFoundError as e:
            print(f"\n✗ 파일을 찾을 수 없습니다: {e}")
            print("  모델 데이터 파일이 존재하는지 확인해주세요.")
            import traceback
            traceback.print_exc()
        except ValueError as e:
            print(f"\n✗ 값 오류 발생: {e}")
            print("  클러스터 정보 파일의 형식을 확인해주세요.")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"\n✗ 임베딩 분류 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠ 경고: 유사 임베딩 검색 결과가 없어 분류를 수행할 수 없습니다.")


    # Ngram sequence 생성 및 Bloom Filter 확인
    ngram_results = None
    if classified_results is not None:
        print("\n[9단계] N-gram 시퀀스 생성 및 Bloom Filter 확인 중...")
        try:
            ngram_results = generate_ngram_and_check_bloom_filter(
                classified_results=classified_results,
                trained_models=trained_models,
                model_data_dir=model_data_dir,
                config=config
            )
            if ngram_results:
                print(f"\n✓ N-gram 생성 및 Bloom Filter 확인 완료")
                
                # 결과 요약 출력
                total_ngrams = ngram_results.get('total_ngrams', 0)
                matched_ngrams = ngram_results.get('matched_ngrams', 0)
                unmatched_ngrams = ngram_results.get('unmatched_ngrams', 0)
                
                print(f"\nN-gram 분석 결과 요약:")
                print(f"  - 총 N-gram 수: {total_ngrams}개")
                print(f"  - Bloom Filter 매칭: {matched_ngrams}개 ({matched_ngrams/total_ngrams*100:.2f}%)" if total_ngrams > 0 else "  - Bloom Filter 매칭: 0개")
                print(f"  - Bloom Filter 미매칭: {unmatched_ngrams}개 ({unmatched_ngrams/total_ngrams*100:.2f}%)" if total_ngrams > 0 else "  - Bloom Filter 미매칭: 0개")
        except Exception as e:
            print(f"\n✗ N-gram 생성 및 Bloom Filter 확인 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("초기화 완료")
    print("=" * 80)
    print("\n다음 단계: 클러스터링 및 이상 탐지 수행")
    print("=" * 80)
    
    # 결과 반환
    return {
        'config': config,
        'text_filter': text_filter,
        'attack_loader': attack_loader,
        'attack_embeddings': attack_embeddings,
        'embedder': embedder,
        'trained_models': trained_models,
        'similar_embeddings_list': similar_embeddings_list,
        'classified_results': classified_results,
        'ngram_results': ngram_results
    }


if __name__ == "__main__":
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(
        description="HIDS 이상 탐지 시스템 - 공격 데이터 검증",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python inference_data.py
  python inference_data.py --path ../data/attack_tracee.json
  python inference_data.py -p ./my_attack_data.json
        """
    )
    parser.add_argument(
        '-p', '--path',
        type=str,
        default=None,
        help='검증할 공격 데이터 파일 경로 (기본값: ../data/attack_tracee.json)'
    )
    
    args = parser.parse_args()
    
    # 메인 함수 실행
    result = main(attack_file_path=args.path)
    
    # 결과를 전역 변수로 저장 (다른 모듈에서 import하여 사용 가능)
    if result:
        globals().update(result)
