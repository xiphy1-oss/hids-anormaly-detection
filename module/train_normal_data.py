# 전체 테스트 메인 함수 작성.

# dataloder.py에서 데이터 로더 호출
# semantic_classifier.py에서 클러스터링 수행

import os
import sys
import json
import pickle
import shutil
import argparse
import torch
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader import TraceeDataset
from utility import Config, TextFilter
from embedder import BERTEmbedder
from vector_db import EmbeddingVectorDB
from clustering import Clusterer
from sequence import NGramGenerator
from bloomfilter import NGramBloomFilter


def create_data_loaders(config: Config, text_filter: TextFilter, data_type: str = 'normal', data_file_path: Optional[str] = None):
    """
    데이터 로더를 생성하는 함수
    
    Args:
        config: 설정 객체
        text_filter: 텍스트 필터 객체
        data_type: 데이터 타입 ('normal' 또는 'attack')
        data_file_path: 데이터 파일 경로 (선택적, 제공되면 이 경로를 직접 사용)
    
    Returns:
        data_loader: 생성된 데이터 로더 (해당 타입에 맞는 로더만 반환)
    """
    print("=" * 80)
    print(f"데이터 로더 생성 시작 (타입: {data_type})")
    print("=" * 80)
    
    if data_type not in ['normal', 'attack']:
        raise ValueError(f"지원하지 않는 데이터 타입: {data_type}. 'normal' 또는 'attack'만 사용 가능합니다.")
    
    filter_func = lambda x: text_filter.filter(x)
    
    # 경로가 제공된 경우 직접 사용, 그렇지 않으면 기본 경로 사용
    if data_file_path is not None:
        data_file = data_file_path
        data_type_name = "정상" if data_type == 'normal' else "공격"
        print(f"사용자 지정 경로 사용: {data_file}")
    else:
        data_dir = "../data"
        
        # 데이터 타입에 따라 파일 경로 설정
        if data_type == 'normal':
            file_name = "normal_tracee.json"
            data_type_name = "정상"
        else:  # attack
            file_name = "attack_tracee.json"
            data_type_name = "공격"
        
        data_file = os.path.join(data_dir, file_name)
    
    # 데이터 로더 생성
    print(f"\n[{data_type_name} 데이터 로더 생성 중...]")
    try:
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_file}")
        
        dataset = TraceeDataset(data_file, filter_func=filter_func)
        data_loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=config.SHUFFLE,
            num_workers=0,
            collate_fn=lambda x: x
        )
        print(f"✓ {data_type_name} 데이터 로더 생성 완료: {len(dataset)}개 샘플")
    except FileNotFoundError as e:
        print(f"✗ 오류: {e}")
        data_loader = None
    
    print("\n" + "=" * 80)
    print("데이터 로더 생성 완료")
    print("=" * 80)
    
    return data_loader


def main(data_file_path: Optional[str] = None):
    """
    메인 함수
    
    Args:
        data_file_path: 데이터 파일 경로 (선택적, 제공되면 이 경로를 사용하여 임베딩 생성)
    """
    print("=" * 80)
    print("HIDS 이상 탐지 시스템 테스트 시작")
    print("=" * 80)
    
    # 설정 로드
    print("\n[1단계] 설정 로드 중...")
    config = Config()
    config.print_config()
    
    # 텍스트 필터 초기화
    print("\n[2단계] 텍스트 필터 초기화 중...")
    text_filter = TextFilter(keep_prefixes=config.KEEP_PREFIXES)
    print(f"✓ 텍스트 필터 초기화 완료 (Keep Prefixes: {config.KEEP_PREFIXES})")
    
    # 데이터 로더 생성
    print("\n[3단계] 데이터 로더 생성 중...")
    # 경로가 제공된 경우 해당 경로 사용, 그렇지 않으면 기본 경로 사용
    normal_loader = create_data_loaders(config, text_filter, data_type='normal', data_file_path=data_file_path)
    # 경로가 제공된 경우 attack_loader는 None으로 설정 (사용자 지정 경로는 normal 데이터로 간주)
    if data_file_path is None:
        attack_loader = create_data_loaders(config, text_filter, data_type='attack')
    else:
        attack_loader = None
        print("\n⚠ 경로가 제공되었으므로 공격 데이터 로더는 생성하지 않습니다.")
    
    # 데이터 로더 검증
    if normal_loader is None:
        print("\n⚠ 경고: 정상 데이터 로더를 생성할 수 없습니다.")
        print("data/normal_tracee.json 파일이 존재하는지 확인해주세요.")
        return None
    
    if attack_loader is None:
        print("\n⚠ 경고: 공격 데이터 로더를 생성할 수 없습니다.")
        print("data/attack_tracee.json 파일이 존재하는지 확인해주세요.")
    
    # 샘플 데이터 확인
    print("\n[4단계] 샘플 데이터 확인 중...")
    sample_batch = next(iter(normal_loader))
    print(f"✓ 정상 데이터 배치 크기: {len(sample_batch)}")
    print(f"✓ 첫 번째 샘플 (처음 100자): {sample_batch[0][:100]}...")
    
    if attack_loader is not None:
        attack_sample_batch = next(iter(attack_loader))
        print(f"✓ 공격 데이터 배치 크기: {len(attack_sample_batch)}")
        print(f"✓ 첫 번째 샘플 (처음 100자): {attack_sample_batch[0][:100]}...")
    
    print("\n" + "=" * 80)
    print("데이터 로더 생성 및 검증 완료")
    print("=" * 80)
    
    # 임베딩 생성
    print("\n[5단계] 정상 데이터 임베딩 생성 중...")
    try:
        # BERT 임베딩 생성기 초기화
        embedder = BERTEmbedder(model_name=config.BERT_MODEL_NAME)
        print(f"✓ BERT 임베딩 생성기 초기화 완료 (모델: {config.BERT_MODEL_NAME})")
        
        # 모든 배치에 대해 임베딩 생성
        all_embeddings = []
        total_samples = len(normal_loader.dataset)
        
        print(f"총 {total_samples}개 샘플에 대한 임베딩 생성 시작...")
        print("(이 작업은 시간이 걸릴 수 있습니다)")
        
        for batch_idx, batch_texts in enumerate(tqdm(normal_loader, desc="임베딩 생성")):
            # 배치별로 임베딩 생성
            batch_embeddings = embedder.embed_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        # 모든 임베딩을 하나의 텐서로 결합
        normal_embeddings = torch.cat(all_embeddings, dim=0)
        
        print(f"\n✓ 임베딩 생성 완료!")
        print(f"  - 총 샘플 수: {normal_embeddings.shape[0]}")
        print(f"  - 임베딩 차원: {normal_embeddings.shape[1]}")
        print(f"  - 텐서 형태: {normal_embeddings.shape}")
        
    except Exception as e:
        print(f"\n✗ 임베딩 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        normal_embeddings = None
        embedder = None
    
    # Vector DB에 저장
    vector_db = None
    if normal_embeddings is not None:
        print("\n[6단계] Vector DB에 임베딩 저장 중...")
        try:
            # Vector DB 인스턴스 생성
            vector_db = EmbeddingVectorDB(
                collection_name="normal_embeddings",
                persist_directory="./faiss_db",
                index_type="cosine"  # "flat" (L2 거리), "ivf" (L2 거리), "cosine" (코사인 유사도)
            )
            
            # 메타데이터 준비
            normal_dataset = normal_loader.dataset
            metadata_list = []
            
            print("메타데이터 준비 중...")
            for idx in range(len(normal_dataset)):
                # 원본 데이터 가져오기
                if hasattr(normal_dataset, 'original_data') and idx < len(normal_dataset.original_data):
                    original_data = normal_dataset.original_data[idx]
                else:
                    original_data = {}
                
                # 필터링된 텍스트 가져오기
                filtered_text = normal_dataset[idx]
                
                # 메타데이터 생성
                metadata = {
                    "index": idx,
                    "text": filtered_text,
                    "processName": original_data.get("processName", "N/A"),
                    "eventName": original_data.get("eventName", "N/A"),
                    "syscall": original_data.get("syscall", "N/A"),
                    "eventId": str(original_data.get("eventId", "N/A")),
                    "data_type": "normal"
                }
                metadata_list.append(metadata)
            
            # Vector DB에 저장
            print("Vector DB에 저장 중...")
            vector_db.save_embeddings(
                embeddings=normal_embeddings,
                metadata_list=metadata_list
            )
            
            # 컬렉션 정보 출력
            info = vector_db.get_collection_info()
            print("\n✓ Vector DB 저장 완료!")
            print(f"  - 컬렉션 이름: {info['collection_name']}")
            print(f"  - 총 임베딩 수: {info['total_embeddings']}")
            print(f"  - 임베딩 차원: {info['embedding_dim']}")
            print(f"  - 인덱스 타입: {info['index_type']}")
            print(f"  - 저장 경로: {info['persist_directory']}")
            
        except Exception as e:
            print(f"\n✗ Vector DB 저장 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            vector_db = None
    
    print("\n" + "=" * 80)
    print("임베딩 생성 및 Vector DB 저장 완료")
    print("=" * 80)
    
    # Vector DB 기반 클러스터링 수행
    cluster_labels = None
    cluster_info = None
    clusterer = None
    if vector_db is not None:
        print("\n[7단계] Vector DB 기반 클러스터링 수행 중...")
        try:
            cluster_labels, cluster_info, clusterer = perform_clustering_from_vector_db(
                vector_db, config
            )
            print("\n✓ 클러스터링 완료!")
            print(f"  - 생성된 클러스터 수: {cluster_info.get('n_clusters', 'N/A')}")
            print(f"  - 클러스터링 방법: {cluster_info.get('method', 'N/A')}")
            if 'silhouette_score' in cluster_info:
                print(f"  - 실루엣 점수: {cluster_info['silhouette_score']:.4f}")
        except Exception as e:
            print(f"\n✗ 클러스터링 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    # N-gram Sequence 생성
    ngrams = None
    ngram_stats = None
    if cluster_labels is not None:
        print("\n[8단계] N-gram Sequence 생성 중...")
        try:
            ngrams, ngram_stats = create_ngram_sequence_from_clusters(
                cluster_labels, config
            )
            print("\n✓ N-gram Sequence 생성 완료!")
            print(f"  - 총 N-gram 수: {ngram_stats.get('total_ngrams', 'N/A')}")
            print(f"  - 고유 N-gram 수: {ngram_stats.get('unique_ngrams', 'N/A')}")
        except Exception as e:
            print(f"\n✗ N-gram Sequence 생성 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    # Bloom Filter 생성 및 업데이트
    bloom_filter = None
    if ngrams is not None:
        print("\n[9단계] Bloom Filter 생성 및 업데이트 중...")
        try:
            bloom_filter = create_bloom_filter_from_ngrams(ngrams, config)
            print("\n✓ Bloom Filter 생성 및 업데이트 완료!")
        except Exception as e:
            print(f"\n✗ Bloom Filter 생성 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    # 모델 데이터 저장
    print("\n[10단계] 모델 데이터 저장 중...")
    try:
        save_model_data(
            cluster_labels=cluster_labels,
            cluster_info=cluster_info,
            clusterer=clusterer,
            ngrams=ngrams,
            ngram_stats=ngram_stats,
            bloom_filter=bloom_filter,
            config=config
        )
        print("\n✓ 모델 데이터 저장 완료!")
    except Exception as e:
        print(f"\n✗ 모델 데이터 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("전체 프로세스 완료")
    print("=" * 80)
     
    # 데이터 로더 반환 (다른 모듈에서 사용할 수 있도록)
    return {
        'config': config,
        'text_filter': text_filter,
        'normal_loader': normal_loader,
        'attack_loader': attack_loader,
        'normal_embeddings': normal_embeddings,
        'embedder': embedder if 'embedder' in locals() else None,
        'vector_db': vector_db if 'vector_db' in locals() else None,
        'cluster_labels': cluster_labels if 'cluster_labels' in locals() else None,
        'cluster_info': cluster_info if 'cluster_info' in locals() else None,
        'clusterer': clusterer if 'clusterer' in locals() else None,
        'ngrams': ngrams if 'ngrams' in locals() else None,
        'ngram_stats': ngram_stats if 'ngram_stats' in locals() else None,
        'bloom_filter': bloom_filter if 'bloom_filter' in locals() else None
    }


def calculate_cluster_statistics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    index_type: str = "flat"
) -> tuple:
    """
    각 클러스터별 대표 벡터(centroid)와 최대 거리를 계산하는 함수
    
    Args:
        embeddings: 임베딩 벡터 배열 (n_samples, embedding_dim)
        labels: 클러스터 레이블 배열 (n_samples,)
        index_type: 인덱스 타입 ("flat", "ivf", "cosine")
    
    Returns:
        cluster_centroids: 클러스터 ID를 키로 하는 대표 벡터 딕셔너리
        cluster_max_distances: 클러스터 ID를 키로 하는 최대 거리 딕셔너리
    """
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
    from sklearn.preprocessing import normalize
    
    cluster_centroids = {}
    cluster_max_distances = {}
    
    unique_labels = np.unique(labels)
    
    # 코사인 인덱스인 경우 임베딩을 정규화해야 함
    # Vector DB에 저장할 때 이미 정규화된 벡터를 저장했으므로,
    # reconstruct로 가져온 벡터도 정규화되어 있을 수 있음
    # 하지만 명시적으로 정규화하여 일관성 유지
    if index_type == "cosine":
        # L2 정규화 (단위 벡터로 변환)
        # normalize 함수는 0 벡터에 대해서도 안전하게 처리함
        embeddings_normalized = normalize(embeddings, norm='l2', axis=1)
        
        # 디버깅: 정규화 확인
        sample_norms = np.linalg.norm(embeddings_normalized[:min(5, len(embeddings_normalized))], axis=1)
        if len(sample_norms) > 0:
            print(f"  [디버깅] 정규화 확인: 샘플 벡터들의 노름 = {sample_norms}")
    else:
        embeddings_normalized = embeddings
    
    for cluster_id in unique_labels:
        # 노이즈 포인트(-1)도 하나의 클러스터로 처리
        # 해당 클러스터에 속한 임베딩 추출
        cluster_mask = labels == cluster_id
        cluster_embeddings = embeddings_normalized[cluster_mask]
        
        if len(cluster_embeddings) == 0:
            continue
        
        # 노이즈 포인트가 1개만 있는 경우 centroid 계산 불가
        if cluster_id == -1 and len(cluster_embeddings) == 1:
            # 단일 노이즈 포인트는 centroid를 자기 자신으로 설정
            centroid = cluster_embeddings[0]
            max_distance = 0.0  # 자기 자신과의 거리는 0
            cluster_centroids[cluster_id] = centroid
            cluster_max_distances[cluster_id] = max_distance
            continue
        
        # 대표 벡터 계산 (평균 벡터)
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # 코사인 인덱스인 경우 centroid도 정규화 필요
        if index_type == "cosine":
            # centroid를 정규화 (단위 벡터로 변환)
            # 평균 벡터는 정규화되지 않았으므로 정규화 필요
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-10:  # 0으로 나누기 방지
                centroid = centroid / centroid_norm
            else:
                # 노름이 0인 경우 (모든 벡터가 동일한 경우)
                centroid = centroid
                # 이 경우 거리는 0이 됨 (정상)
        
        cluster_centroids[cluster_id] = centroid
        
        # 코사인 인덱스인 경우 코사인 거리 사용, 그 외에는 L2 거리 사용
        if index_type == "cosine":
            # 코사인 거리 계산 (1 - 코사인 유사도)
            # 정규화된 벡터들 간의 코사인 거리
            # 정규화된 벡터의 경우: 코사인 유사도 = 내적 (||a|| = ||b|| = 1이므로)
            # 코사인 거리 = 1 - 코사인 유사도 = 1 - 내적
            
            # 각 벡터와 centroid 간의 내적 계산 (정규화된 벡터이므로 내적 = 코사인 유사도)
            cosine_similarities = np.dot(cluster_embeddings, centroid)
            # 코사인 거리 = 1 - 코사인 유사도
            distances = 1.0 - cosine_similarities
            
            # 디버깅: 거리 값 확인 (처음 3개 클러스터만)
            if cluster_id < 3 and len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                mean_dist = np.mean(distances)
                min_sim = np.min(cosine_similarities)
                max_sim = np.max(cosine_similarities)
                print(f"  [디버깅] 클러스터 {cluster_id}: "
                      f"거리 범위 [{min_dist:.6f}, {max_dist:.6f}], 평균={mean_dist:.6f}, "
                      f"유사도 범위 [{min_sim:.6f}, {max_sim:.6f}], 샘플 수={len(distances)}")
        else:
            # L2 거리 계산
            distances = euclidean_distances(cluster_embeddings, centroid.reshape(1, -1)).flatten()
        
        # 최대 거리 계산
        max_distance = np.max(distances)
        cluster_max_distances[cluster_id] = max_distance
    
    return cluster_centroids, cluster_max_distances


def perform_clustering_from_vector_db(
    vector_db: EmbeddingVectorDB,
    config: Config
) -> tuple:
    """
    Vector DB에서 임베딩을 가져와 클러스터링을 수행하는 함수
    
    Args:
        vector_db: EmbeddingVectorDB 인스턴스
        config: 설정 객체
    
    Returns:
        cluster_labels: 클러스터 레이블 배열
        cluster_info: 클러스터 정보 딕셔너리
    """
    print("Vector DB에서 임베딩 가져오는 중...")
    
    # Vector DB에서 모든 임베딩 벡터 가져오기
    embeddings = vector_db.get_all_embeddings_vectors()
    print(f"✓ {embeddings.shape[0]}개의 임베딩을 가져왔습니다. (차원: {embeddings.shape[1]})")
    
    # 클러스터링 수행
    print(f"\n클러스터링 수행 중... (방법: {config.CLUSTERING_METHOD})")
    clusterer = Clusterer(
        method=config.CLUSTERING_METHOD,
        alpha=config.CLUSTERING_ALPHA,
        random_state=config.CLUSTERING_RANDOM_STATE
    )
    
    labels, n_clusters, cluster_info = clusterer.fit(embeddings)
    
    # 클러스터별 통계 출력
    print("\n클러스터별 통계:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"  노이즈 포인트: {count}개")
        else:
            print(f"  클러스터 {label}: {count}개")
    
    # 각 클러스터별 대표 벡터와 최대 거리 계산 및 출력
    print("\n" + "=" * 80)
    print("클러스터별 대표 벡터 및 최대 거리 분석")
    print("=" * 80)
    cluster_centroids, cluster_max_distances = calculate_cluster_statistics(
        embeddings, labels, vector_db.index_type
    )
    
    # 결과 출력
    print(f"\n총 {len(cluster_centroids)}개의 클러스터 분석 완료 (노이즈 포인트 포함)\n")
    
    # 일반 클러스터 출력
    for cluster_id in sorted(cluster_centroids.keys()):
        if cluster_id == -1:
            continue  # 노이즈 포인트는 아래에서 별도로 출력
        
        centroid = cluster_centroids[cluster_id]
        max_distance = cluster_max_distances[cluster_id]
        count = counts[unique_labels == cluster_id][0]
        
        print(f"[클러스터 {cluster_id}]")
        print(f"  - 샘플 수: {count}개")
        print(f"  - 대표 벡터 (centroid) 차원: {centroid.shape}")
        print(f"  - 대표 벡터 (처음 10개 값): {centroid[:10]}")
        print(f"  - 대표 벡터 (마지막 10개 값): {centroid[-10:]}")
        if vector_db.index_type == "cosine":
            print(f"  - 최대 코사인 거리: {max_distance:.6f}")
        else:
            print(f"  - 최대 L2 거리: {max_distance:.6f}")
        print()
    
    # 노이즈 포인트가 있는 경우 별도 출력
    if -1 in unique_labels and -1 in cluster_centroids:
        noise_count = counts[unique_labels == -1][0]
        noise_centroid = cluster_centroids[-1]
        noise_max_dist = cluster_max_distances.get(-1, None)
        
        print(f"[노이즈 포인트 (클러스터 -1)]")
        print(f"  - 샘플 수: {noise_count}개")
        print(f"  - 대표 벡터 (centroid) 차원: {noise_centroid.shape}")
        print(f"  - 대표 벡터 (처음 10개 값): {noise_centroid[:10]}")
        print(f"  - 대표 벡터 (마지막 10개 값): {noise_centroid[-10:]}")
        if noise_max_dist is not None:
            if vector_db.index_type == "cosine":
                print(f"  - 최대 코사인 거리: {noise_max_dist:.6f}")
            else:
                print(f"  - 최대 L2 거리: {noise_max_dist:.6f}")
        print(f"  - 참고: 노이즈 포인트도 하나의 클러스터로 분류됩니다.\n")
    
    # 클러스터 정보에 통계 추가 (노이즈 포인트 포함)
    cluster_info['cluster_centroids'] = {
        str(k): v.tolist() for k, v in cluster_centroids.items()
    }
    cluster_info['cluster_max_distances'] = {
        str(k): float(v) for k, v in cluster_max_distances.items()
    }
    
    return labels, cluster_info, clusterer


def create_ngram_sequence_from_clusters(
    cluster_labels: np.ndarray,
    config: Config
) -> tuple:
    """
    클러스터 레이블로부터 N-gram Sequence를 생성하는 함수
    
    Args:
        cluster_labels: 클러스터 레이블 배열 (numpy array)
        config: 설정 객체
    
    Returns:
        ngrams: N-gram 시퀀스 리스트
        ngram_stats: N-gram 통계 정보 딕셔너리
    """
    print(f"N-gram 생성 중... (N={config.N_GRAM_SIZE})")
    
    # numpy array를 리스트로 변환
    if isinstance(cluster_labels, np.ndarray):
        cluster_sequence = cluster_labels.tolist()
    else:
        cluster_sequence = list(cluster_labels)
    
    print(f"✓ 클러스터 시퀀스 길이: {len(cluster_sequence)}")
    
    # N-gram 생성기 초기화
    ngram_generator = NGramGenerator(n=config.N_GRAM_SIZE)
    
    # N-gram 생성
    ngrams = ngram_generator.create_from_sequence(cluster_sequence)
    
    print(f"✓ {config.N_GRAM_SIZE}-gram 생성 완료: {len(ngrams)}개")
    
    # 통계 분석
    ngram_stats = ngram_generator.analyze(ngrams, top_k=config.TOP_N_CLUSTERS)
    
    # 통계 출력
    print(f"\nN-gram 통계:")
    print(f"  - 총 {config.N_GRAM_SIZE}-gram 수: {ngram_stats['total_ngrams']}")
    print(f"  - 고유 {config.N_GRAM_SIZE}-gram 수: {ngram_stats['unique_ngrams']}")
    print(f"\n  상위 {config.TOP_N_CLUSTERS}개 빈도 {config.N_GRAM_SIZE}-gram:")
    for i, (ngram, count) in enumerate(ngram_stats['top_ngrams'], 1):
        percentage = (count / ngram_stats['total_ngrams']) * 100
        print(f"    {i:2d}. {ngram}: {count:4d}회 ({percentage:.2f}%)")
    
    return ngrams, ngram_stats


def create_bloom_filter_from_ngrams(
    ngrams: List[Tuple],
    config: Config
) -> NGramBloomFilter:
    """
    N-gram 리스트로부터 Bloom Filter를 생성하고 업데이트하는 함수
    
    Args:
        ngrams: N-gram 시퀀스 리스트
        config: 설정 객체
    
    Returns:
        bloom_filter: 생성된 NGramBloomFilter 객체
    """
    print(f"Bloom Filter 생성 중... (Error Rate: {config.BLOOM_FILTER_ERROR_RATE})")
    
    # NGramBloomFilter 인스턴스 생성
    bloom_filter = NGramBloomFilter(error_rate=config.BLOOM_FILTER_ERROR_RATE)
    
    # N-gram을 Bloom Filter에 추가
    bloom_filter.create(ngrams)
    
    # Bloom Filter 정보 출력
    print(f"✓ Bloom Filter 생성 완료!")
    bloom_filter.print_info()
    
    # 검증: 일부 N-gram이 제대로 추가되었는지 확인
    if len(ngrams) > 0:
        test_ngrams = ngrams[:min(10, len(ngrams))]  # 처음 10개만 테스트
        found_count = sum(1 for ngram in test_ngrams if bloom_filter.check(ngram))
        print(f"✓ 검증: 테스트 N-gram {len(test_ngrams)}개 중 {found_count}개가 Bloom Filter에 존재")
    
    return bloom_filter


def save_model_data(
    cluster_labels: np.ndarray = None,
    cluster_info: dict = None,
    clusterer: Clusterer = None,
    ngrams: List[Tuple] = None,
    ngram_stats: dict = None,
    bloom_filter: NGramBloomFilter = None,
    config: Config = None,
    output_dir: str = "model_data"
):
    """
    각 단계에서 생성된 모델 데이터를 저장하는 함수
    
    Args:
        cluster_labels: 클러스터 레이블 배열
        cluster_info: 클러스터 정보 딕셔너리
        clusterer: 클러스터링 모델 객체
        ngrams: N-gram 시퀀스 리스트
        ngram_stats: N-gram 통계 정보
        bloom_filter: Bloom Filter 객체
        config: 설정 객체
        output_dir: 저장 디렉토리 경로
    """
    # 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    print(f"모델 데이터 저장 디렉토리: {output_dir}")
    
    saved_files = []
    
    # 1. 클러스터 레이블 저장 (numpy array)
    if cluster_labels is not None:
        cluster_labels_path = os.path.join(output_dir, "cluster_labels.npy")
        np.save(cluster_labels_path, cluster_labels)
        saved_files.append(cluster_labels_path)
        print(f"  ✓ 클러스터 레이블 저장: {cluster_labels_path}")
    
    # 2. 클러스터 정보 저장 (JSON)
    if cluster_info is not None:
        cluster_info_path = os.path.join(output_dir, "cluster_info.json")
        # numpy 타입을 JSON 직렬화 가능한 형태로 변환
        cluster_info_serializable = {}
        for key, value in cluster_info.items():
            if isinstance(value, (np.integer, np.floating)):
                cluster_info_serializable[key] = float(value)
            elif isinstance(value, np.ndarray):
                cluster_info_serializable[key] = value.tolist()
            elif isinstance(value, (str, int, float, bool, type(None), list, dict)):
                cluster_info_serializable[key] = value
            else:
                cluster_info_serializable[key] = str(value)
        
        with open(cluster_info_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_info_serializable, f, ensure_ascii=False, indent=2)
        saved_files.append(cluster_info_path)
        print(f"  ✓ 클러스터 정보 저장: {cluster_info_path}")
    
    # 3. 클러스터링 모델 저장 (pickle)
    if clusterer is not None:
        clusterer_path = os.path.join(output_dir, "clusterer.pkl")
        with open(clusterer_path, 'wb') as f:
            pickle.dump(clusterer, f)
        saved_files.append(clusterer_path)
        print(f"  ✓ 클러스터링 모델 저장: {clusterer_path}")
    
    # 4. N-gram 시퀀스 저장 (JSON)
    if ngrams is not None:
        ngrams_path = os.path.join(output_dir, "ngrams.json")
        # 튜플을 리스트로 변환하여 JSON 직렬화 가능하게 만듦
        ngrams_serializable = [list(ngram) for ngram in ngrams]
        with open(ngrams_path, 'w', encoding='utf-8') as f:
            json.dump(ngrams_serializable, f, ensure_ascii=False, indent=2)
        saved_files.append(ngrams_path)
        print(f"  ✓ N-gram 시퀀스 저장: {ngrams_path} ({len(ngrams)}개)")
    
    # 5. N-gram 통계 저장 (JSON)
    if ngram_stats is not None:
        ngram_stats_path = os.path.join(output_dir, "ngram_stats.json")
        # 통계 정보를 JSON 직렬화 가능한 형태로 변환
        ngram_stats_serializable = {}
        for key, value in ngram_stats.items():
            if key == 'top_ngrams':
                # 튜플을 리스트로 변환
                ngram_stats_serializable[key] = [
                    (list(ngram), count) for ngram, count in value
                ]
            elif key == 'ngram_counts':
                # 딕셔너리의 키(튜플)를 문자열로 변환
                ngram_stats_serializable[key] = {
                    str(ngram): count for ngram, count in value.items()
                }
            elif isinstance(value, (str, int, float, bool, type(None), list)):
                ngram_stats_serializable[key] = value
            elif isinstance(value, dict):
                # 딕셔너리인 경우, 키가 튜플일 수 있으므로 확인
                serializable_dict = {}
                for k, v in value.items():
                    if isinstance(k, tuple):
                        serializable_dict[str(k)] = v
                    else:
                        serializable_dict[k] = v
                ngram_stats_serializable[key] = serializable_dict
            else:
                ngram_stats_serializable[key] = str(value)
        
        with open(ngram_stats_path, 'w', encoding='utf-8') as f:
            json.dump(ngram_stats_serializable, f, ensure_ascii=False, indent=2)
        saved_files.append(ngram_stats_path)
        print(f"  ✓ N-gram 통계 저장: {ngram_stats_path}")
    
    # 6. Bloom Filter 저장 (pickle)
    if bloom_filter is not None:
        bloom_filter_path = os.path.join(output_dir, "bloom_filter.pkl")
        with open(bloom_filter_path, 'wb') as f:
            pickle.dump(bloom_filter, f)
        saved_files.append(bloom_filter_path)
        print(f"  ✓ Bloom Filter 저장: {bloom_filter_path}")
    
    # 7. 설정 정보 저장 (JSON) - 참고용
    if config is not None:
        config_path = os.path.join(output_dir, "config.json")
        config_dict = {
            'BERT_MODEL_NAME': config.BERT_MODEL_NAME,
            'CLUSTERING_METHOD': config.CLUSTERING_METHOD,
            'CLUSTERING_ALPHA': config.CLUSTERING_ALPHA,
            'CLUSTERING_RANDOM_STATE': config.CLUSTERING_RANDOM_STATE,
            'N_GRAM_SIZE': config.N_GRAM_SIZE,
            'BLOOM_FILTER_ERROR_RATE': config.BLOOM_FILTER_ERROR_RATE,
            'TOP_N_CLUSTERS': config.TOP_N_CLUSTERS,
            'KEEP_PREFIXES': list(config.KEEP_PREFIXES),
            'BATCH_SIZE': config.BATCH_SIZE,
            'SHUFFLE': config.SHUFFLE
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        saved_files.append(config_path)
        print(f"  ✓ 설정 정보 저장: {config_path}")
    
    # 8. 추가 JSON 파일들 복사 (프로젝트 루트에서 찾아서 복사)
    additional_files = [
        'cluster_mapping.json',
        'cluster_mapping_refined.json',
        'matched_sequence.json'
    ]
    
    # 프로젝트 루트 디렉토리 찾기 (현재 파일의 위치에서 상위로 올라가기)
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))  # module/testMain.py -> module/ -> project_root/
    
    for filename in additional_files:
        source_path = os.path.join(project_root, filename)
        if os.path.exists(source_path):
            dest_path = os.path.join(output_dir, filename)
            try:
                shutil.copy2(source_path, dest_path)
                saved_files.append(dest_path)
                print(f"  ✓ {filename} 복사: {dest_path}")
            except Exception as e:
                print(f"  ✗ {filename} 복사 실패: {e}")
        else:
            print(f"  ⚠ {filename} 파일을 찾을 수 없습니다: {source_path}")
    
    print(f"\n총 {len(saved_files)}개의 파일이 저장되었습니다.")


if __name__ == "__main__":
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(
        description='HIDS 이상 탐지 시스템 - 정상 데이터 학습',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python train_normal_data.py                           # 기본 경로 사용
  python train_normal_data.py --path data/my_data.json  # 사용자 지정 경로 사용
  python train_normal_data.py -p data/my_data.json      # 짧은 옵션 사용
        """
    )
    parser.add_argument(
        '--path', '-p',
        type=str,
        default=None,
        help='임베딩 생성을 위한 데이터 파일 경로 (기본값: None, 기본 경로 사용)'
    )
    
    args = parser.parse_args()
    
    # 파일 경로 검증
    if args.path is not None:
        if not os.path.exists(args.path):
            print(f"\n✗ 오류: 지정한 파일을 찾을 수 없습니다: {args.path}")
            sys.exit(1)
        if not os.path.isfile(args.path):
            print(f"\n✗ 오류: 지정한 경로가 파일이 아닙니다: {args.path}")
            sys.exit(1)
        print(f"\n✓ 사용자 지정 파일 경로: {args.path}")
    
    result = main(data_file_path=args.path)
    
    # 결과를 전역 변수로 저장 (다른 모듈에서 import하여 사용 가능)
    if result:
        globals().update(result)