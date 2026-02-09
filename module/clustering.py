"""
클러스터링 모듈
"""
import json
import numpy as np
import torch
import pandas as pd
from collections import defaultdict
from typing import Literal, Optional, Tuple, Dict, Any, Union
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class Clusterer:
    """클러스터링 클래스"""
    
    def __init__(self, method: str = 'dbscan', alpha: float = 1.0, random_state: int = 42):
        """
        Args:
            method: 클러스터링 방법 ('kmeans', 'dbscan', 'hdbscan', 'agglomerative', 'meanshift')
            alpha: 클러스터링 비율 조절 파라미터 (0.1 ~ 2.0)
            random_state: 랜덤 시드
        """
        self.method = method
        self.alpha = alpha
        self.random_state = random_state
        self.labels_ = None
        self.n_clusters_ = None
        self.cluster_info_ = None
    
    def fit(self, embeddings: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        임베딩을 클러스터링
        
        Args:
            embeddings: 임베딩 텐서 또는 numpy 배열 (n_samples, embedding_dim)
        
        Returns:
            labels: 각 샘플의 클러스터 레이블 (numpy array)
            n_clusters: 생성된 클러스터 수
            cluster_info: 클러스터 정보 딕셔너리
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy()
        else:
            embeddings_np = np.array(embeddings)
        
        n_samples = embeddings_np.shape[0]
        
        if self.method == 'kmeans':
            labels, n_clusters, cluster_info = self._kmeans_cluster(embeddings_np, n_samples)
        elif self.method == 'dbscan':
            labels, n_clusters, cluster_info = self._dbscan_cluster(embeddings_np, n_samples)
        elif self.method == 'hdbscan':
            labels, n_clusters, cluster_info = self._hdbscan_cluster(embeddings_np, n_samples)
        elif self.method == 'agglomerative':
            labels, n_clusters, cluster_info = self._agglomerative_cluster(embeddings_np, n_samples)
        elif self.method == 'meanshift':
            labels, n_clusters, cluster_info = self._meanshift_cluster(embeddings_np, n_samples)
        else:
            raise ValueError(
                f"지원하지 않는 클러스터링 방법: {self.method}. "
                f"사용 가능한 방법: 'kmeans', 'dbscan', 'hdbscan', 'agglomerative', 'meanshift'"
            )
        
        self.labels_ = labels
        self.n_clusters_ = n_clusters
        self.cluster_info_ = cluster_info
        
        return labels, n_clusters, cluster_info
    
    def _kmeans_cluster(self, embeddings_np: np.ndarray, n_samples: int) -> Tuple[np.ndarray, int, Dict]:
        """K-means 클러스터링"""
        base_clusters = int(np.sqrt(n_samples))
        n_clusters = max(2, int(base_clusters * self.alpha))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings_np)
        
        if n_clusters > 1:
            silhouette_avg = silhouette_score(embeddings_np, labels)
        else:
            silhouette_avg = 0.0
        
        cluster_info = {
            'method': 'kmeans',
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'inertia': kmeans.inertia_,
            'centers': kmeans.cluster_centers_
        }
        
        return labels, n_clusters, cluster_info
    
    def _dbscan_cluster(self, embeddings_np: np.ndarray, n_samples: int) -> Tuple[np.ndarray, int, Dict]:
        """DBSCAN 클러스터링"""
        neighbors = NearestNeighbors(n_neighbors=min(10, n_samples))
        neighbors_fit = neighbors.fit(embeddings_np)
        distances, indices = neighbors_fit.kneighbors(embeddings_np)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        
        base_eps = np.median(distances)
        eps = base_eps / self.alpha
        min_samples = max(3, int(np.log(n_samples) * self.alpha))
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(embeddings_np)
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > 1:
                silhouette_avg = silhouette_score(embeddings_np[mask], labels[mask])
            else:
                silhouette_avg = 0.0
        else:
            silhouette_avg = 0.0
        
        cluster_info = {
            'method': 'dbscan',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette_avg,
            'eps': eps,
            'min_samples': min_samples
        }
        
        return labels, n_clusters, cluster_info
    
    def _hdbscan_cluster(self, embeddings_np: np.ndarray, n_samples: int) -> Tuple[np.ndarray, int, Dict]:
        """HDBSCAN 클러스터링"""
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN이 설치되지 않았습니다. 'pip install hdbscan'을 실행하세요.")
        
        min_cluster_size = max(2, int(n_samples / (10 * self.alpha)))
        min_samples = max(3, int(np.log(n_samples) * self.alpha))
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.0
        )
        labels = clusterer.fit_predict(embeddings_np)
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > 1:
                silhouette_avg = silhouette_score(embeddings_np[mask], labels[mask])
            else:
                silhouette_avg = 0.0
        else:
            silhouette_avg = 0.0
        
        cluster_info = {
            'method': 'hdbscan',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette_avg,
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'cluster_probabilities': clusterer.probabilities_
        }
        
        return labels, n_clusters, cluster_info
    
    def _agglomerative_cluster(self, embeddings_np: np.ndarray, n_samples: int) -> Tuple[np.ndarray, int, Dict]:
        """Agglomerative Clustering"""
        distances = pairwise_distances(embeddings_np)
        base_threshold = np.percentile(distances[distances > 0], 50)
        distance_threshold = base_threshold / self.alpha
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='ward'
        )
        labels = clustering.fit_predict(embeddings_np)
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters > 1:
            silhouette_avg = silhouette_score(embeddings_np, labels)
        else:
            silhouette_avg = 0.0
        
        cluster_info = {
            'method': 'agglomerative',
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'distance_threshold': distance_threshold
        }
        
        return labels, n_clusters, cluster_info
    
    def _meanshift_cluster(self, embeddings_np: np.ndarray, n_samples: int) -> Tuple[np.ndarray, int, Dict]:
        """Mean Shift 클러스터링"""
        from sklearn.cluster import estimate_bandwidth
        
        try:
            sample_size = min(500, n_samples)
            if n_samples > sample_size:
                sample_indices = np.random.choice(n_samples, sample_size, replace=False)
                sample_data = embeddings_np[sample_indices]
            else:
                sample_data = embeddings_np
            
            distances = pairwise_distances(sample_data)
            non_zero_distances = distances[distances > 0]
            if len(non_zero_distances) > 0:
                base_bandwidth = np.median(non_zero_distances)
            else:
                base_bandwidth = estimate_bandwidth(embeddings_np, quantile=0.3, n_samples=sample_size)
        except:
            base_bandwidth = estimate_bandwidth(embeddings_np, quantile=0.3, n_samples=min(500, n_samples))
        
        bandwidth = max(base_bandwidth / self.alpha, base_bandwidth * 0.1)
        
        try:
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, max_iter=300)
            labels = ms.fit_predict(embeddings_np)
            
            unique_labels = set(labels)
            n_clusters = len(unique_labels)
            
            if n_clusters > 1:
                silhouette_avg = silhouette_score(embeddings_np, labels)
            else:
                silhouette_avg = 0.0
            
            cluster_info = {
                'method': 'meanshift',
                'n_clusters': n_clusters,
                'silhouette_score': silhouette_avg,
                'bandwidth': bandwidth
            }
        except ValueError as e:
            if "bandwidth" in str(e).lower():
                bandwidth = bandwidth * 2
                try:
                    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, max_iter=300)
                    labels = ms.fit_predict(embeddings_np)
                    unique_labels = set(labels)
                    n_clusters = len(unique_labels)
                    if n_clusters > 1:
                        silhouette_avg = silhouette_score(embeddings_np, labels)
                    else:
                        silhouette_avg = 0.0
                    cluster_info = {
                        'method': 'meanshift',
                        'n_clusters': n_clusters,
                        'silhouette_score': silhouette_avg,
                        'bandwidth': bandwidth,
                        'note': 'bandwidth 자동 조정됨'
                    }
                except:
                    labels = np.zeros(n_samples, dtype=int)
                    cluster_info = {
                        'method': 'meanshift',
                        'n_clusters': 1,
                        'silhouette_score': 0.0,
                        'bandwidth': bandwidth,
                        'note': 'bandwidth 조정 실패, 모든 포인트를 하나의 클러스터로 처리'
                    }
            else:
                raise
        
        return labels, n_clusters, cluster_info


class ClusterTracer:
    """클러스터링 결과와 원본 데이터를 매칭하여 추적하는 클래스"""
    
    def __init__(self, dataloader, cluster_labels: np.ndarray, filter_func=None):
        """
        Args:
            dataloader: 데이터 로더 (원본 데이터 접근용)
            cluster_labels: 클러스터링 결과 레이블 (numpy array)
            filter_func: 필터링 함수 (원본 데이터 로드 시 사용)
        """
        self.dataloader = dataloader
        self.cluster_labels = cluster_labels
        self.filter_func = filter_func
        self.original_data = []
        self.cluster_to_indices = defaultdict(list)
        
        self._load_original_data()
        self._map_clusters()
    
    def _load_original_data(self):
        """원본 JSON 데이터를 로드"""
        dataset = self.dataloader.dataset
        
        if hasattr(dataset, 'original_data'):
            self.original_data = dataset.original_data
        elif hasattr(dataset, 'data_path'):
            data_path = dataset.data_path
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            json_obj = json.loads(line)
                            self.original_data.append(json_obj)
                        except json.JSONDecodeError:
                            continue
        else:
            for i in range(len(dataset)):
                text = dataset[i]
                self.original_data.append({'filtered_text': text})
    
    def _map_clusters(self):
        """클러스터 레이블과 데이터 인덱스를 매핑"""
        for idx, label in enumerate(self.cluster_labels):
            self.cluster_to_indices[label].append(idx)
    
    def get_cluster_data(self, cluster_id: int) -> list:
        """
        특정 클러스터에 속한 모든 데이터 반환
        
        Args:
            cluster_id: 클러스터 ID
        
        Returns:
            해당 클러스터에 속한 데이터 리스트
        """
        if cluster_id not in self.cluster_to_indices:
            return []
        
        indices = self.cluster_to_indices[cluster_id]
        return [self.original_data[idx] for idx in indices]
    
    def get_cluster_summary(self) -> Dict:
        """
        클러스터별 요약 정보 반환
        
        Returns:
            클러스터 요약 정보 딕셔너리
        """
        summary = {}
        for cluster_id, indices in self.cluster_to_indices.items():
            summary[cluster_id] = {
                'count': len(indices),
                'indices': indices,
                'sample_data': self.original_data[indices[0]] if indices else None
            }
        return summary
    
    def print_cluster_summary(self, top_n: int = 10):
        """
        클러스터 요약 정보 출력
        
        Args:
            top_n: 출력할 상위 클러스터 개수
        """
        sorted_clusters = sorted(
            self.cluster_to_indices.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        print("=" * 80)
        print("클러스터 요약 정보")
        print("=" * 80)
        print(f"총 클러스터 수: {len(self.cluster_to_indices)}")
        print(f"총 데이터 수: {len(self.cluster_labels)}")
        print()
        
        for i, (cluster_id, indices) in enumerate(sorted_clusters[:top_n]):
            print(f"[클러스터 {cluster_id}] 데이터 개수: {len(indices)}")
            if indices:
                sample = self.original_data[indices[0]]
                if isinstance(sample, dict):
                    key_fields = ['processName', 'eventName', 'syscall', 'eventId']
                    sample_info = {k: sample.get(k, 'N/A') for k in key_fields if k in sample}
                    print(f"  샘플 데이터: {sample_info}")
                else:
                    print(f"  샘플 데이터: {str(sample)[:200]}...")
            print()
    
    def get_cluster_statistics(self) -> pd.DataFrame:
        """
        클러스터 통계 정보 반환 (DataFrame)
        
        Returns:
            클러스터 통계 DataFrame
        """
        stats = []
        for cluster_id, indices in self.cluster_to_indices.items():
            cluster_data = [self.original_data[idx] for idx in indices]
            
            process_names = [d.get('processName', 'N/A') for d in cluster_data if isinstance(d, dict)]
            event_names = [d.get('eventName', 'N/A') for d in cluster_data if isinstance(d, dict)]
            syscalls = [d.get('syscall', 'N/A') for d in cluster_data if isinstance(d, dict)]
            
            stats.append({
                'cluster_id': cluster_id,
                'count': len(indices),
                'unique_processes': len(set(process_names)),
                'unique_events': len(set(event_names)),
                'unique_syscalls': len(set(syscalls)),
                'most_common_process': max(set(process_names), key=process_names.count) if process_names else 'N/A',
                'most_common_event': max(set(event_names), key=event_names.count) if event_names else 'N/A',
                'most_common_syscall': max(set(syscalls), key=syscalls.count) if syscalls else 'N/A',
            })
        
        return pd.DataFrame(stats).sort_values('count', ascending=False)
    
    def export_cluster_mapping(self, output_path: str = 'cluster_mapping.json'):
        """
        클러스터 매핑 정보를 JSON 파일로 저장
        
        Args:
            output_path: 출력 파일 경로
        """
        mapping = {
            'total_clusters': len(self.cluster_to_indices),
            'total_samples': len(self.cluster_labels),
            'clusters': {}
        }
        
        for cluster_id, indices in self.cluster_to_indices.items():
            indexed_data = []
            for idx in indices:
                data = self.original_data[idx]
                indexed_data.append({
                    'index': idx,
                    'data': data
                })
            
            mapping['clusters'][str(cluster_id)] = {
                'count': len(indices),
                'indices': indices,
                'data': indexed_data,
                'sample_data': self.original_data[indices[0]] if indices else None
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        
        print(f"클러스터 매핑 정보가 {output_path}에 저장되었습니다.")


def refine_cluster_mapping(
    mapping_file: str = 'cluster_mapping.json',
    keep_prefixes: Optional[Tuple[str, ...]] = None,
    output_file: Optional[str] = None
) -> Dict:
    """
    cluster_mapping.json 파일의 내용을 정제하여 KEEP_PREFIXES에 있는 필드만 추출
    
    Args:
        mapping_file: cluster_mapping.json 파일 경로
        keep_prefixes: 유지할 필드 접두사 튜플 (None이면 기본값 사용)
        output_file: 정제된 결과를 저장할 파일 경로 (None이면 저장하지 않음)
    
    Returns:
        정제된 클러스터 매핑 딕셔너리
    """
    if keep_prefixes is None:
        keep_prefixes = ('processName', 'eventName', 'syscall', 'args', 'executable', 'returnValue')
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    refined_mapping = {
        'total_clusters': mapping.get('total_clusters', 0),
        'total_samples': mapping.get('total_samples', 0),
        'keep_prefixes': list(keep_prefixes),
        'clusters': {}
    }
    
    for cluster_id, cluster_info in mapping.get('clusters', {}).items():
        refined_cluster = {
            'cluster_id': int(cluster_id) if cluster_id != '-1' else -1,
            'count': cluster_info.get('count', 0),
            'data': []
        }
        
        for item in cluster_info.get('data', []):
            original_data = item.get('data', {})
            refined_data = {
                'index': item.get('index'),
                'filtered_data': {}
            }
            
            if isinstance(original_data, dict):
                for key, value in original_data.items():
                    if any(key.startswith(prefix) for prefix in keep_prefixes):
                        refined_data['filtered_data'][key] = value
            else:
                refined_data['filtered_data'] = original_data
            
            refined_cluster['data'].append(refined_data)
        
        if refined_cluster['data']:
            first_sample = refined_cluster['data'][0].get('filtered_data', {})
            refined_cluster['sample_data'] = first_sample
            
            process_names = []
            event_names = []
            syscalls = []
            
            for item in refined_cluster['data']:
                filtered = item.get('filtered_data', {})
                if isinstance(filtered, dict):
                    if 'processName' in filtered:
                        process_names.append(filtered['processName'])
                    if 'eventName' in filtered:
                        event_names.append(filtered['eventName'])
                    if 'syscall' in filtered:
                        syscalls.append(filtered['syscall'])
            
            refined_cluster['statistics'] = {
                'unique_processes': len(set(process_names)) if process_names else 0,
                'unique_events': len(set(event_names)) if event_names else 0,
                'unique_syscalls': len(set(syscalls)) if syscalls else 0,
                'most_common_process': max(set(process_names), key=process_names.count) if process_names else None,
                'most_common_event': max(set(event_names), key=event_names.count) if event_names else None,
                'most_common_syscall': max(set(syscalls), key=syscalls.count) if syscalls else None,
            }
        
        refined_mapping['clusters'][cluster_id] = refined_cluster
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(refined_mapping, f, ensure_ascii=False, indent=2)
        print(f"정제된 결과가 {output_file}에 저장되었습니다.")
    
    return refined_mapping
