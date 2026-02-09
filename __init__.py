"""
HIDS 이상 탐지 모델 패키지

이 패키지는 BERT 임베딩 기반 클러스터링과 N-gram Bloom Filter를 사용한
HIDS(Host-based Intrusion Detection System) 이상 탐지 모델을 제공합니다.
"""

from .utility import Config, TextFilter
from .embedder import BERTEmbedder
from .dataloader import TraceeDataset, DataLoaderFactory
from .clustering import Clusterer, ClusterTracer, refine_cluster_mapping
from .sequence import SequenceMatcher, NGramGenerator
from .bloomfilter import NGramBloomFilter, create_ngram_bloom_filter, check_ngram_in_bloom, check_ngrams_batch, analyze_bloom_filter_performance

__all__ = [
    # Utility
    'Config',
    'TextFilter',
    'BERTEmbedder',
    # DataLoader
    'TraceeDataset',
    'DataLoaderFactory',
    # Clustering
    'Clusterer',
    'ClusterTracer',
    'refine_cluster_mapping',
    # Sequence
    'SequenceMatcher',
    'NGramGenerator',
    # Bloom Filter
    'NGramBloomFilter',
    'create_ngram_bloom_filter',
    'check_ngram_in_bloom',
    'check_ngrams_batch',
    'analyze_bloom_filter_performance',
]

__version__ = '1.0.0'
