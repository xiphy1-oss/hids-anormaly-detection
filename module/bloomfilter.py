"""
Bloom Filter 모듈
"""
import warnings
from typing import List, Tuple, Optional, Dict, Any

try:
    from pybloom_live import BloomFilter
    BLOOM_FILTER_AVAILABLE = True
except ImportError:
    try:
        from pybloom import BloomFilter
        BLOOM_FILTER_AVAILABLE = True
    except ImportError:
        BLOOM_FILTER_AVAILABLE = False
        BloomFilter = None


class NGramBloomFilter:
    """N-gram Bloom Filter 클래스"""
    
    def __init__(self, error_rate: float = 0.001):
        """
        Args:
            error_rate: False positive 확률 (기본값: 0.001 = 0.1%)
        """
        if not BLOOM_FILTER_AVAILABLE:
            raise ImportError(
                "Bloom Filter 라이브러리가 설치되지 않았습니다. "
                "'pip install pybloom-live' 또는 'pip install pybloom'을 실행하세요."
            )
        
        if error_rate <= 0 or error_rate >= 1:
            raise ValueError(f"error_rate는 0과 1 사이의 값이어야 합니다. 현재 값: {error_rate}")
        
        if error_rate < 1e-10:
            warnings.warn(
                f"매우 작은 error_rate ({error_rate})를 사용하면 메모리 사용량이 극도로 증가할 수 있습니다. "
                f"일반적으로 0.001 (0.1%) 이상의 값을 권장합니다.",
                UserWarning
            )
        
        self.error_rate = error_rate
        self.bloom_filter: Optional[BloomFilter] = None
        self.capacity_ = None
    
    def create(self, ngrams: List[Tuple]) -> BloomFilter:
        """
        N-gram 리스트로부터 Bloom filter를 생성
        
        Args:
            ngrams: n-gram 리스트 (튜플들의 리스트)
        
        Returns:
            생성된 Bloom filter 객체
        """
        unique_ngrams = set(ngrams)
        capacity = len(unique_ngrams)
        
        self.bloom_filter = BloomFilter(capacity=capacity, error_rate=self.error_rate)
        self.capacity_ = capacity
        
        for ngram in unique_ngrams:
            self.bloom_filter.add(str(ngram))
        
        return self.bloom_filter
    
    def check(self, ngram: Tuple) -> bool:
        """
        N-gram이 Bloom filter에 존재하는지 확인
        
        Args:
            ngram: 확인할 n-gram (튜플)
        
        Returns:
            True면 존재 가능 (False positive 가능), False면 확실히 없음
        """
        if self.bloom_filter is None:
            raise ValueError("Bloom filter가 생성되지 않았습니다. 먼저 create()를 실행하세요.")
        
        return str(ngram) in self.bloom_filter
    
    def check_batch(self, ngrams: List[Tuple]) -> List[bool]:
        """
        여러 N-gram을 한 번에 체크
        
        Args:
            ngrams: 확인할 n-gram 리스트
        
        Returns:
            각 n-gram의 존재 여부 (True/False) 리스트
        """
        return [self.check(ngram) for ngram in ngrams]
    
    def analyze_performance(
        self,
        known_ngrams: List[Tuple],
        test_ngrams: List[Tuple]
    ) -> Dict[str, Any]:
        """
        Bloom filter의 성능을 분석
        
        Args:
            known_ngrams: Bloom filter에 추가된 n-gram 리스트 (실제로 존재해야 함)
            test_ngrams: 테스트할 n-gram 리스트
        
        Returns:
            성능 통계 딕셔너리
        """
        if self.bloom_filter is None:
            raise ValueError("Bloom filter가 생성되지 않았습니다. 먼저 create()를 실행하세요.")
        
        known_set = set(known_ngrams)
        
        tp = sum(1 for ngram in test_ngrams 
                if ngram in known_set and self.check(ngram))
        
        fp = sum(1 for ngram in test_ngrams 
                if ngram not in known_set and self.check(ngram))
        
        tn = sum(1 for ngram in test_ngrams 
                if ngram not in known_set and not self.check(ngram))
        
        fn = sum(1 for ngram in test_ngrams 
                if ngram in known_set and not self.check(ngram))
        
        total = len(test_ngrams)
        
        stats = {
            'total_tested': total,
            'true_positive': tp,
            'false_positive': fp,
            'true_negative': tn,
            'false_negative': fn,
            'false_positive_rate': fp / total if total > 0 else 0.0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0
        }
        
        return stats
    
    def print_info(self):
        """Bloom filter 정보 출력"""
        if self.bloom_filter is None:
            print("Bloom filter가 생성되지 않았습니다.")
            return
        
        print(f"Bloom Filter 크기: {self.capacity_}개 용량")
        print(f"False Positive Rate: {self.error_rate * 100:.3f}%")


def create_ngram_bloom_filter(ngrams: List[Tuple], error_rate: float = 0.001) -> BloomFilter:
    """
    N-gram 리스트로부터 Bloom filter를 생성하는 함수 (호환성을 위한 함수)
    
    Args:
        ngrams: n-gram 리스트 (튜플들의 리스트)
        error_rate: False positive 확률 (기본값: 0.001 = 0.1%)
    
    Returns:
        생성된 Bloom filter 객체
    """
    bloom = NGramBloomFilter(error_rate=error_rate)
    return bloom.create(ngrams)


def check_ngram_in_bloom(bloom_filter: BloomFilter, ngram: Tuple) -> bool:
    """
    N-gram이 Bloom filter에 존재하는지 확인하는 함수 (호환성을 위한 함수)
    
    Args:
        bloom_filter: BloomFilter 객체
        ngram: 확인할 n-gram (튜플)
    
    Returns:
        True면 존재 가능 (False positive 가능), False면 확실히 없음
    """
    return str(ngram) in bloom_filter


def check_ngrams_batch(bloom_filter: BloomFilter, ngrams: List[Tuple]) -> List[bool]:
    """
    여러 N-gram을 한 번에 체크하는 함수 (호환성을 위한 함수)
    
    Args:
        bloom_filter: BloomFilter 객체
        ngrams: 확인할 n-gram 리스트
    
    Returns:
        각 n-gram의 존재 여부 (True/False) 리스트
    """
    return [check_ngram_in_bloom(bloom_filter, ngram) for ngram in ngrams]


def analyze_bloom_filter_performance(
    bloom_filter: BloomFilter,
    known_ngrams: List[Tuple],
    test_ngrams: List[Tuple]
) -> Dict[str, Any]:
    """
    Bloom filter의 성능을 분석하는 함수 (호환성을 위한 함수)
    
    Args:
        bloom_filter: BloomFilter 객체
        known_ngrams: Bloom filter에 추가된 n-gram 리스트 (실제로 존재해야 함)
        test_ngrams: 테스트할 n-gram 리스트
    
    Returns:
        성능 통계 딕셔너리
    """
    known_set = set(known_ngrams)
    
    tp = sum(1 for ngram in test_ngrams 
            if ngram in known_set and check_ngram_in_bloom(bloom_filter, ngram))
    
    fp = sum(1 for ngram in test_ngrams 
            if ngram not in known_set and check_ngram_in_bloom(bloom_filter, ngram))
    
    tn = sum(1 for ngram in test_ngrams 
            if ngram not in known_set and not check_ngram_in_bloom(bloom_filter, ngram))
    
    fn = sum(1 for ngram in test_ngrams 
            if ngram in known_set and not check_ngram_in_bloom(bloom_filter, ngram))
    
    total = len(test_ngrams)
    
    stats = {
        'total_tested': total,
        'true_positive': tp,
        'false_positive': fp,
        'true_negative': tn,
        'false_negative': fn,
        'false_positive_rate': fp / total if total > 0 else 0.0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0
    }
    
    return stats
