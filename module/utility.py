"""
유틸리티 함수 및 설정 관리 모듈
"""
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from typing import Tuple, Optional


class Config:
    """설정 관리 클래스"""
    
    def __init__(self, env_file: str = None):
        """
        Args:
            env_file: .env 파일 경로 (None이면 프로젝트 루트에서 자동으로 찾음)
        """
        if env_file is None:
            # 프로젝트 루트 디렉토리 찾기 (현재 파일의 위치에서 상위로 올라가며 .env 파일 찾기)
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent  # module/utility.py -> module/ -> project_root/
            env_file = project_root / '.env'
        
        # 문자열로 변환 (load_dotenv는 문자열 경로를 받음)
        env_file_str = str(env_file) if isinstance(env_file, Path) else env_file
        
        # .env 파일 로드
        load_dotenv(env_file_str, override=True)
        self._load_config()
    
    def _load_config(self):
        """환경변수에서 설정 로드"""
        self.BERT_MODEL_NAME = os.getenv('BERT_MODEL_NAME', 'bert-base-uncased')
        self.CLUSTERING_ALPHA = float(os.getenv('CLUSTERING_ALPHA', '1.0'))
        self.CLUSTERING_METHOD = os.getenv('CLUSTERING_METHOD', 'dbscan')
        self.CLUSTERING_RANDOM_STATE = int(os.getenv('CLUSTERING_RANDOM_STATE', '42'))
        
        keep_prefixes_str = os.getenv('KEEP_PREFIXES', 'processName,eventName,syscall,args,executable,returnValue')
        self.KEEP_PREFIXES = tuple([prefix.strip() for prefix in keep_prefixes_str.split(',')])
        
        self.BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
        self.SHUFFLE = os.getenv('SHUFFLE', 'False').lower() == 'true'
        self.TOP_N_CLUSTERS = int(os.getenv('TOP_N_CLUSTERS', '10'))
        self.N_GRAM_SIZE = int(os.getenv('N_GRAM_SIZE_CONF', '2'))
        self.BLOOM_FILTER_ERROR_RATE = float(os.getenv('BLOOM_FILTER_ERROR_RATE', '0.001'))
        self.CLASS_THRESHOLD = float(os.getenv('CLASS_THRESHOLD', '0.5'))
    
    def print_config(self):
        """설정 정보 출력"""
        print("=" * 80)
        print("설정 로드 완료")
        print("=" * 80)
        print(f"BERT 모델: {self.BERT_MODEL_NAME}")
        print(f"클러스터링 Alpha: {self.CLUSTERING_ALPHA}")
        print(f"클러스터링 Method: {self.CLUSTERING_METHOD}")
        print(f"Random State: {self.CLUSTERING_RANDOM_STATE}")
        print(f"Keep Prefixes: {self.KEEP_PREFIXES}")
        print(f"Batch Size: {self.BATCH_SIZE}")
        print(f"Shuffle: {self.SHUFFLE}")
        print(f"Top N Clusters: {self.TOP_N_CLUSTERS}")
        print(f"N-gram Size: {self.N_GRAM_SIZE}")
        print(f"Bloom Filter Error Rate: {self.BLOOM_FILTER_ERROR_RATE} ({self.BLOOM_FILTER_ERROR_RATE*100:.3f}%)")
        print(f"Class Threshold: {self.CLASS_THRESHOLD}")
        print("=" * 80)


class TextFilter:
    """텍스트 필터링 클래스"""
    
    def __init__(self, keep_prefixes: Optional[Tuple[str, ...]] = None):
        """
        Args:
            keep_prefixes: 유지할 키의 접두사 튜플
        """
        self.keep_prefixes = keep_prefixes
        self.kv_pattern = re.compile(r'"([^"]+)":\s*"([^"]*)"')
    
    def filter(self, raw_text: str, keep_keys: Optional[Tuple[str, ...]] = None) -> str:
        """
        텍스트에서 지정된 키만 남기는 필터링 함수
        
        Args:
            raw_text: 원본 JSON 문자열
            keep_keys: 유지할 키의 튜플 (None이면 self.keep_prefixes 사용)
        
        Returns:
            필터링된 텍스트 문자열
        """
        if keep_keys is None:
            keep_keys = self.keep_prefixes
        
        if keep_keys is None:
            return raw_text
        
        pairs = self.kv_pattern.findall(raw_text)
        kept = []
        for key, value in pairs:
            if key.startswith(keep_keys):
                kept.append(f"{key}={value}")
        return " ".join(kept)
