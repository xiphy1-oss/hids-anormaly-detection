"""
데이터 로더 모듈
"""
import json
import os
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Optional, Callable


class TraceeDataset(Dataset):
    """Tracee 데이터셋 클래스"""
    
    def __init__(self, data_path: str, filter_func: Optional[Callable[[str], str]] = None):
        """
        Args:
            data_path: JSON Lines 파일 경로
            filter_func: 텍스트 필터링 함수 (선택사항)
        """
        self.data = []
        self.data_path = data_path
        self.filter_func = filter_func
        self.original_data = []
        
        self._load_data()
    
    def _load_data(self):
        """데이터 로드"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        json_obj = json.loads(line)
                        self.original_data.append(json_obj)
                        
                        text = json.dumps(json_obj, ensure_ascii=False)
                        if self.filter_func:
                            text = self.filter_func(text)
                        self.data.append(text)
                    except json.JSONDecodeError:
                        continue
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> str:
        return self.data[idx]


class DataLoaderFactory:
    """데이터 로더 팩토리 클래스"""
    
    @staticmethod
    def create(
        split: Literal["train", "validation"],
        data_type: Literal["attack", "normal"],
        batch_size: int = 32,
        shuffle: bool = True,
        filter_func: Optional[Callable[[str], str]] = None,
        num_workers: int = 0,
        data_dir: str = "data"
    ) -> DataLoader:
        """
        데이터 로더를 생성하는 함수
        
        Args:
            split: "train" 또는 "validation"
            data_type: "attack" 또는 "normal"
            batch_size: 배치 크기
            shuffle: 셔플 여부
            filter_func: 텍스트 필터링 함수 (선택사항)
            num_workers: 데이터 로딩 워커 수
            data_dir: 데이터 디렉토리 경로
        
        Returns:
            DataLoader 객체
        """
        prefix = "tr" if split == "train" else "val"
        filename = f"{prefix}_{data_type}_tracee.json"
        data_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        
        dataset = TraceeDataset(data_path, filter_func=filter_func)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda x: x
        )
        
        return dataloader
