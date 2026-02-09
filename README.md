# hids-anormaly-detection

A semantic n-gram–based anomaly detection framework that integrates syscall sequences and argument semantics to efficiently detect unseen behavioral patterns through high-recall membership filtering.

## 1. Environment Setup

### 가상환경 생성 및 활성화

#### Windows (PowerShell)

```powershell
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
.\.venv\Scripts\Activate.ps1
```

또는 제공된 스크립트 사용:
```powershell
.\setup_venv.ps1
```

#### Linux / macOS

```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
source .venv/bin/activate
```

### requirements.txt 적용

가상환경이 활성화된 상태에서 다음 명령어를 실행합니다:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 설치되는 주요 패키지

- **Core ML Libraries**: torch (2.10.0), transformers (5.1.0), numpy (2.4.2), scipy (1.17.0), scikit-learn (1.8.0)
- **Clustering**: hdbscan (>=0.8.0)
- **Vector Database**: faiss-cpu (>=1.7.0)
- **Bloom Filter**: pybloom-live (>=3.0.0)
- **Data Processing**: pandas (3.0.0), tqdm (4.67.3)
- **Utilities**: joblib (1.5.3), python-dotenv (1.2.1), pyyaml (6.0.3)

## 2. Data 수집 방법

### Docker 설정

#### Windows에서 Docker 설치

1. **Docker Desktop 다운로드**
   - [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) 공식 웹사이트에서 다운로드
   - Windows 10/11 64-bit 필요
   - WSL 2 기능 활성화 필요

2. **WSL 2 설치 및 설정**
   ```powershell
   # PowerShell을 관리자 권한으로 실행 후
   wsl --install
   ```

3. **Docker Desktop 설치 및 실행**
   - 다운로드한 `Docker Desktop Installer.exe` 실행
   - 설치 완료 후 재부팅 (필요한 경우)
   - Docker Desktop 실행 후 다음 명령어로 확인:
   ```powershell
   docker --version
   docker run hello-world
   ```

### Tracee를 이용한 Systemcall JSON Logging

Tracee를 사용하여 시스템 콜(syscall)을 JSON 형식으로 로깅하는 방법:

#### Linux / WSL 2 환경

```bash
# 출력 디렉토리 생성
mkdir -p ~/tracee

# Tracee 실행 (JSON 형식으로 로깅)
docker run --name tracee_1 --rm \
  --privileged \
  --pid=host \
  --cgroupns=host \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /boot/config-$(uname -r):/boot/config-$(uname -r) \
  -v /lib/modules/$(uname -r):/lib/modules/$(uname -r) \
  -v /etc/os-release:/etc/os-release-host:ro \
  aquasec/tracee:latest \
  -o json > ~/tracee/tracee.json
```

#### 파라미터 설명

- `--privileged`: 컨테이너에 모든 호스트 디바이스 접근 권한 부여 (커널 레벨 이벤트 모니터링 필요)
- `--pid=host`: 호스트의 PID 네임스페이스 접근 (모든 프로세스 모니터링)
- `--cgroupns=host`: 호스트의 cgroup 네임스페이스 사용
- `-v /var/run/docker.sock:/var/run/docker.sock`: Docker 소켓 마운트
- `-v /boot/config-$(uname -r):/boot/config-$(uname -r)`: 커널 설정 파일 마운트
- `-v /lib/modules/$(uname -r):/lib/modules/$(uname -r)`: 커널 모듈 디렉토리 마운트
- `-o json`: 출력 형식을 JSON으로 지정

#### 데이터 파일 준비

생성된 JSON 파일을 프로젝트의 `data/` 디렉토리에 복사합니다:

```bash
# 정상 데이터 (학습용)
cp ~/tracee/tracee.json data/training_normal.json

# 공격 데이터 (검증용)
cp ~/tracee/attack_tracee.json data/attack_reverse_shell.json
```

## 3. Hyper Parameter Setup

프로젝트 루트 디렉토리에 `.env` 파일을 생성하여 하이퍼파라미터를 설정합니다.

### .env 파일 생성

프로젝트 루트에 `.env` 파일을 생성하고 다음 변수들을 설정합니다:

```env
# BERT 모델 설정
BERT_MODEL_NAME=bert-base-uncased

# 클러스터링 설정
CLUSTERING_METHOD=dbscan
CLUSTERING_ALPHA=1.0
CLUSTERING_RANDOM_STATE=42

# 데이터 필터링 설정
KEEP_PREFIXES=processName,eventName,syscall,args,executable,returnValue

# 데이터 로더 설정
BATCH_SIZE=32
SHUFFLE=False

# N-gram 및 Bloom Filter 설정
N_GRAM_SIZE_CONF=2
BLOOM_FILTER_ERROR_RATE=0.001
TOP_N_CLUSTERS=10

# 분류 임계값 설정
CLASS_THRESHOLD=0.5
```

### 환경 변수 설명

#### BERT 모델 설정

- **BERT_MODEL_NAME** (기본값: `bert-base-uncased`)
  - 사용할 BERT 모델 이름
  - Hugging Face에서 제공하는 모델 사용 가능 (예: `bert-base-uncased`, `bert-large-uncased`)
  - 임베딩 생성에 사용됨

#### 클러스터링 설정

- **CLUSTERING_METHOD** (기본값: `dbscan`)
  - 클러스터링 알고리즘 선택
  - 지원 방법: `kmeans`, `dbscan`, `hdbscan`, `agglomerative`, `meanshift`
  - `dbscan`: 밀도 기반 클러스터링, 노이즈 포인트 자동 처리
  - `hdbscan`: 계층적 DBSCAN, 더 정밀한 클러스터링

- **CLUSTERING_ALPHA** (기본값: `1.0`)
  - 클러스터링 비율 조절 파라미터
  - 범위: 0.1 ~ 2.0
  - 값이 작을수록 더 많은 클러스터 생성, 클수록 적은 클러스터 생성

- **CLUSTERING_RANDOM_STATE** (기본값: `42`)
  - 랜덤 시드 값
  - 재현 가능한 결과를 위해 설정

#### 데이터 필터링 설정

- **KEEP_PREFIXES** (기본값: `processName,eventName,syscall,args,executable,returnValue`)
  - JSON 데이터에서 유지할 필드 접두사 목록
  - 쉼표로 구분된 문자열
  - 이 필드들만 추출하여 임베딩 생성에 사용

#### 데이터 로더 설정

- **BATCH_SIZE** (기본값: `32`)
  - 배치 크기
  - 메모리 사용량과 학습 속도에 영향

- **SHUFFLE** (기본값: `False`)
  - 데이터 셔플 여부
  - `True` 또는 `False` (문자열)

#### N-gram 및 Bloom Filter 설정

- **N_GRAM_SIZE_CONF** (기본값: `2`)
  - N-gram 크기
  - 클러스터 시퀀스에서 생성할 N-gram의 N 값
  - 일반적으로 2~5 사이의 값 사용

- **BLOOM_FILTER_ERROR_RATE** (기본값: `0.001`)
  - Bloom Filter의 False Positive 확률
  - 범위: 0.0 ~ 1.0
  - 값이 작을수록 메모리 사용량 증가, 정확도 향상
  - 권장값: 0.001 (0.1%)

- **TOP_N_CLUSTERS** (기본값: `10`)
  - 통계 출력 시 상위 N개 클러스터 표시
  - 분석 및 디버깅용

#### 분류 임계값 설정

- **CLASS_THRESHOLD** (기본값: `0.5`)
  - 이상 탐지 분류 임계값
  - 코사인 거리가 이 값보다 크면 `unknown`으로 분류
  - 범위: 0.0 ~ 2.0 (코사인 거리 범위)

## 4. Training 방법

`module/train_normal_data.py`를 사용하여 정상 데이터로 모델을 학습합니다.

### 기본 사용법

```bash
# 프로젝트 루트에서 실행
cd module
python train_normal_data.py 
```

 default 값으로는 기본적으로 다음 경로의 데이터를 사용합니다:
- 정상 데이터: `../data/normal_tracee.json`
- 공격 데이터: `../data/attack_tracee.json` (선택적)

### 사용자 지정 데이터 파일 사용

```bash
# 사용자 지정 경로의 데이터 파일 사용
python train_normal_data.py --path ../data/training_normal.json
```

또는 짧은 옵션:

```bash
python train_normal_data.py -p ../data/training_normal.json
```

### 학습 프로세스

학습은 다음 단계로 진행됩니다:

1. **설정 로드**: `.env` 파일에서 하이퍼파라미터 로드
2. **텍스트 필터 초기화**: `KEEP_PREFIXES`에 지정된 필드만 추출
3. **데이터 로더 생성**: JSON 데이터를 배치로 로드
4. **임베딩 생성**: BERT 모델을 사용하여 시퀀스 임베딩 생성
5. **Vector DB 저장**: 생성된 임베딩을 FAISS Vector DB에 저장
6. **클러스터링 수행**: 저장된 임베딩을 클러스터링하여 정상 패턴 그룹화
7. **N-gram 생성**: 클러스터 시퀀스에서 N-gram 생성
8. **Bloom Filter 생성**: 생성된 N-gram을 Bloom Filter에 추가
9. **모델 데이터 저장**: 학습된 모델을 `module/model_data/` 디렉토리에 저장

### 출력 파일

학습 완료 후 다음 파일들이 생성됩니다:

- `module/model_data/cluster_labels.npy`: 클러스터 레이블 배열
- `module/model_data/cluster_info.json`: 클러스터 정보 (centroids, max_distances 등)
- `module/model_data/clusterer.pkl`: 클러스터링 모델 객체
- `module/model_data/ngrams.json`: N-gram 시퀀스
- `module/model_data/ngram_stats.json`: N-gram 통계 정보
- `module/model_data/bloom_filter.pkl`: Bloom Filter 객체
- `module/model_data/config.json`: 학습에 사용된 설정 정보
- `module/faiss_db/`: FAISS Vector DB 파일들

## 5. Inference 방법

`module/inference_data.py`를 사용하여 실제 데이터에 대한 공격 여부를 판별합니다.

### 기본 사용법

```bash
# 프로젝트 루트에서 실행
cd module
python inference_data.py
```

default 설정으로  다음 경로의 공격 데이터를 사용합니다:
- 공격 데이터: `../data/attack_tracee.json`

### 사용자 지정 데이터 파일 사용

```bash
# 사용자 지정 경로의 공격 데이터 파일 사용
python inference_data.py --path ../data/attack_reverse_shell.json
```

또는 짧은 옵션:

```bash
python inference_data.py -p ./my_attack_data.json
```

### 추론 프로세스

추론은 다음 단계로 진행됩니다:

1. **설정 로드**: `.env` 파일에서 하이퍼파라미터 로드
2. **학습된 모델 로드**: `module/model_data/` 디렉토리에서 모델 로드
3. **공격 데이터 로더 생성**: 검증할 공격 데이터를 배치로 로드
4. **임베딩 생성**: 공격 데이터에 대한 BERT 임베딩 생성
5. **유사 임베딩 검색**: Vector DB에서 유사한 정상 임베딩 검색 (Top-K)
6. **클래스 분류**: 검색된 임베딩의 클러스터 정보를 기반으로 분류
   - 클러스터의 `max_distance` 내에 있으면 해당 클러스터로 분류
   - `max_distance` 바깥이면 `CLASS_THRESHOLD`와 비교하여 `unknown` 처리
7. **N-gram 생성 및 Bloom Filter 확인**: 분류된 클러스터 시퀀스에서 N-gram 생성 후 Bloom Filter에서 확인
8. **이상 탐지**: Bloom Filter에 없는 N-gram을 이상으로 탐지

### 결과 해석

추론 결과는 다음과 같이 해석할 수 있습니다:

- **클래스 분류 결과**:
  - 정수 클러스터 ID: 정상 패턴으로 분류됨
  - `unknown`: 학습되지 않은 패턴으로 판단

- **Bloom Filter 매칭 결과**:
  - **매칭된 N-gram**: 정상 패턴으로 학습된 시퀀스
  - **미매칭 N-gram**: 이상 패턴으로 탐지됨

- **이상 탐지 인덱스**: Bloom Filter에 매칭되지 않은 N-gram의 시작 인덱스 리스트

### 출력 예시

```
분류 결과 통계:
  클래스 5: 120개
  클래스 3: 85개
  unknown: 15개

N-gram 분석 결과 요약:
  - 총 N-gram 수: 220개
  - Bloom Filter 매칭: 180개 (81.82%)
  - Bloom Filter 미매칭: 40개 (18.18%)
  - 이상 탐지된 인덱스: 40개
```

## 추가 정보

### 프로젝트 구조

```
hids-anormaly-detection/
├── module/
│   ├── train_normal_data.py      # 학습 스크립트
│   ├── inference_data.py          # 추론 스크립트
│   ├── model_data/                 # 학습된 모델 저장 디렉토리
│   ├── faiss_db/                  # Vector DB 저장 디렉토리
│   ├── dataloader.py              # 데이터 로더
│   ├── embedder.py                # BERT 임베딩 생성기
│   ├── clustering.py              # 클러스터링 모듈
│   ├── sequence.py                # N-gram 생성 모듈
│   ├── bloomfilter.py             # Bloom Filter 모듈
│   ├── vector_db.py               # Vector DB 모듈
│   └── utility.py                 # 유틸리티 및 설정 관리
├── data/                          # 데이터 파일 디렉토리
├── requirements.txt               # Python 패키지 의존성
├── .env                          # 환경 변수 설정 파일
└── README.md                      # 프로젝트 문서
```

### 문제 해결

#### 가상환경 활성화 오류 (Windows)

PowerShell에서 실행 정책 오류가 발생하는 경우:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Docker 권한 오류

Linux에서 Docker 권한 오류가 발생하는 경우:

```bash
sudo usermod -aG docker $USER
# 로그아웃 후 다시 로그인
```

#### 모델 파일을 찾을 수 없음

학습을 먼저 수행하여 `module/model_data/` 디렉토리에 모델 파일이 생성되어 있는지 확인하세요.
