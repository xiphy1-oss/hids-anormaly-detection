# 가상환경 설정 스크립트
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "가상환경 설정 시작" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 가상환경이 없으면 생성
if (-not (Test-Path .venv)) {
    Write-Host "가상환경 생성 중..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "가상환경 생성 실패!" -ForegroundColor Red
        exit 1
    }
    Write-Host "가상환경 생성 완료!" -ForegroundColor Green
} else {
    Write-Host "가상환경이 이미 존재합니다." -ForegroundColor Green
}

# 가상환경 활성화
Write-Host "가상환경 활성화 중..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# pip 업그레이드
Write-Host "pip 업그레이드 중..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 패키지 설치
Write-Host "패키지 설치 중..." -ForegroundColor Yellow
if (Test-Path requirements.txt) {
    pip install -r requirements.txt
    Write-Host "패키지 설치 완료!" -ForegroundColor Green
} else {
    Write-Host "requirements.txt 파일이 없습니다. 기본 패키지를 설치합니다..." -ForegroundColor Yellow
    pip install torch transformers numpy scipy scikit-learn hdbscan python-dotenv pybloom-live faiss-cpu pandas tqdm joblib
    Write-Host "기본 패키지 설치 완료!" -ForegroundColor Green
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "가상환경 설정 완료!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "가상환경을 활성화하려면 다음 명령을 실행하세요:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Jupyter Notebook에서 사용하려면:" -ForegroundColor Yellow
Write-Host "  python -m ipykernel install --user --name=.venv --display-name 'Python (.venv)'" -ForegroundColor White
