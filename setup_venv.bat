@echo off
REM 가상환경 설정 배치 파일
echo ========================================
echo 가상환경 설정 시작
echo ========================================

REM 가상환경이 없으면 생성
if not exist .venv (
    echo 가상환경 생성 중...
    python -m venv .venv
    if errorlevel 1 (
        echo 가상환경 생성 실패!
        exit /b 1
    )
    echo 가상환경 생성 완료!
) else (
    echo 가상환경이 이미 존재합니다.
)

REM 가상환경 활성화
echo 가상환경 활성화 중...
call .venv\Scripts\activate.bat

REM pip 업그레이드
echo pip 업그레이드 중...
python -m pip install --upgrade pip

REM 패키지 설치
echo 패키지 설치 중...
if exist requirements.txt (
    pip install -r requirements.txt
    echo 패키지 설치 완료!
) else (
    echo requirements.txt 파일이 없습니다. 기본 패키지를 설치합니다...
    pip install torch transformers numpy scipy scikit-learn hdbscan python-dotenv pybloom-live faiss-cpu pandas tqdm joblib
    echo 기본 패키지 설치 완료!
)

echo ========================================
echo 가상환경 설정 완료!
echo ========================================
echo.
echo 가상환경을 활성화하려면 다음 명령을 실행하세요:
echo   .venv\Scripts\activate.bat
echo.
echo Jupyter Notebook에서 사용하려면:
echo   python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
pause
