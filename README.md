<div style="font-size: 0.9em;">

# hids-anormaly-detection
A semantic n-gram–based anomaly detection framework that integrates syscall sequences and argument semantics to efficiently detect unseen behavioral patterns through high-recall membership filtering.


## How to install docker on windows

Windows에서 Docker를 설치하는 방법:

### 방법 1: Docker Desktop 설치 (권장)

1. **Docker Desktop 다운로드**
   - [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) 공식 웹사이트에서 다운로드
   - 또는 [Docker Hub](https://hub.docker.com/editions/community/docker-ce-desktop-windows)에서 다운로드

2. **시스템 요구사항 확인**
   - Windows 10 64-bit: Pro, Enterprise, or Education (Build 19041 이상)
   - Windows 11 64-bit: Home 또는 Pro 버전
   - WSL 2 기능 활성화 필요
   - 가상화 기능 활성화 필요 (BIOS/UEFI 설정)

3. **WSL 2 설치 및 설정**
   ```powershell
   # PowerShell을 관리자 권한으로 실행 후 다음 명령어 실행
   wsl --install
   ```
   - 또는 수동으로:
     - Windows 기능에서 "Linux용 Windows 하위 시스템" 활성화
     - "가상 머신 플랫폼" 활성화
     - Microsoft Store에서 Ubuntu 또는 다른 Linux 배포판 설치

4. **Docker Desktop 설치**
   - 다운로드한 `Docker Desktop Installer.exe` 실행
   - 설치 마법사 따라 진행
   - 설치 완료 후 재부팅 (필요한 경우)

5. **Docker Desktop 실행 및 확인**
   - Docker Desktop 실행
   - 시스템 트레이에서 Docker 아이콘 확인
   - PowerShell 또는 명령 프롬프트에서 확인:
   ```powershell
   docker --version
   docker run hello-world
   ```

### 방법 2: WSL 2에서 직접 Docker 설치

WSL 2 Ubuntu 환경에서 직접 Docker를 설치할 수도 있습니다:

```bash
# WSL 2 Ubuntu 터미널에서 실행
sudo apt update
sudo apt install -y docker.io
sudo service docker start
```

### 문제 해결

- **WSL 2 업데이트**: `wsl --update` 실행
- **가상화 확인**: 작업 관리자 > 성능 탭에서 가상화 활성화 여부 확인
- **Hyper-V 활성화**: Windows 기능에서 Hyper-V 활성화 (Pro 버전 이상)

## Logging syscall using tracee

Tracee를 사용하여 시스템 콜(syscall)을 로깅하는 방법:

```bash
docker run --name tracee_1 --rm \
  --privileged \
  --pid=host \
  --cgroupns=host \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /boot/config-$(uname -r):/boot/config-$(uname -r) \
  -v /lib/modules/$(uname -r):/lib/modules/$(uname -r) \
  -v /etc/os-release:/etc/os-release-host:ro \
  aquasec/tracee:latest \
  -o json > /home/sylee/tracee/tracee.json
```

### 파라미터 설명

#### Docker 기본 옵션
- `--name tracee_1`: 컨테이너 이름을 `tracee_1`로 지정합니다.
- `--rm`: 컨테이너가 종료되면 자동으로 삭제됩니다.

#### 권한 및 네임스페이스 옵션
- `--privileged`: 컨테이너에 모든 호스트 디바이스에 대한 접근 권한을 부여합니다. Tracee가 커널 레벨 이벤트를 모니터링하기 위해 필요합니다.
- `--pid=host`: 컨테이너가 호스트의 PID 네임스페이스에 접근할 수 있도록 합니다. 모든 프로세스를 모니터링하기 위해 필요합니다.
- `--cgroupns=host`: 컨테이너가 호스트의 cgroup 네임스페이스를 사용하도록 합니다. 리소스 제한 및 프로세스 그룹 정보에 접근하기 위해 필요합니다.

#### 볼륨 마운트 옵션
- `-v /var/run/docker.sock:/var/run/docker.sock`: Docker 소켓을 마운트하여 컨테이너 내부에서 Docker API에 접근할 수 있게 합니다. Docker 컨테이너 이벤트를 모니터링하기 위해 필요합니다.
- `-v /boot/config-$(uname -r):/boot/config-$(uname -r)`: 현재 실행 중인 커널의 설정 파일을 마운트합니다. `$(uname -r)`는 현재 커널 버전을 반환합니다. Tracee가 커널 구성을 읽기 위해 필요합니다.
- `-v /lib/modules/$(uname -r):/lib/modules/$(uname -r)`: 현재 커널 버전의 모듈 디렉토리를 마운트합니다. 커널 모듈에 접근하기 위해 필요합니다.
- `-v /etc/os-release:/etc/os-release-host:ro`: OS 릴리스 정보 파일을 읽기 전용으로 마운트합니다. 호스트 OS 정보를 확인하기 위해 사용됩니다.

#### Tracee 옵션
- `aquasec/tracee:latest`: 사용할 Tracee Docker 이미지를 지정합니다. Aqua Security에서 제공하는 최신 버전입니다.
- `-o json`: 출력 형식을 JSON으로 지정합니다. 구조화된 데이터로 로그를 저장할 수 있습니다.
- `> /home/sylee/tracee/tracee.json`: JSON 출력을 지정된 파일 경로로 리다이렉션합니다. 파일이 없으면 생성되고, 있으면 덮어씁니다.

### 사용 전 준비사항

1. 출력 디렉토리 생성:
   ```bash
   mkdir -p /home/sylee/tracee
   ```

2. 파일 쓰기 권한 확인:
   ```bash
   chmod 755 /home/sylee/tracee
   ```

### Windows에서 사용 시 주의사항

Windows에서는 Linux 경로가 다르므로 WSL 2 환경에서 실행하거나 경로를 수정해야 합니다:
```bash
# WSL 2에서 실행 시
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


## How to generate abnormal behaior syscall log

</div>