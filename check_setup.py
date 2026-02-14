import torch
import executorch
import transformers
import sys
import platform

print("\n" + "="*40)
print("     🛠️ SAM 3 모바일 환경 점검 레포트     ")
print("="*40)

# 1. 시스템 정보
print(f"✅ Python 버전: {sys.version.split()[0]}")
print(f"✅ OS 플랫폼: {platform.platform()}")

# 2. PyTorch 및 M4 Pro GPU (MPS) 확인
print(f"✅ PyTorch 버전: {torch.__version__}")
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print("🚀 [성공] Apple M4 Pro GPU (MPS) 가속 활성화됨!")
else:
    print("❌ [실패] MPS 가속을 사용할 수 없습니다. PyTorch Nightly 설치를 확인하세요.")

# 3. 필수 라이브러리 확인
try:
    print(f"✅ ExecuTorch 버전: {executorch.__version__} (설치 성공)")
except ImportError:
    print("❌ [실패] ExecuTorch가 설치되지 않았습니다.")

try:
    print(f"✅ Transformers 버전: {transformers.__version__} (설치 성공)")
except ImportError:
    print("❌ [실패] Transformers가 설치되지 않았습니다.")

print("="*40 + "\n")
