#!/bin/bash
# ============================================================
# SAM 3 데이터셋 다운로드 스크립트
# SA-1B (3개 tar, ~33GB) + SA-V (1개 tar, ~8GB)
# ============================================================
set -e

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
SA1B_DIR="$DATA_DIR/sa1b"
SAV_DIR="$DATA_DIR/sa_v"

mkdir -p "$SA1B_DIR" "$SAV_DIR"

echo "============================================================"
echo "  SAM 3 데이터셋 다운로드"
echo "  저장 위치: $DATA_DIR"
echo "============================================================"

# ── SA-1B 다운로드 (3개 tar, 각 ~11GB) ──────────────────

SA1B_URLS=(
  "https://scontent.xx.fbcdn.net/m1/v/t6/An_YmP5OIPXun-vu3hkckAZZ2s4lPYoVkiyvCcWiVY21mu1Ng5_1HeCa2CWiSTsskj8HQ8bN013HxNpYDdSC_7jWQq_svcg.tar?_nc_gid&ccb=10-5&oh=00_Afs8LIXUCJcRjgXp1RQScW5d0Llkp7sNKV2F-nd7e2cwJg&oe=69AFC8E8&_nc_sid=0fdd51"
  "https://scontent.xx.fbcdn.net/m1/v/t6/An_pa489zcoOeY_kCzFmHPyiKL-X9Qo-FImAUBBWUmMI9DA8JnvoUPvuV_EpnYUMT_SbP54uo1zo9-h9vSYWRWSu_pBPuL0.tar?_nc_gid&ccb=10-5&oh=00_AfuCQ1rexk_i4mCx67iVpdKFZ5YGghzIXFU9WXlB2SihkQ&oe=69AFA53A&_nc_sid=2"
  "https://scontent.xx.fbcdn.net/m1/v/t6/An-szUPLg5rMWT5xsliXsNN_8IzC8sJlIGj3bSB6rtcLXqy1bXxU1hqfdwk4M8SAZvluZdnlwRqmICIEVx40f2FDe-Z8pQI.tar?_nc_gid&ccb=10-5&oh=00_AfsVpLKD2oHBmFSRrR9CJSbOxii6h3yUc4_lIVlM2HwP3g&oe=69AFC6DB&_nc_sid=0fdd51"
)
SA1B_NAMES=("sa_000020.tar" "sa_000097.tar" "sa_000524.tar")

echo ""
echo "[SA-1B] 3개 tar 파일 다운로드 시작 (~33GB 총)"
for i in "${!SA1B_URLS[@]}"; do
  NAME="${SA1B_NAMES[$i]}"
  URL="${SA1B_URLS[$i]}"
  DEST="$SA1B_DIR/$NAME"

  echo "  [$((i+1))/3] $NAME 다운로드 중... (이어받기 지원)"
  curl -L -C - -o "$DEST" "$URL"
  echo "  [$((i+1))/3] $NAME 완료!"
done

# ── SA-V 다운로드 (1개 tar, ~8GB) ────────────────────────

SAV_URL="https://scontent.xx.fbcdn.net/m1/v/t6/An-tPiXhtjFu_KnO3KjjcMReWlc1m_oYKcpw7Bj7r0Oa32_oMYC-Frt17lRsjwBWNqRYJNsFKh_Xud4jNd6evmU.tar?_nc_gid&ccb=10-5&oh=00_Afvej4LcHZeEaFO1Lhnl1_WWzu3nY05lBHs3Au8lkfl-ug&oe=69AFA652&_nc_sid=7b5a27"
SAV_NAME="sav_000.tar"
SAV_DEST="$SAV_DIR/$SAV_NAME"

echo ""
echo "[SA-V] 1개 tar 파일 다운로드 시작 (~8GB)"
echo "  $SAV_NAME 다운로드 중... (이어받기 지원)"
curl -L -C - -o "$SAV_DEST" "$SAV_URL"
echo "  $SAV_NAME 완료!"

# ── 압축 해제 ─────────────────────────────────────────────

echo ""
echo "[압축 해제] tar 파일을 풀고 있습니다..."

for TAR in "$SA1B_DIR"/*.tar; do
  [ -f "$TAR" ] || continue
  echo "  $(basename $TAR) 해제 중..."
  tar -xf "$TAR" -C "$SA1B_DIR/"
  echo "  $(basename $TAR) 해제 완료 → 원본 tar 삭제"
  rm "$TAR"
done

for TAR in "$SAV_DIR"/*.tar; do
  [ -f "$TAR" ] || continue
  echo "  $(basename $TAR) 해제 중..."
  tar -xf "$TAR" -C "$SAV_DIR/"
  echo "  $(basename $TAR) 해제 완료 → 원본 tar 삭제"
  rm "$TAR"
done

# ── 완료 ──────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  다운로드 및 압축 해제 완료!"
echo "  SA-1B: $(find "$SA1B_DIR" -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ') images"
echo "  SA-V:  $(find "$SAV_DIR" -type d -mindepth 1 2>/dev/null | wc -l | tr -d ' ') items"
echo "  디스크 사용량:"
du -sh "$SA1B_DIR" "$SAV_DIR"
echo "============================================================"
