#!/bin/bash
# 이미지 웹 최적화 (긴 변 기준 축소 + 재압축)
# 사용법:
#   ./scripts/optimize-image.sh assets/images/foo.jpg          # 긴 변 1000px
#   ./scripts/optimize-image.sh assets/images/foo.jpg 720      # 긴 변 720px
# 프로필 사진은 1000, 프로젝트 카드 이미지는 720 권장.

set -e

FILE="$1"
MAX="${2:-1000}"

if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
  echo "파일 없음: $FILE"
  echo "사용법: $0 <이미지경로> [긴변px=1000]"
  exit 1
fi

before=$(du -h "$FILE" | cut -f1)
dim=$(sips -g pixelWidth -g pixelHeight "$FILE" | awk '/pixel/{print $2}' | paste -sd x -)

sips -Z "$MAX" -s formatOptions 80 "$FILE" >/dev/null

after=$(du -h "$FILE" | cut -f1)
newdim=$(sips -g pixelWidth -g pixelHeight "$FILE" | awk '/pixel/{print $2}' | paste -sd x -)

echo "$FILE"
echo "  $dim ($before)  ->  $newdim ($after)"
