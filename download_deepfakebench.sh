#!/usr/bin/env bash
# Tải FaceForensics++.zip đã xử lý sẵn của DeepfakeBench (frame mặt + mask + landmark).
# Tải tiếp được khi bị ngắt; chạy lại cùng lệnh là tiếp tục từ chỗ dừng.
# Chạy (Git Bash): bash download_deepfakebench.sh
cd "$(dirname "$0")"
URL='https://drive.usercontent.google.com/download?id=1mZ9NNtgW_4oo9S996uQh9-SmRYaLxPnb&export=download&confirm=t'
OUT='data/deepfakebench/FaceForensics++.zip'
LOG='data/download_deepfakebench.log'
mkdir -p "$(dirname "$OUT")"

log() { echo "=== $(date '+%Y-%m-%d %H:%M:%S') $* ===" | tee -a "$LOG"; }

# Tổng dung lượng thật của file, lấy từ Content-Range của một yêu cầu 1 byte
total_size() {
    curl -sL -m 60 -r 0-0 -D - -o /dev/null "$URL" |
        tr -d '\r' | grep -i '^content-range:' | tail -1 | sed 's#.*/##'
}

while true; do
    TOTAL=$(total_size)
    if ! [[ "$TOTAL" =~ ^[0-9]+$ ]]; then
        # Không có Content-Range: Drive trả về trang HTML (vượt hạn mức hoặc lỗi)
        log "cannot get file size (Drive quota exceeded or unreachable), waiting 30 min"
        sleep 1800
        continue
    fi
    HAVE=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
    if [ "$HAVE" -eq "$TOTAL" ]; then
        log "DONE $OUT ($TOTAL bytes)"
        break
    fi
    if [ "$HAVE" -gt "$TOTAL" ]; then
        log "local file larger than remote ($HAVE > $TOTAL), restarting from scratch"
        rm -f "$OUT"
        HAVE=0
    fi
    log "downloading from byte $HAVE of $TOTAL"
    curl -L -C - --retry 5 --retry-delay 30 --speed-limit 10240 --speed-time 120 \
        -o "$OUT" "$URL" 2>>"$LOG"
    log "curl exited with code $?"
    sleep 60
done
