# Tải mask FF++ và DeepFakeDetection, tự chờ khi server EU2 sập rồi tải tiếp.
# Chạy: powershell -ExecutionPolicy Bypass -File download_all.ps1
$ErrorActionPreference = 'Continue'
$env:PYTHONUNBUFFERED = '1'   # ghi log ngay, không giữ trong bộ đệm
$server = 'https://kaldir.vc.in.tum.de/faceforensics/v3/misc/filelist.json'
$workers = 8
$jobs = @(
    @('Face2Face', 'masks'), @('FaceSwap', 'masks'), @('NeuralTextures', 'masks'),
    @('DeepFakeDetection', 'masks'),
    @('DeepFakeDetection', 'videos'), @('DeepFakeDetection_original', 'videos')
)

function Write-Log($log, $msg) {
    "=== {0} {1} ===" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg |
        Out-File -Append -Encoding utf8 $log
}

function Wait-Server($log) {
    while ($true) {
        try {
            Invoke-WebRequest -Uri $server -Method Head -UseBasicParsing -TimeoutSec 30 | Out-Null
            return
        } catch {
            Write-Log $log 'server unreachable, waiting 5 min'
            Start-Sleep 300
        }
    }
}

foreach ($job in $jobs) {
    $dataset, $type = $job
    $log = if ($type -eq 'masks') { 'data\download_masks.log' } else { 'data\download_dfd.log' }
    while ($true) {
        Wait-Server $log
        cmd /c "echo.| python faceforensics_download_v4.py data\FaceForensics -d $dataset -c c23 -t $type --server EU2 --workers $workers >> $log 2>&1"
        $code = $LASTEXITCODE
        Write-Log $log "$type $dataset finished (exit $code)"
        if ($code -eq 0) { break }
        # Lỗi giữa chừng: xoá file tải dở rồi chạy lại (file đã xong sẽ được bỏ qua)
        Get-ChildItem -Recurse -File data\FaceForensics -Filter 'tmp*' | Remove-Item -Confirm:$false
        Start-Sleep 60
    }
}
Write-Log 'data\download_dfd.log' 'ALL DONE'
