# Tải dữ liệu FF++ và DeepFakeDetection, tự chờ khi server EU2 sập rồi tải tiếp.
# Chạy: powershell -ExecutionPolicy Bypass -File download_all.ps1
#       powershell -ExecutionPolicy Bypass -File download_all.ps1 -Compression c40
# c23: mask 4 phương pháp + DFD (video c23 FF++ đã tải trước đó).
# Mức nén khác (c40, raw): chỉ video của original, 4 phương pháp, DFD và DFD thật;
# mask không phụ thuộc mức nén nên dùng chung mask đã tải.
param([string]$Compression = 'c23')
$ErrorActionPreference = 'Continue'
$env:PYTHONUNBUFFERED = '1'   # ghi log ngay, không giữ trong bộ đệm
$server = 'https://kaldir.vc.in.tum.de/faceforensics/v3/misc/filelist.json'
$workers = 8
if ($Compression -eq 'c23') {
    $jobs = @(
        @('Face2Face', 'masks'), @('FaceSwap', 'masks'), @('NeuralTextures', 'masks'),
        @('DeepFakeDetection', 'masks'),
        @('DeepFakeDetection', 'videos'), @('DeepFakeDetection_original', 'videos')
    )
} else {
    $jobs = @('original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures',
              'DeepFakeDetection', 'DeepFakeDetection_original') |
        ForEach-Object { , @($_, 'videos') }
}

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
    $log = if ($Compression -ne 'c23') { "data\download_$Compression.log" }
           elseif ($type -eq 'masks') { 'data\download_masks.log' }
           else { 'data\download_dfd.log' }
    while ($true) {
        Wait-Server $log
        cmd /c "echo.| python faceforensics_download_v4.py data\FaceForensics -d $dataset -c $Compression -t $type --server EU2 --workers $workers >> $log 2>&1"
        $code = $LASTEXITCODE
        Write-Log $log "$Compression $type $dataset finished (exit $code)"
        if ($code -eq 0) { break }
        # Lỗi giữa chừng: xoá file tải dở rồi chạy lại (file đã xong sẽ được bỏ qua)
        Get-ChildItem -Recurse -File data\FaceForensics -Filter 'tmp*' | Remove-Item -Confirm:$false
        Start-Sleep 60
    }
}
Write-Log $log 'ALL DONE'
