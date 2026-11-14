# Tải bản gốc FF++ và DeepFakeDetection (video + mask) từ server của TUM về data\FaceForensics,
# kèm split chính thức (data\splits\{train,val,test}.json). Tự chờ khi server sập rồi tải tiếp.
#
# Cần trước: điền form xin quyền FF++ (https://github.com/ondyari/FaceForensics) và đồng ý
# điều khoản sử dụng; script tải faceforensics_download_v4.py là bản nhóm tác giả gửi.
#
# Chạy: powershell -ExecutionPolicy Bypass -File download_all.ps1                   (c23)
#       powershell -ExecutionPolicy Bypass -File download_all.ps1 -Compression c40  (thêm c40)
# c23: video original + 4 phương pháp + DFD (thật và giả), mask 4 phương pháp + DFD.
# c40: chỉ video; mask không phụ thuộc mức nén nên dùng chung mask đã tải ở bước c23.
# Chạy lại an toàn: file đã tải xong được bỏ qua.
param([string]$Compression = 'c23')
$ErrorActionPreference = 'Continue'
$env:PYTHONUNBUFFERED = '1'   # ghi log ngay, không giữ trong bộ đệm
$server = 'https://kaldir.vc.in.tum.de/faceforensics/v3/misc/filelist.json'
$workers = 8
New-Item -ItemType Directory -Force data\splits | Out-Null
foreach ($s in 'train', 'val', 'test') {
    if (-not (Test-Path "data\splits\$s.json")) {
        Invoke-WebRequest -UseBasicParsing -OutFile "data\splits\$s.json" `
            "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/$s.json"
    }
}
$videos = @('original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures',
            'DeepFakeDetection', 'DeepFakeDetection_original') | ForEach-Object { , @($_, 'videos') }
if ($Compression -eq 'c23') {
    $jobs = $videos + @(
        @('Deepfakes', 'masks'), @('Face2Face', 'masks'), @('FaceSwap', 'masks'),
        @('NeuralTextures', 'masks'), @('DeepFakeDetection', 'masks')
    )
} else {
    $jobs = $videos
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
           else { 'data\download.log' }
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
