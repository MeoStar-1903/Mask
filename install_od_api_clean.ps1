# =========================================================================
# Kịch bản cài đặt môi trường sạch cho TensorFlow Object Detection API
# Tương thích với: TensorFlow 2.11, Python 3.10, NumPy 1.23.5
# =========================================================================

param (
    [string]$EnvName = "mask_env",
    [string]$RepoPath = $PWD.Path
)

Write-Host "
=========================================================================
🔥 Bắt đầu quá trình cài đặt môi trường '$EnvName'
=========================================================================
" -ForegroundColor Green

# Bước 1: Dọn dẹp môi trường cũ (nếu có)
Write-Host "🧹 [Bước 1/8] Đang dọn dẹp môi trường '$EnvName' cũ (nếu tồn tại)..."
conda deactivate | Out-Null
conda env remove -n $EnvName -y | Out-Null
Write-Host "✅ Dọn dẹp xong."

# Bước 2: Tạo môi trường conda mới với Python 3.10
Write-Host "🐍 [Bước 2/8] Đang tạo môi trường '$EnvName' với Python 3.10..."
conda create -n $EnvName python=3.10 -y
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ LỖI: Không thể tạo môi trường conda. Dừng lại." -ForegroundColor Red
    return
}
Write-Host "✅ Tạo môi trường '$EnvName' thành công."

# Bước 3: Kích hoạt môi trường (chỉ cho script này)
Write-Host "🚀 [Bước 3/8] Đang kích hoạt môi trường '$EnvName'..."
conda activate $EnvName
Write-Host "✅ Môi trường đã được kích hoạt."

# Bước 4: Cài đặt các gói C-dependency BẰNG CONDA (gây xung đột)
Write-Host "🥥 [Bước 4/8] Đang cài 'pycocotools' và 'lvis' bằng conda..."
# Đây là bước quan trọng. Chúng ta để conda cài các gói này và các phụ thuộc
# (bao gồm cả numpy 2.x và scipy 2.x) trước.
conda install -c conda-forge pycocotools lvis -y
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ LỖI: Không thể cài 'pycocotools'/'lvis' bằng conda. Dừng lại." -ForegroundColor Red
    return
}
Write-Host "✅ Cài 'pycocotools' và 'lvis' xong."

# Bước 5: Cài đặt các gói chính bằng PIP (sẽ GHI ĐÈ các gói của conda)
Write-Host "📦 [Bước 5/8] Đang nâng cấp pip và cài đặt các gói PIP chính..."

# 5.1. Nâng cấp pip
python -m pip install --upgrade pip

# 5.2. Cài đặt các gói PIP, BAO GỒM VIỆC ÉP SCIPY CÀI LẠI
Write-Host "   ...Đang cài TensorFlow 2.11, Scipy (ép cài lại), và các thư viện phụ trợ..."
pip install scipy --force-reinstall
pip install tensorflow==2.11.0 "protobuf==3.19.6" pillow==9.5.0 lxml==6.0.2 matplotlib==3.7.1 cython contextlib2 opencv-python==4.7.0.72 tf-slim tensorflow-io tf-models-official==2.11.0 pandas psutil tensorflow-addons==0.19.0
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ LỖI: Cài đặt các gói PIP thất bại. Dừng lại." -ForegroundColor Red
    return
}

# 5.3. GHI ĐÈ Numpy về 1.23.5 (BẮT BUỘC cho TF 2.11 - CHẠY CUỐI CÙNG)
Write-Host "   ...Đang ép NumPy về phiên bản 1.23.5 (bước quan trọng nhất)..."
pip install numpy==1.23.5 --force-reinstall

# 5.4. Cài đặt PyYAML
Write-Host "   ...Đang cài PyYAML (bỏ qua cảnh báo tương thích)..."
pip install pyyaml==6.0.1 --force-reinstall

Write-Host "✅ Cài đặt các gói PIP thành công."

# Bước 6: Biên dịch các tệp Protobuf
Write-Host "📝 [Bước 6/8] Đang biên dịch các tệp Protobuf..."
# Đảm bảo chúng ta đang ở trong thư mục 'research'
$ResearchPath = Join-Path -Path $RepoPath -ChildPath "models\research"
if (-not (Test-Path $ResearchPath)) {
    Write-Host "❌ LỖI: Không tìm thấy thư mục '$ResearchPath'. Bạn đang chạy script ở thư mục gốc chứa 'models' chứ?" -ForegroundColor Red
    return
}
cd $ResearchPath

# Chạy lệnh protoc
protoc object_detection/protos/*.proto --python_out=.
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ LỖI: Không thể biên dịch Protobuf. Bạn đã cài 'protoc' và thêm vào PATH chưa?" -ForegroundColor Red
} else {
    Write-Host "✅ Biên dịch Protobuf thành công."
}
cd $RepoPath # Quay lại thư mục gốc

# Bước 7: Thiết lập PYTHONPATH vĩnh viễn
Write-Host "🛤️ [Bước 7/8] Đang thiết lập PYTHONPATH vĩnh viễn..."
$SlimPath = Join-Path -Path $ResearchPath -ChildPath "slim"
$EnvPath = "$ResearchPath;$SlimPath"

# Sử dụng 'setx' để đặt biến môi trường vĩnh viễn cho User
setx PYTHONPATH $EnvPath
Write-Host "✅ PYTHONPATH đã được đặt thành: $EnvPath"
Write-Host "   (Lưu ý: Bạn có thể cần khởi động lại PowerShell/PC để thay đổi này có hiệu lực)"

# Bước 8: Hoàn tất
Write-Host "
=========================================================================
🎉 CÀI ĐẶT HOÀN TẤT! 🎉
=========================================================================
" -ForegroundColor Green

Write-Host "Các bước tiếp theo:"
Write-Host "1. 🛑 QUAN TRỌNG: Bạn CẦN ĐÓNG và MỞ LẠI cửa sổ PowerShell này."
Write-Host "2. Kích hoạt lại môi trường: conda activate $EnvName"
Write-Host "3. Chạy kiểm tra cài đặt:"
Write-Host "   python models/research/object_detection/builders/model_builder_tf2_test.py"
Write-Host "
Chúc may mắn!
"

