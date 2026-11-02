# Dự án Hồi Sinh Ký Ức

Phục chế, tô màu và tái tạo chân dung liệt sĩ / chiến sĩ đã mất.

> Dự án kết hợp Xử lý ảnh số, Thị giác máy tính, Deep Learning và giá trị nhân văn:
> giúp các gia đình có một bức di ảnh trang nghiêm, rõ nét và chân thực nhất.

<p align="center">
  <img src="docs/images/before-after.jpg" alt="So sánh ảnh gốc và ảnh sau phục chế" width="100%">
</p>

<p align="center">
  <em>Ảnh thử nghiệm do người dùng cung cấp: phục hồi khuôn mặt, tô màu và upscale 2x.</em>
</p>

## Mục lục

- [Trạng thái hiện tại](#trạng-thái-hiện-tại)
- [Kết quả thực tế](#kết-quả-thực-tế)
- [Mục tiêu](#mục-tiêu)
- [Nguyên tắc quan trọng](#nguyên-tắc-quan-trọng)
- [Tech stack](#tech-stack)
- [Pipeline 5 bước](#pipeline-5-bước)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Chạy nhanh trên Windows](#chạy-nhanh-trên-windows)
- [Cài đặt cơ bản](#cài-đặt-cơ-bản)
- [Chạy CLI](#chạy-cli)
- [Chạy giao diện Streamlit](#chạy-giao-diện-streamlit)
- [Cài đặt AI](#cài-đặt-ai-tùy-chọn)
- [Tham số khuyến nghị](#tham-số-khuyến-nghị)
- [Xử lý sự cố](#xử-lý-sự-cố)
- [Kiểm thử](#kiểm-thử)
- [Bản quyền ảnh và model](#bản-quyền-ảnh-và-model)
- [Hướng phát triển](#hướng-phát-triển)
- [Định hướng xã hội](#định-hướng-xã-hội)
- [Tác giả](#tác-giả)

## Trạng thái hiện tại

Pipeline đã chạy end-to-end trên CPU với đầy đủ các bước chính:

- Đọc/ghi ảnh hỗ trợ đường dẫn Unicode trên Windows.
- Khử nhiễu trung vị / Non-Local Means.
- Cân bằng tương phản CLAHE, chỉnh gamma, làm nét unsharp mask.
- Vá vùng hỏng bằng mask có sẵn (OpenCV Telea / Navier-Stokes).
- Upscale bằng Lanczos (fallback CPU) hoặc Real-ESRGAN.
- Adapter cho GFPGAN, Real-ESRGAN, Caffe Colorization và DeOldify.
- Tô màu ảnh đen trắng bằng Caffe (nhẹ, mặc định) hoặc DeOldify (tùy chọn).
- CLI xử lý một ảnh và ứng dụng Streamlit để so sánh trước/sau.
- GFPGAN, Real-ESRGAN và Caffe Colorization đã chạy thử thành công trên CPU.
- Bộ test hiện có **12/12 test pass**.
- Model AI được tải cục bộ vào `models/` (không commit vì dung lượng lớn).
  Xem phần [Cài đặt AI](#cài-đặt-ai-tùy-chọn).

## Kết quả thực tế

<p align="center">
  <img src="docs/images/restored.jpg" alt="Ảnh sau phục chế" width="420">
</p>

<p align="center">
  <em>Kết quả sau GFPGAN → Caffe Colorization → Real-ESRGAN 2x.</em>
</p>

| Hạng mục | Ảnh gốc | Sau phục chế |
|---|---:|---:|
| Kích thước | 388 × 515 | 776 × 1030 |
| Khuôn mặt phát hiện | 1 | 1 |
| Colorfulness | 5.61 | 22.93 |
| Laplacian sharpness | 63.54 | 1388.03 |
| Định dạng | JPG | PNG |

> Các chỉ số trên chỉ dùng để kiểm tra kỹ thuật, không thay thế đánh giá bằng
> mắt. Màu da, quân phục và phù hiệu vẫn cần con người kiểm tra theo ảnh gốc và
> tư liệu lịch sử.

Lệnh đã dùng để tạo kết quả:

```bash
python scripts/run_pipeline.py \
  --input data/input/images.jpg \
  --output data/output/images_restored.png \
  --face-restore \
  --fidelity 0.7 \
  --colorize \
  --colorize-engine caffe \
  --color-strength 0.85 \
  --color-saturation 1.0 \
  --use-ai-upscale \
  --upscale 2
```

## Mục tiêu

- Số hóa, làm sạch nhiễu, phục hồi chi tiết bị mờ/rách/ố vàng trên ảnh cũ.
- Tái tạo khuôn mặt bằng AI nhưng vẫn giữ nét mặt gốc.
- Tô màu có kiểm soát cho ảnh trắng đen.
- Xuất file chất lượng cao để in di ảnh thờ hoặc trưng bày.
- Bảo đảm tính lịch sử của quân phục, phù hiệu, cấp hiệu.

## Nguyên tắc quan trọng

Đây là dự án nhạy cảm về ký ức và danh dự của người đã khuất.

- Luôn có sự đồng ý của gia đình trước khi xử lý hoặc công bố ảnh.
- Không để AI tự ý thay đổi cấu trúc khuôn mặt, giới tính, tuổi tác.
- Không tự ý thêm huy hiệu, quân hàm, phù hiệu hoặc chi tiết không có trong ảnh gốc.
- Màu quân phục phải được kiểm chứng theo từng thời kỳ lịch sử.
- Kết quả AI chỉ là bản nháp; con người phải kiểm tra lần cuối.
- Lưu trữ ảnh gốc riêng tư, không đưa lên mạng khi chưa được phép.

## Tech stack

| Thành phần | Công nghệ |
|------------|-----------|
| Ngôn ngữ | Python 3.10/3.11 cho AI; 3.13 chỉ nên dùng cho pipeline CPU |
| Xử lý ảnh cơ bản | OpenCV, NumPy, SciPy, Pillow |
| Deep Learning | PyTorch |
| Phục hồi khuôn mặt | GFPGAN / CodeFormer |
| Super-resolution | Real-ESRGAN |
| Tô màu | Caffe Colorization (mặc định) / DeOldify |
| Inpainting | OpenCV Inpainting, LaMa / LaBRIM mở rộng |
| Giao diện | Streamlit |

## Pipeline 5 bước

1. **Tiền xử lý**: khử nhiễu muối tiêu/Gauss, cân bằng histogram, gamma.
2. **Vá lỗi cấu trúc**: dùng mask vùng rách/xước và inpainting.
3. **Phục hồi khuôn mặt**: GFPGAN/CodeFormer với mức fidelity phù hợp.
4. **Tô màu**: Caffe/DeOldify tạo màu nền, sau đó chỉnh tay quân phục/phù hiệu.
5. **Super-resolution & hậu kỳ**: Real-ESRGAN, làm nét, cân bằng màu, xuất PNG/TIFF.

## Cấu trúc dự án

```
DuAn_HoiSinhKyUc/
├── README.md
├── requirements.txt              # thư viện CPU/UI cơ bản
├── requirements-ai.txt           # PyTorch + model AI (tùy chọn)
├── run.bat                       # tạo venv, cài thư viện, mở Streamlit
├── setup_ai.bat                  # tạo venv Python 3.11 + cài AI
├── .gitignore
├── app/
│   └── streamlit_app.py          # upload / slider trước-sau / tải về
├── config/
│   └── settings.py               # đường dẫn model, tham số mặc định
├── data/
│   ├── input/                    # ảnh gốc (không commit)
│   └── output/                   # ảnh kết quả (không commit)
├── docs/
│   └── images/                   # ảnh minh họa cho README
├── models/                       # chứa file .pth (không commit)
│   └── colorization/             # model Caffe tô màu (không commit)
├── scripts/
│   ├── run_pipeline.py           # CLI xử lý một ảnh
│   ├── download_models.py        # tải GFPGAN / Real-ESRGAN / colorization
│   └── patch_basicsr.py          # vá tương thích torchvision mới
├── src/
│   ├── __init__.py
│   ├── io_utils.py               # đọc/ghi ảnh Unicode-safe
│   ├── preprocessing.py          # khử nhiễu, CLAHE, gamma, sharpen
│   ├── inpainting.py             # vá vùng hỏng bằng mask
│   ├── ai_models.py              # adapter GFPGAN, Real-ESRGAN, Caffe, DeOldify
│   ├── preview.py                # tạo HTML slider so sánh trước/sau
│   └── pipeline.py               # ghép các bước theo cấu hình
└── tests/
    ├── test_preprocessing.py
    ├── test_pipeline.py
    ├── test_preview.py
    └── test_ai_models.py
```

## Chạy nhanh trên Windows

```bat
setup_ai.bat
.venv\Scripts\python.exe scripts\download_models.py --all
run.bat
```

Sau đó mở trình duyệt tại địa chỉ Streamlit in ra, tải ảnh lên và giữ mặc định:

- **Phục hồi khuôn mặt (GFPGAN)**: bật.
- **Tô màu bằng AI**: bật, engine `caffe`.
- **Upscale bằng Real-ESRGAN**: bật, `2x`.

Nếu chỉ muốn chạy pipeline CPU nhẹ, bỏ chọn cả ba mục AI.

## Cài đặt cơ bản

Windows PowerShell:

```powershell
cd D:\Baitapopencv\DuAn_HoiSinhKyUc
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/WSL:

```bash
cd /mnt/d/Baitapopencv/DuAn_HoiSinhKyUc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy CLI

```bash
python scripts/run_pipeline.py \
  --input data/input/chan_dung.jpg \
  --output data/output/chan_dung_phuc_che.png \
  --denoise median \
  --upscale 2 \
  --sharpen 0.6
```

Các tham số chính:

| Tham số | Ý nghĩa |
|---------|---------|
| `--denoise none\|median\|nlm` | Chọn bộ lọc khử nhiễu |
| `--denoise-strength` | Kích thước median hoặc cường độ NLM |
| `--clahe` | Cường độ cân bằng tương phản, `0` để tắt |
| `--gamma` | `1.0` giữ nguyên; `<1` sáng hơn, `>1` tối hơn |
| `--inpaint-mask` | Ảnh mask trắng đen, vùng trắng là vùng cần vá |
| `--upscale 1\|2\|4` | Hệ số phóng to |
| `--sharpen` | Cường độ làm nét, `0` để tắt |
| `--face-restore` | Bật GFPGAN (cần cài AI) |
| `--fidelity` | Mức giữ nét gốc của GFPGAN, `0..1` |
| `--colorize` | Bật tô màu AI |
| `--colorize-engine caffe\|deoldify` | Chọn bộ tô màu |
| `--color-strength` | Cường độ màu, `0..1` (mặc định `0.85`) |
| `--color-saturation` | Độ bão hòa, `0..2` (mặc định `1.0`) |
| `--use-ai-upscale` | Dùng Real-ESRGAN thay Lanczos |

Ví dụ có mask vá ảnh:

```bash
python scripts/run_pipeline.py \
  --input data/input/chan_dung.jpg \
  --inpaint-mask data/input/chan_dung_mask.png \
  --output data/output/chan_dung_phuc_che.png \
  --denoise nlm \
  --denoise-strength 7
```

Mask phải cùng kích thước với ảnh gốc. Tạo mask bằng GIMP/Photoshop hoặc bằng
ngưỡng sáng nếu vết hỏng có màu sắc rõ ràng.

## Tham số khuyến nghị

Đây là điểm khởi đầu, không phải công thức cố định. Luôn giữ ảnh gốc để so sánh.

| Trường hợp | Gợi ý |
|---|---|
| Chân dung cũ, mờ nhẹ | `--denoise median --clahe 2 --gamma 1 --fidelity 0.7 --sharpen 0.4` |
| Ảnh nhiễu hạt, xước nhẹ | `--denoise nlm --denoise-strength 7 --clahe 2 --fidelity 0.6` |
| Ảnh đã khá nét | `--denoise none --clahe 1 --fidelity 0.8 --sharpen 0.2` |
| Ảnh quá tối | `--gamma 0.8 --clahe 2` |
| Màu bị nhân tạo | Giảm `--color-strength 0.6`, `--color-saturation 0.9` |
| Muốn giữ nét gốc tối đa | Tăng `--fidelity 0.85`, giảm `--sharpen 0.2` |
| Chỉ cần phóng to, không AI | `--upscale 2` và bỏ `--use-ai-upscale` |

Với ảnh in di ảnh, nên xuất PNG rồi kiểm tra ở kích thước in thật. Không nên
upscale quá 2x nếu ảnh gốc quá nhỏ, vì AI có thể tạo chi tiết không có thật.

## Chạy giao diện Streamlit

Cách nhanh trên Windows: nhấp đúp `run.bat`, hoặc chạy:

```bat
run.bat
```

Script tự tạo `.venv`, cài `requirements.txt` nếu cần và mở Streamlit.

Hoặc chạy thủ công:

```bash
streamlit run app/streamlit_app.py
```

Giao diện cho phép:

- Tải ảnh chân dung và mask vùng hỏng (tùy chọn).
- Chỉnh khử nhiễu, CLAHE, gamma, độ phóng to, làm nét.
- Bật/tắt GFPGAN, tô màu AI (Caffe/DeOldify), Real-ESRGAN.
- Mặc định bật phục hồi khuôn mặt, tô màu và upscale 2x bằng Real-ESRGAN.
- Kéo thanh trượt để so sánh trực tiếp ảnh gốc / ảnh phục chế.
- Xem cạnh nhau và tải kết quả PNG.

Nếu chưa cài AI, bỏ chọn các mục trong nhóm **AI nâng cao** để dùng pipeline CPU.
Khi bật AI nhưng thiếu model/thư viện, giao diện sẽ báo lỗi và hướng dẫn cài đặt.

## Cài đặt AI (tùy chọn)

Trên Windows, cách nhanh nhất là chạy:

```bat
setup_ai.bat
```

Script sẽ tạo lại `.venv` bằng **Python 3.11**, cài thư viện AI, vá lỗi tương
thích `basicsr`/`torchvision` và kiểm tra import. Script dùng thư mục tạm/cache
trong chính dự án (`.tmp/`, `.pip-cache/`, `wheels/`) để tránh lỗi file bị khóa
khi `%TEMP%` nằm ở ổ khác.

> Không dùng Python 3.13 để cài `basicsr==1.4.2`: bản này lỗi
> `KeyError: '__version__'` khi build. Dùng Python 3.10 hoặc 3.11.

Cấu hình đã chạy thử thành công:

- Python 3.11.9, venv tại `.venv/`
- `torch 2.15.0+cpu`, `torchvision 0.29.2+cpu`, `opencv-python 4.14.0.94`
- `gfpgan 1.3.8`, `basicsr 1.4.2`, `facexlib 0.3.0`, `realesrgan 0.3.0`
- GFPGAN, Real-ESRGAN và Caffe Colorization đã chạy thử thành công.

Cài thủ công bằng Python 3.11:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-ai.txt
python scripts\patch_basicsr.py
```

Tải model vào thư mục `models/`:

```bash
python scripts/download_models.py --all
```

Hoặc tải riêng:

```bash
python scripts/download_models.py --gfpgan
python scripts/download_models.py --realesrgan
python scripts/download_models.py --colorization
```

Các file chính:

- GFPGAN v1.4: `models/GFPGANv1.4.pth` (~350 MB)
- Real-ESRGAN x4plus: `models/RealESRGAN_x4plus.pth` (~64 MB)
- Caffe Colorization: `models/colorization/` (~123 MB)
  - `colorization_deploy_v2.prototxt`
  - `colorization_release_v2.caffemodel`
  - `pts_in_hull.npy`

Nguồn model: GFPGAN (TencentARC), Real-ESRGAN (xinntao),
Caffe Colorization của Richard Zhang và cộng sự (BSD-2-Clause; bản mirror
`spmallick/learnopencv`).

Lần chạy GFPGAN đầu tiên, `facexlib` tự tải thêm model phát hiện/parsing khuôn
mặt (~186 MB) vào `gfpgan/weights/`. Cần kết nối mạng cho lần chạy đó; thư mục
này đã được gitignore.

Ví dụ CLI đầy đủ (phục hồi + tô màu + upscale):

```bash
python scripts/run_pipeline.py \
  --input data/input/chan_dung.jpg \
  --output data/output/chan_dung_ai.png \
  --face-restore \
  --fidelity 0.7 \
  --colorize \
  --colorize-engine caffe \
  --color-strength 0.85 \
  --color-saturation 1.0 \
  --use-ai-upscale \
  --upscale 2
```

Caffe là bộ tô màu nhẹ, không cần thêm thư viện ngoài. DeOldify cho màu đậm hơn
nhưng cần cài riêng và dễ gặp lỗi phiên bản; chỉ dùng khi đã cài thành công.

Lưu ý:

- Bản đã cài dùng PyTorch CPU. Máy có GPU NVIDIA (ví dụ RTX 3050) có thể cài
  thêm bản CUDA của PyTorch để chạy nhanh hơn.
- Chạy CPU rất chậm; nên dùng GPU CUDA nếu có.
- GFPGAN có thể làm khuôn mặt "trẻ hơn" hoặc thay đổi nét gốc nếu fidelity thấp.
- Real-ESRGAN có thể tạo chi tiết không có trong ảnh; cần kiểm tra thủ công.
- DeOldify cần cài riêng theo hướng dẫn của repo gốc và tải model colorize.
  Adapter trong `src/ai_models.py` chỉ hoạt động khi môi trường đã có DeOldify.
- Màu do AI tạo chỉ là gợi ý; cần chỉnh tay màu da, quân phục, phù hiệu theo
  ảnh gốc và tư liệu lịch sử.
- Với PyTorch mới, dự án tự đặt `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` để nạp
  checkpoint GFPGAN/Real-ESRGAN cũ. Chỉ dùng model tải từ nguồn tin cậy.
- Nếu `basicsr` báo thiếu `torchvision.transforms.functional_tensor`, chạy:
  `python scripts\patch_basicsr.py`.

## Xử lý sự cố

| Lỗi / hiện tượng | Cách xử lý |
|---|---|
| `AttributeError: module 'cv2.dnn' has no attribute 'readNetFromCaffe'` | OpenCV 5 đã bỏ Caffe. Cài OpenCV 4.x: `pip install "opencv-python>=4.9,<5"` |
| `KeyError: '__version__'` khi cài `basicsr` | Dùng Python 3.10/3.11, không dùng 3.13 |
| Thiếu `torchvision.transforms.functional_tensor` | Chạy `python scripts\patch_basicsr.py` |
| `Failed to remove contents in ... ~v2` | Cảnh báo vô hại; có thể xóa thủ công thư mục `~v2` trong `site-packages` |
| Pip báo file bị khóa / `Access denied` | Đóng tiến trình đang dùng file, chạy `setup_ai.bat` để dùng cache trong dự án |
| GFPGAN không tìm thấy khuôn mặt | Dùng ảnh chân dung rõ mặt, hoặc tắt `--face-restore` |
| Màu da / quân phục sai | Giảm cường độ màu, bật `deoldify` nếu đã cài, hoặc chỉnh tay sau khi xuất |
| Chạy CPU quá chậm | Cài PyTorch bản CUDA tương ứng; kiểm tra bằng `python -c "import torch; print(torch.cuda.is_available())"` |
| Streamlit báo cổng đang dùng | Chạy `streamlit run app/streamlit_app.py --server.port 8502` |
| Ảnh kết quả quá nét | Giảm `--sharpen` xuống `0.2` hoặc tắt upscale AI |

## Kiểm thử

```bash
python -m unittest discover -s tests -v
```

Kết quả hiện tại: **12/12 test pass**. Bộ test kiểm tra tiền xử lý, inpainting,
pipeline CPU, preview HTML và việc thiếu model AI. Nếu chưa cài OpenCV, một số
test sẽ được bỏ qua thay vì báo lỗi.

Kiểm tra nhanh môi trường AI:

```bash
python -c "import torch, cv2, gfpgan, realesrgan; print(torch.__version__, cv2.__version__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -m pip check
```

## Bản quyền ảnh và model

- **Ảnh minh họa trong README** do người dùng cung cấp cho mục đích thử nghiệm.
  Không tái sử dụng ngoài dự án khi chưa được phép.
- **Model GFPGAN, Real-ESRGAN, Caffe Colorization** thuộc bản quyền của các tác
  giả gốc và tuân theo giấy phép của từng dự án. Xem phần
  [Cài đặt AI](#cài-đặt-ai-tùy-chọn) để biết nguồn tải.
- **Mã nguồn dự án** hiện chưa kèm file `LICENSE`. Nếu muốn cho phép tái sử
  dụng, nên bổ sung giấy phép phù hợp trước khi công bố rộng rãi.
- Không đưa ảnh của người đã khuất lên mạng nếu gia đình chưa đồng ý.

## Hướng phát triển

- [x] Giao diện so sánh trước/sau bằng thanh trượt.
- [x] Tích hợp GFPGAN / Real-ESRGAN qua adapter và tải model tự động.
- [ ] Thêm phát hiện vùng hỏng tự động (scratch/crack detection).
- [ ] Tích hợp CodeFormer như lựa chọn thay GFPGAN.
- [ ] Thêm bộ lọc màu quân phục theo giai đoạn lịch sử.
- [ ] Lưu metadata và lịch sử chỉnh sửa cho từng ảnh.
- [ ] Xử lý hàng loạt (batch) có hàng đợi.
- [ ] Xuất TIFF 300-600 DPI kèm profile màu.
- [ ] Quy trình kiểm duyệt hai bước: AI xử lý → người kiểm tra.

## Định hướng xã hội

Dự án có thể phát triển thành chiến dịch tình nguyện công nghệ, phối hợp với
Đoàn thanh niên và cơ quan thương binh xã hội để tiếp nhận ảnh từ các gia đình
liệt sĩ, Bà mẹ Việt Nam Anh hùng trên toàn quốc.

## Tác giả

- **Phan Đình Mạnh** — [@MANH-IT](https://github.com/MANH-IT)
- Repository: <https://github.com/MANH-IT/HoiSinhKyUc>

Đóng góp, báo lỗi hoặc đề xuất cải thiện xin gửi qua GitHub Issues.

## Lời cảm ơn

Dự án sử dụng và kế thừa ý tưởng từ các dự án mã nguồn mở:

- [GFPGAN](https://github.com/TencentARC/GFPGAN)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [Colorization](https://github.com/richzhang/colorization) của Richard Zhang
- [facexlib](https://github.com/xinntao/facexlib)
- [Streamlit](https://streamlit.io/) và [OpenCV](https://opencv.org/)
