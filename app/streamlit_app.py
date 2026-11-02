import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import streamlit as st

from src.ai_models import ModelUnavailable
from src.io_utils import to_rgb
from src.pipeline import PipelineConfig, restore_array
from src.preview import comparison_html

st.set_page_config(
    page_title="Hoi Sinh Ky Uc",
    page_icon=None,
    layout="wide",
)

st.title("Hồi Sinh Ký Ức")
st.caption(
    "Phục chế ảnh chân dung liệt sĩ / chiến sĩ bằng xử lý ảnh số. "
    "Kết quả AI cần được kiểm tra thủ công trước khi in ấn."
)

with st.sidebar:
    st.header("Xử lý cơ bản")
    denoise_method = st.selectbox(
        "Khử nhiễu",
        options=["median", "nlm", "none"],
        index=0,
    )
    denoise_strength = st.slider(
        "Cường độ khử nhiễu",
        min_value=1,
        max_value=15,
        value=3,
    )
    clahe_clip = st.slider(
        "Cân bằng tương phản (CLAHE)",
        min_value=0.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
    )
    gamma = st.slider(
        "Gamma",
        min_value=0.3,
        max_value=3.0,
        value=1.0,
        step=0.05,
    )
    upscale = st.select_slider(
        "Phóng to",
        options=[1, 2, 4],
        value=2,
    )
    sharpen = st.slider(
        "Làm nét",
        min_value=0.0,
        max_value=2.0,
        value=0.4,
        step=0.1,
    )

    st.divider()
    st.header("AI nâng cao")
    face_restore = st.checkbox(
        "Phục hồi khuôn mặt (GFPGAN)",
        value=True,
    )
    fidelity = st.slider(
        "Giữ nét gốc",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        disabled=not face_restore,
    )
    colorize = st.checkbox("Tô màu bằng AI", value=True)
    colorize_engine = st.selectbox(
        "Bộ tô màu",
        options=["caffe", "deoldify"],
        index=0,
        format_func=lambda value: {
            "caffe": "Caffe (nhẹ, khuyên dùng)",
            "deoldify": "DeOldify (nếu đã cài)",
        }[value],
        disabled=not colorize,
    )
    color_strength = st.slider(
        "Cường độ màu",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.05,
        disabled=not colorize,
    )
    color_saturation = st.slider(
        "Độ bão hòa",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.05,
        disabled=not colorize,
    )
    use_ai_upscale = st.checkbox(
        "Upscale bằng Real-ESRGAN",
        value=upscale > 1,
        disabled=upscale == 1,
    )
    st.caption(
        "Mặc định bật phục hồi + tô màu. Lần đầu chạy AI trên CPU có thể chậm."
    )

uploaded = st.file_uploader(
    "Tải lên ảnh chân dung",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
)

if uploaded is None:
    st.info("Chọn một ảnh để bắt đầu.")
    st.stop()

data = np.frombuffer(uploaded.read(), dtype=np.uint8)
image = cv2.imdecode(data, cv2.IMREAD_COLOR)
if image is None:
    st.error("Không đọc được ảnh. Hãy thử định dạng JPG hoặc PNG.")
    st.stop()

mask_file = st.file_uploader(
    "Mask vùng hỏng (tùy chọn, ảnh trắng đen cùng kích thước)",
    type=["png", "bmp", "tif", "tiff"],
)
inpaint_mask = None
if mask_file is not None:
    mask_data = np.frombuffer(mask_file.read(), dtype=np.uint8)
    inpaint_mask = cv2.imdecode(mask_data, cv2.IMREAD_GRAYSCALE)
    if inpaint_mask is None:
        st.error("Không đọc được mask.")
        st.stop()
    if inpaint_mask.shape[:2] != image.shape[:2]:
        st.error(
            "Mask phải cùng kích thước với ảnh gốc "
            f"({image.shape[1]}×{image.shape[0]})."
        )
        st.stop()

config = PipelineConfig(
    denoise=denoise_method,
    denoise_strength=denoise_strength,
    clahe=clahe_clip,
    gamma=gamma,
    upscale=upscale,
    sharpen=sharpen,
    face_restore=face_restore,
    fidelity=fidelity,
    colorize=colorize,
    colorize_engine=colorize_engine,
    color_strength=color_strength,
    color_saturation=color_saturation,
    use_ai_upscale=use_ai_upscale,
)

try:
    with st.spinner("Đang xử lý ảnh..."):
        result = restore_array(image, config, inpaint_mask=inpaint_mask)
except ModelUnavailable as exc:
    st.error(str(exc))
    st.info(
        "Cài AI bằng `pip install -r requirements-ai.txt`, "
        "tải model bằng `python scripts/download_models.py --all`, "
        "hoặc bỏ chọn các bước AI để dùng pipeline CPU."
    )
    st.stop()
except Exception as exc:
    st.error(f"Lỗi xử lý: {exc}")
    st.stop()

height, width = result.shape[:2]
st.success(f"Đã xử lý xong: {width} × {height} px")

comparison_result = result
if result.shape[:2] != image.shape[:2]:
    comparison_result = cv2.resize(
        result,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_AREA,
    )


tab_compare, tab_side_by_side = st.tabs(
    ["Thanh trượt trước / sau", "So sánh cạnh nhau"]
)

with tab_compare:
    html, component_height = comparison_html(image, comparison_result)
    st.iframe(html, height=component_height)

with tab_side_by_side:
    left, right = st.columns(2)
    with left:
        st.subheader("Ảnh gốc")
        st.image(to_rgb(image), width="stretch")
    with right:
        st.subheader("Sau xử lý")
        st.image(to_rgb(result), width="stretch")

success, encoded = cv2.imencode(".png", result)
if success:
    st.download_button(
        label="Tải kết quả PNG",
        data=encoded.tobytes(),
        file_name="hoi_sinh_ky_uc.png",
        mime="image/png",
    )

st.warning(
    "Ảnh phục chế cần được gia đình và người có trách nhiệm kiểm tra. "
    "Không thêm huy hiệu, quân hàm hoặc chi tiết không có trong ảnh gốc."
)
