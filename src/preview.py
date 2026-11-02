import base64

import cv2


def encode_preview(image_bgr, max_width=1100):
    height, width = image_bgr.shape[:2]
    preview = image_bgr
    if width > max_width:
        scale = max_width / width
        preview = cv2.resize(
            image_bgr,
            (max_width, max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    success, encoded = cv2.imencode(".png", preview)
    if not success:
        raise RuntimeError("Khong ma hoa duoc anh xem truoc")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def comparison_html(before_bgr, after_bgr, max_width=900):
    before = encode_preview(before_bgr)
    after = encode_preview(after_bgr)
    height, width = before_bgr.shape[:2]
    component_height = min(
        760,
        max(320, int(max_width * height / width) + 42),
    )
    html = """
<style>
  .compare-wrap {
    position: relative;
    width: 100%;
    max-width: __MAX_WIDTH__px;
    margin: 0 auto;
    border-radius: 10px;
    overflow: hidden;
    background: #111;
  }
  .compare-wrap img {
    display: block;
    width: 100%;
    height: auto;
  }
  .compare-top {
    position: absolute;
    inset: 0;
    clip-path: inset(0 50% 0 0);
  }
  .compare-line {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    width: 2px;
    background: #fff;
    box-shadow: 0 0 8px rgba(0,0,0,.8);
    pointer-events: none;
  }
  .compare-label {
    position: absolute;
    top: 10px;
    padding: 4px 9px;
    border-radius: 999px;
    color: #fff;
    background: rgba(0,0,0,.55);
    font: 600 12px/1.2 system-ui, sans-serif;
  }
  .label-before { left: 10px; }
  .label-after { right: 10px; }
  .compare-slider {
    width: calc(100% - 24px);
    margin: 8px 12px 12px;
    accent-color: #2e7d32;
  }
</style>
<div class="compare-wrap">
  <img src="data:image/png;base64,__BEFORE__" alt="Anh goc">
  <img class="compare-top" id="compare-top"
       src="data:image/png;base64,__AFTER__" alt="Anh phuc che">
  <div class="compare-line" id="compare-line"></div>
  <span class="compare-label label-before">Gốc</span>
  <span class="compare-label label-after">Phục chế</span>
</div>
<input class="compare-slider" id="compare-range" type="range"
       min="0" max="100" value="50">
<script>
  const range = document.getElementById('compare-range');
  const top = document.getElementById('compare-top');
  const line = document.getElementById('compare-line');
  function updateCompare() {
    const value = Number(range.value);
    top.style.clipPath = `inset(0 ${100 - value}% 0 0)`;
    line.style.left = `${value}%`;
  }
  range.addEventListener('input', updateCompare);
  updateCompare();
</script>
"""
    html = (
        html.replace("__MAX_WIDTH__", str(max_width))
        .replace("__BEFORE__", before)
        .replace("__AFTER__", after)
    )
    return html, component_height
