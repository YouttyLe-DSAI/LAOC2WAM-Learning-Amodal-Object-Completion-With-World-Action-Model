"""
Các agent suy luận, mô phỏng theo kiến trúc của Multi-Agent Amodal Completion
(Fan et al., ACM MM'25). Mỗi agent là một lời gọi có cấu trúc tới VLM
(mặc định GPT-4o qua OpenAI API), KHÔNG cần huấn luyện — chỉ dùng để
tự động sinh pseudo-label (mask + mô tả) cho dữ liệu ít của bạn.

Segmentation thật (Grounded-SAM / SAM) không bundle sẵn ở đây vì cần cài
đặt riêng khá nặng — xem hàm `segment_visible_and_occluders` để biết chỗ
cắm vào (TODO), tạm thời dùng placeholder đơn giản dựa trên box do agent trả về.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass

from openai import OpenAI


def _b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Thiếu OPENAI_API_KEY. Set biến môi trường hoặc Kaggle Secret trước khi chạy."
        )
    return OpenAI(api_key=api_key)


@dataclass
class OcclusionResult:
    target_object: str
    occluders: list[str]
    is_occluded: bool


def identify_occlusion(image_path: str, query: str, model: str = "gpt-4o") -> OcclusionResult:
    """Occlusion Identification Agent: xác định vật mục tiêu và các occluder."""
    client = _client()
    img_b64 = _b64_image(image_path)
    prompt = (
        f"Bạn đang phân tích một bức ảnh để tìm vật thể: '{query}'. "
        "Xác định xem vật thể này có đang bị vật khác che khuất một phần không, "
        "và liệt kê tên các vật che khuất (nếu có). "
        'Trả lời DUY NHẤT bằng JSON theo schema: '
        '{"target_object": str, "is_occluded": bool, "occluders": [str, ...]}'
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    return OcclusionResult(
        target_object=data.get("target_object", query),
        occluders=data.get("occluders", []),
        is_occluded=data.get("is_occluded", False),
    )


def analyze_boundary(image_path: str, edge_touch_prior: list[str], model: str = "gpt-4o") -> dict:
    """Boundary Analysis Agent: ước lượng object có bị cắt ở biên ảnh không,
    và tỉ lệ cần mở rộng khung ảnh mỗi cạnh."""
    client = _client()
    img_b64 = _b64_image(image_path)
    prompt = (
        "Phân tích xem vật thể chính trong ảnh có bị cắt bởi biên ảnh không. "
        f"Gợi ý hình học: bbox của phần nhìn thấy chạm các cạnh sau: {edge_touch_prior or 'không cạnh nào'}. "
        "Nếu bị cắt, ước lượng tỉ lệ mở rộng cần thiết mỗi cạnh (0-1, theo phần trăm kích thước ảnh). "
        'Trả lời DUY NHẤT bằng JSON: '
        '{"is_truncated": bool, "extend": {"left": float, "right": float, "top": float, "bottom": float}}'
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def generate_description(image_path: str, target_object: str, model: str = "gpt-4o") -> str:
    """Description Agent: sinh P_text — mô tả chi tiết object bao gồm cả
    phần suy luận cho vùng bị che (dùng làm caption cho fine-tuning)."""
    client = _client()
    img_b64 = _b64_image(image_path)
    prompt = (
        f"Mô tả chi tiết vật thể '{target_object}' trong ảnh: màu sắc, kết cấu, "
        "đặc điểm riêng, tư thế. Nếu vật thể bị che một phần, hãy suy luận và "
        "mô tả luôn cả đặc điểm hợp lý của phần bị che, như thể vật đó hoàn toàn "
        "không bị che. Viết thành MỘT câu mô tả liền mạch, súc tích, tiếng Anh "
        "(để tương thích với text encoder của Stable Diffusion)."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }
        ],
    )
    return resp.choices[0].message.content.strip()


def segment_visible_and_occluders(image_path: str, target_object: str, occluders: list[str]):
    """
    Placeholder cho Segmentation Agent (Grounded-SAM / LISA trong paper gốc).
    TODO: cắm Grounded-SAM thật ở đây:
        https://github.com/IDEA-Research/Grounded-Segment-Anything
    Hàm này hiện raise NotImplementedError để bạn tự chọn engine segmentation
    phù hợp với GPU bạn có (Grounded-SAM khá nặng, cân nhắc bản 'lite' hoặc
    dùng SAM2 + text-to-box qua OWL-ViT nếu chạy trên Kaggle GPU đơn).
    """
    raise NotImplementedError(
        "Cắm Grounded-SAM hoặc SAM2+OWL-ViT vào đây để sinh M_visible và M_occ thật. "
        "Xem TODO trong docstring."
    )
