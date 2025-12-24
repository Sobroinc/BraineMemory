"""Vision adapter - ONLY gpt-5.2 for image understanding."""

import base64
import json
import logging
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = logging.getLogger(__name__)


class ExtractedText(BaseModel):
    """Extracted text from image/document."""

    content: str
    lang: str = "multi"
    blocks: list[dict[str, Any]] = []  # [{text, bbox, type}]


class ExtractedDrawing(BaseModel):
    """Extracted objects and dimensions from drawing."""

    objects: list[dict[str, Any]] = []  # [{type, label, bbox, description}]
    dimensions: list[dict[str, Any]] = []  # [{kind, text_raw, value, unit, p1, p2}]
    notes: list[str] = []


class ExtractedPhoto(BaseModel):
    """Extracted observations and measurements from photo."""

    observations: list[dict[str, Any]] = []  # [{type, severity, description, bbox}]
    measurements: list[dict[str, Any]] = []  # [{description, method, value, unit, p1, p2}]
    overall_description: str = ""


class VisionAdapter:
    """
    Vision adapter using ONLY gpt-5.2.

    RULES:
    - Model: gpt-5.2-2025-12-11 (or compatible snapshot)
    - Context: 400K tokens
    - Max output: 128K tokens
    - Pricing: $1.75 in / $14 out / 1M tokens
    """

    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for vision")

        # Validate model
        if not settings.vision_model.startswith("gpt-5.2"):
            raise ValueError(
                f"Invalid vision_model: {settings.vision_model}. "
                "Only 'gpt-5.2-*' snapshots are allowed."
            )

        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.vision_model
        logger.info(f"VisionAdapter initialized: {self._model}")

    @staticmethod
    def encode_image(image_path: str | Path) -> str:
        """Encode image to base64."""
        path = Path(image_path)
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    @staticmethod
    def get_mime_type(image_path: str | Path) -> str:
        """Get MIME type from file extension."""
        suffix = Path(image_path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".pdf": "application/pdf",
        }
        return mime_map.get(suffix, "image/png")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def _call_vision(
        self,
        image_url: str | None = None,
        image_base64: str | None = None,
        mime_type: str = "image/png",
        prompt: str = "",
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Call GPT-5.2 vision API."""
        # Build image content
        if image_url:
            image_content = {"type": "image_url", "image_url": {"url": image_url}}
        elif image_base64:
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
            }
        else:
            raise ValueError("Either image_url or image_base64 required")

        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 4096,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def extract_text(
        self,
        image_path: str | Path | None = None,
        image_url: str | None = None,
    ) -> ExtractedText:
        """Extract text from scanned document/image."""
        prompt = """
Extract all text from this document/image.
Return JSON:
{
    "content": "full extracted text",
    "lang": "ru|en|fr|multi",
    "blocks": [
        {
            "text": "block text",
            "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 50},
            "type": "paragraph|heading|table|list|caption"
        }
    ]
}
"""
        if image_path:
            base64_img = self.encode_image(image_path)
            mime = self.get_mime_type(image_path)
            result = await self._call_vision(
                image_base64=base64_img,
                mime_type=mime,
                prompt=prompt,
                response_format={"type": "json_object"},
            )
        else:
            result = await self._call_vision(
                image_url=image_url,
                prompt=prompt,
                response_format={"type": "json_object"},
            )

        data = json.loads(result)
        return ExtractedText(**data)

    async def analyze_drawing(
        self,
        image_path: str | Path | None = None,
        image_url: str | None = None,
    ) -> ExtractedDrawing:
        """Analyze technical drawing and extract objects/dimensions."""
        prompt = """
Analyze this technical drawing. Extract:
{
    "objects": [
        {
            "type": "door|window|wall|hinge|axis|note|symbol|section|detail",
            "label": "D1",
            "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
            "description": "Main entrance door"
        }
    ],
    "dimensions": [
        {
            "kind": "linear|diameter|radius|angle|arc_length",
            "text_raw": "Ø12±0.5",
            "value": 12.0,
            "unit": "mm",
            "tol_plus": 0.5,
            "tol_minus": 0.5,
            "p1": {"x": 50, "y": 100},
            "p2": {"x": 150, "y": 100}
        }
    ],
    "notes": ["Any text notes found on the drawing"]
}
Be precise with dimensions. Extract ALL visible measurements.
"""
        if image_path:
            base64_img = self.encode_image(image_path)
            mime = self.get_mime_type(image_path)
            result = await self._call_vision(
                image_base64=base64_img,
                mime_type=mime,
                prompt=prompt,
                response_format={"type": "json_object"},
            )
        else:
            result = await self._call_vision(
                image_url=image_url,
                prompt=prompt,
                response_format={"type": "json_object"},
            )

        data = json.loads(result)
        return ExtractedDrawing(**data)

    async def analyze_photo(
        self,
        image_path: str | Path | None = None,
        image_url: str | None = None,
    ) -> ExtractedPhoto:
        """Analyze photo for defects, objects, measurements."""
        prompt = """
Analyze this photo. Extract:
{
    "observations": [
        {
            "type": "crack|leak|stain|corrosion|damage|object|label|defect|other",
            "severity": "minor|moderate|severe|critical",
            "description": "Detailed description of what is observed",
            "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
        }
    ],
    "measurements": [
        {
            "description": "What is being measured",
            "method": "aruco|ruler|known_object|estimated",
            "value": 15.5,
            "unit": "cm",
            "p1": {"x": 50, "y": 100},
            "p2": {"x": 150, "y": 100}
        }
    ],
    "overall_description": "General description of what the photo shows"
}
Note: If you see ArUco markers or rulers, use them for measurements.
Otherwise, mark measurements as "estimated".
"""
        if image_path:
            base64_img = self.encode_image(image_path)
            mime = self.get_mime_type(image_path)
            result = await self._call_vision(
                image_base64=base64_img,
                mime_type=mime,
                prompt=prompt,
                response_format={"type": "json_object"},
            )
        else:
            result = await self._call_vision(
                image_url=image_url,
                prompt=prompt,
                response_format={"type": "json_object"},
            )

        data = json.loads(result)
        return ExtractedPhoto(**data)

    async def ask_about_image(
        self,
        question: str,
        image_path: str | Path | None = None,
        image_url: str | None = None,
    ) -> str:
        """Ask a question about an image."""
        if image_path:
            base64_img = self.encode_image(image_path)
            mime = self.get_mime_type(image_path)
            return await self._call_vision(
                image_base64=base64_img,
                mime_type=mime,
                prompt=question,
            )
        else:
            return await self._call_vision(
                image_url=image_url,
                prompt=question,
            )


# Global adapter instance
vision = VisionAdapter() if settings.openai_api_key else None
