"""
Utility: Image and text preprocessing helpers.
"""

from pathlib import Path
from PIL import Image


def load_and_validate_image(image_path: Path) -> Image.Image:
    """
    Open and validate a PIL image.

    Converts to RGB, strips EXIF metadata.

    Args:
        image_path: Path to the image.

    Returns:
        PIL.Image in RGB mode.

    Raises:
        ValueError: If image is corrupt or unreadable.
    """
    try:
        image = Image.open(image_path)
        # Strip EXIF to avoid orientation issues
        image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Cannot read image at '{image_path}': {e}") from e


def normalize_claim(claim_text: str, max_length: int = 512) -> str:
    """
    Normalize claim text: strip whitespace, collapse multiple spaces,
    and truncate to max token-safe length.

    Args:
        claim_text: Raw claim string.
        max_length: Maximum character length.

    Returns:
        Normalized string.
    """
    import re
    text = claim_text.strip()
    text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces/newlines
    return text[:max_length]


def safe_truncate(text: str, max_chars: int = 500) -> str:
    """Safely truncate text to max_chars without cutting mid-word."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return truncated[:last_space] + "..." if last_space > 0 else truncated + "..."
