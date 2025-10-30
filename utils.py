import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_logger():
    logger = logging.getLogger(__name__)
    return logger


def sanitize_text(text: str) -> str:
    text = " ".join(text.split())
    return text.replace("\x00", "").strip()
