import logging

def setup_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True  # Memastikan konfigurasi di-apply ulang jika dipanggil berkali-kali (idempotent)
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)