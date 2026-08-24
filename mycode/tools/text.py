SUPPORTED_TEXT_ENCODINGS = ("utf-8", "gbk")


def contains_nul_byte(raw_content: bytes) -> bool:
    return b"\x00" in raw_content


def decode_text(raw_content: bytes) -> tuple[str, str] | None:
    for encoding in SUPPORTED_TEXT_ENCODINGS:
        try:
            return raw_content.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    return None
