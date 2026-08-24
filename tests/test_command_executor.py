from mycode.tools.command_output import BoundedOutputCapture


def test_bounded_output_capture_keeps_tail_across_chunks() -> None:
    capture = BoundedOutputCapture(max_chars=5)

    capture.append("abc")
    capture.append("defg")
    capture.append("h")

    snapshot = capture.snapshot()

    assert snapshot.content == "defgh"
    assert snapshot.chars == 8
    assert snapshot.truncated is True


def test_bounded_output_capture_supports_zero_char_limit() -> None:
    capture = BoundedOutputCapture(max_chars=0)

    capture.append("abc")

    snapshot = capture.snapshot()

    assert snapshot.content == ""
    assert snapshot.chars == 3
    assert snapshot.truncated is True
