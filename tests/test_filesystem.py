import json
from pathlib import Path

import pytest

from mycode.persistence.filesystem import (
    JsonLinesCorruptionError,
    JsonSnapshotError,
    StorageBoundaryError,
    TrailingRecordPolicy,
    append_jsonl_record,
    read_json_snapshot,
    read_jsonl_records,
    write_json_snapshot,
    prepare_jsonl_for_append,
)


def test_json_snapshot_round_trip_is_atomic_and_leaves_no_temp_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session" / "meta.json"

    write_json_snapshot(tmp_path, path, {"title": "第一轮", "version": 1})
    write_json_snapshot(tmp_path, path, {"title": "第二轮", "version": 1})

    assert read_json_snapshot(tmp_path, path) == {
        "title": "第二轮",
        "version": 1,
    }
    assert not list(path.parent.glob(".meta.json.*.tmp"))


def test_json_snapshot_rejects_non_object_and_non_finite_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "meta.json"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(JsonSnapshotError, match="must contain an object"):
        read_json_snapshot(tmp_path, path)
    with pytest.raises(JsonSnapshotError, match="not serializable"):
        write_json_snapshot(tmp_path, path, {"ratio": float("nan")})


def test_jsonl_append_round_trip_uses_one_canonical_line_per_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transcript.jsonl"

    append_jsonl_record(tmp_path, path, {"sequence": 1, "text": "你好"})
    append_jsonl_record(tmp_path, path, {"sequence": 2, "text": "done"})

    assert read_jsonl_records(tmp_path, path) == [
        {"sequence": 1, "text": "你好"},
        {"sequence": 2, "text": "done"},
    ]
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 2
    assert all(isinstance(json.loads(line), dict) for line in raw_lines)


@pytest.mark.parametrize("policy", ["ignore", "truncate"])
def test_jsonl_can_recover_only_invalid_trailing_record(
    tmp_path: Path,
    policy: TrailingRecordPolicy,
) -> None:
    path = tmp_path / "transcript.jsonl"
    good = b'{"sequence":1}\n'
    path.write_bytes(good + b'{"sequence":')

    records = read_jsonl_records(tmp_path, path, trailing_record=policy)

    assert records == [{"sequence": 1}]
    if policy == "truncate":
        assert path.read_bytes() == good
    else:
        assert path.read_bytes() != good


def test_jsonl_error_policy_rejects_unterminated_invalid_tail(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transcript.jsonl"
    original = b'{"sequence":1}\n{"sequence":'
    path.write_bytes(original)

    with pytest.raises(JsonLinesCorruptionError, match="trailing.*line 2"):
        read_jsonl_records(tmp_path, path, trailing_record="error")

    assert path.read_bytes() == original


@pytest.mark.parametrize("policy", ["error", "ignore", "truncate"])
def test_jsonl_rejects_invalid_final_record_with_newline(
    tmp_path: Path,
    policy: TrailingRecordPolicy,
) -> None:
    path = tmp_path / "transcript.jsonl"
    original = b'{"sequence":1}\nnot-json\n'
    path.write_bytes(original)

    with pytest.raises(JsonLinesCorruptionError, match="trailing.*line 2"):
        read_jsonl_records(tmp_path, path, trailing_record=policy)

    assert path.read_bytes() == original


def test_jsonl_reads_valid_final_record_without_newline(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    path.write_bytes(b'{"sequence":1}\n{"sequence":2}')

    assert read_jsonl_records(
        tmp_path,
        path,
        trailing_record="truncate",
    ) == [{"sequence": 1}, {"sequence": 2}]
    assert path.read_bytes().endswith(b"}")


def test_jsonl_append_after_truncated_partial_tail_remains_readable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transcript.jsonl"
    path.write_bytes(b'{"sequence":1}\n{"sequence":')

    assert read_jsonl_records(
        tmp_path,
        path,
        trailing_record="truncate",
    ) == [{"sequence": 1}]
    append_jsonl_record(tmp_path, path, {"sequence": 2})

    assert read_jsonl_records(tmp_path, path) == [
        {"sequence": 1},
        {"sequence": 2},
    ]


def test_jsonl_middle_corruption_is_never_recovered(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    original = b'{"sequence":1}\nnot-json\n{"sequence":3}\n'
    path.write_bytes(original)

    with pytest.raises(JsonLinesCorruptionError, match="middle.*line 2"):
        read_jsonl_records(tmp_path, path, trailing_record="truncate")

    assert path.read_bytes() == original


def test_storage_paths_reject_traversal_and_outside_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(StorageBoundaryError, match="traversal"):
        write_json_snapshot(root, root / "nested" / ".." / "meta.json", {})

    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")

    with pytest.raises(StorageBoundaryError, match="escapes|symlink"):
        write_json_snapshot(root, link / "meta.json", {})


@pytest.mark.parametrize("data,expected", [
    (b"", b""),
    (b'{"a":1}\n', b'{"a":1}\n'),
    (b'{"a":1}\r\n', b'{"a":1}\r\n'),
    (b'{"a":1}', b'{"a":1}\n'),
    (b'{"a":1}\n{"a":', b'{"a":1}\n'),
    (b'{"a":', b""),
])
def test_prepare_for_append_normalizes_once(tmp_path, data, expected):
    path = tmp_path / "log.jsonl"
    path.write_bytes(data)
    records = prepare_jsonl_for_append(tmp_path, path)
    assert records == ([{"a": 1}] if expected else [])
    assert path.read_bytes() == expected
    prepare_jsonl_for_append(tmp_path, path)
    assert path.read_bytes() == expected
    append_jsonl_record(tmp_path, path, {"b": 2})
    assert read_jsonl_records(tmp_path, path)[-1] == {"b": 2}


@pytest.mark.parametrize("data", [b"bad\n", b"bad\r\n", b"bad\n{}", b"[]"])
def test_prepare_for_append_rejects_committed_or_schema_corruption(tmp_path, data):
    path = tmp_path / "log.jsonl"
    path.write_bytes(data)
    with pytest.raises(JsonLinesCorruptionError):
        prepare_jsonl_for_append(tmp_path, path)
    assert path.read_bytes() == data
