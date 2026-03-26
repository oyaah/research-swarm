from __future__ import annotations

from frontend.cli_http import parse_sse_lines


class _FakeResp:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def iter_lines(self, decode_unicode: bool = True):
        for line in self._lines:
            yield line


def test_parse_sse_lines_yields_event_payload_pairs():
    resp = _FakeResp([
        "event: trace",
        "data: {\"a\":1}",
        "",
        "event: done",
        "data: {\"status\":\"completed\"}",
        "",
    ])
    out = list(parse_sse_lines(resp))
    assert out[0][0] == "trace"
    assert "\"a\":1" in out[0][1]
    assert out[1][0] == "done"
