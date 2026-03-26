from __future__ import annotations

import argparse

from frontend.cli_commands import handle_slash_command


def test_handle_slash_resume_action():
    args = argparse.Namespace(detail_level="brief", thinking_default="on")
    out = handle_slash_command(
        "/resume sess_abc",
        api_base="http://x",
        args=args,
        body=None,
        list_sessions_cb=lambda _api: [],
        provider_setup_cb=lambda: None,
    )
    assert out.get("action") == "resume"


def test_handle_slash_memory_toggle():
    args = argparse.Namespace(detail_level="brief", thinking_default="on")
    body = {"cross_session_memory": False}
    _ = handle_slash_command(
        "/memory on",
        api_base="http://x",
        args=args,
        body=body,
        list_sessions_cb=lambda _api: [],
        provider_setup_cb=lambda: None,
    )
    assert body["cross_session_memory"] is True
