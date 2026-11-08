"""Unit tests for DACP remote control of the AirPlay client."""

from __future__ import annotations

import asyncio
from typing import Any, Self, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.enums import SourceControl
from music_assistant_models.errors import UnsupportedFeaturedException

from music_assistant.providers.airplay_receiver import AirPlayReceiverProvider
from music_assistant.providers.airplay_receiver.dacp import (
    DacpRemote,
    volume_steps,
)


def test_volume_steps() -> None:
    """Steps follow the 16-step AirPlay scale, with mute counting as the bottom."""
    assert volume_steps(-144.0, 50) == 8
    assert volume_steps(-15.0, 25) == -4
    assert volume_steps(-22.5, 75) == 8
    assert volume_steps(-15.0, 50) == 0
    assert volume_steps(-7.5, 100) == 4
    assert volume_steps(0.0, 0) == -16


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def _http(status: int = 200) -> MagicMock:
    http = MagicMock()
    http.get = MagicMock(return_value=_Response(status))
    return http


def _remote() -> DacpRemote:
    remote = DacpRemote()
    remote.update(dacp_id="A1B2C3D4E5F60718", active_remote="1234567890", client_ip="192.0.2.10")
    remote._address = ("192.0.2.10", 50001)
    return remote


async def test_send_uses_active_remote_header() -> None:
    """A command is a GET to /ctrl-int/1/<command> carrying the session's Active-Remote."""
    http = _http()
    await _remote().send(MagicMock(), http, "playpause")
    url = http.get.call_args.args[0]
    assert url == "http://192.0.2.10:50001/ctrl-int/1/playpause"
    assert http.get.call_args.kwargs["headers"] == {"Active-Remote": "1234567890"}


async def test_new_dacp_id_forgets_the_resolved_address() -> None:
    """A new session may come from another device: resolve its service again."""
    remote = _remote()
    remote.update(dacp_id="B2C3D4E5F6071829")
    assert remote._address is None


async def test_set_volume_walks_in_steps_and_marks_echo() -> None:
    """The volume is reached with relative steps, re-read from the client's reports."""
    remote = _remote()
    remote.airplay_volume = -144.0
    sent: list[str] = []

    async def _send(_zc: Any, _http: Any, command: str) -> None:
        sent.append(command)
        # the client reports each step back, as iOS does
        current = remote.airplay_volume
        current = -30.0 if current is None or current < -30 else current
        remote.airplay_volume = current + (1.875 if command == "volumeup" else -1.875)

    with (
        patch.object(remote, "send", side_effect=_send),
        patch("music_assistant.providers.airplay_receiver.dacp.asyncio.sleep", AsyncMock()),
    ):
        await remote.set_volume(MagicMock(), MagicMock(), 50)
    assert sent == ["volumeup"] * 8
    assert remote.airplay_volume == -15.0
    assert remote.volume_echo


def _provider_with_daemon() -> tuple[AirPlayReceiverProvider, MagicMock]:
    prov = AirPlayReceiverProvider.__new__(AirPlayReceiverProvider)
    prov.logger = MagicMock()
    prov.mass = MagicMock()
    prov.mass.players.cmd_stop = AsyncMock()
    # run real coroutines (the volume walk) and ignore the mocked ones
    prov.mass.create_task = lambda c: asyncio.ensure_future(c) if asyncio.iscoroutine(c) else None
    daemon = MagicMock()
    daemon.dacp = _remote()
    daemon.last_client_volume = None
    prov._daemons = {"player-a": daemon}
    return prov, daemon


async def test_source_control_without_remote_is_unsupported() -> None:
    """A client that offered no DACP remote gets a clear error, not a silent no-op."""
    prov, daemon = _provider_with_daemon()
    daemon.dacp.clear()
    with pytest.raises(UnsupportedFeaturedException):
        await prov.on_source_control("player-a", SourceControl.PAUSE)


async def test_volume_change_from_the_client_is_not_sent_back() -> None:
    """The client's own volume change, applied to the player, must not echo back to it."""
    prov, daemon = _provider_with_daemon()
    daemon.last_client_volume = 40
    with patch.object(daemon.dacp, "set_volume", AsyncMock()) as set_volume:
        await prov.on_volume_change("player-a", 40)
        await asyncio.sleep(0)
        set_volume.assert_not_awaited()
        await prov.on_volume_change("player-a", 70)
        await asyncio.sleep(0)
        assert set_volume.await_args is not None
        assert set_volume.await_args.args[2] == 70


async def test_pause_keeps_the_source_when_the_client_ends_its_session() -> None:
    """A DACP pause stops the player at once and keeps the source for a later PLAY."""
    prov, daemon = _provider_with_daemon()
    daemon.remote_paused = False
    daemon.in_use_by_player = "player-a"
    daemon.active_player_id = "player-a"
    prov._clear_active_player = MagicMock()  # type: ignore[method-assign]
    prov._write_silence_to_unblock_stream = MagicMock()  # type: ignore[method-assign]
    with patch.object(daemon.dacp, "send", AsyncMock()):
        await prov.on_source_control("player-a", SourceControl.PAUSE)
    assert daemon.remote_paused
    # the player stops at once, so MA shows it paused and a toggle sends PLAY next
    cmd_stop = cast("AsyncMock", prov.mass.players.cmd_stop)
    cmd_stop.assert_awaited_once_with("player-a")
    # iOS then ends its session: that must not release the source or the remote
    prov._handle_play_state_change(daemon, "stopped")
    prov._clear_active_player.assert_not_called()
    assert daemon.dacp.available
    assert daemon.active_player_id == "player-a"
    assert daemon.in_use_by_player is None


async def test_resume_restarts_playback_on_the_player() -> None:
    """PLAY after a DACP pause resumes the client and plays the source on the player again."""
    prov, daemon = _provider_with_daemon()
    daemon.remote_paused = True
    daemon.in_use_by_player = None
    daemon.active_player_id = "player-a"
    daemon.pending_start_task = None
    prov._start_playback = MagicMock()  # type: ignore[method-assign]
    with patch.object(daemon.dacp, "send", AsyncMock()) as send:
        await prov.on_source_control("player-a", SourceControl.PLAY)
    assert send.await_args is not None
    assert send.await_args.args[2] == "play"
    assert not daemon.remote_paused
    prov._start_playback.assert_called_once_with(daemon, "player-a")


async def test_client_stop_without_our_pause_releases_the_source() -> None:
    """A stop the client chose itself still ends everything, as before."""
    prov, daemon = _provider_with_daemon()
    daemon.remote_paused = False
    daemon.in_use_by_player = "player-a"
    prov._clear_active_player = MagicMock()  # type: ignore[method-assign]
    prov._write_silence_to_unblock_stream = MagicMock()  # type: ignore[method-assign]
    prov._handle_play_state_change(daemon, "stopped")
    prov._clear_active_player.assert_called_once_with(daemon)
    assert not daemon.dacp.available


async def test_send_falls_back_to_the_next_address() -> None:
    """An advertised address that does not answer is skipped for the next one."""
    remote = _remote()
    remote._address = None
    calls: list[str] = []

    def _get(url: str, **_kwargs: Any) -> _Response:
        calls.append(url)
        if "[2001:db8::9]" in url:
            raise OSError("connect failed")
        return _Response(200)

    http = MagicMock()
    http.get = MagicMock(side_effect=_get)
    candidates = [("[2001:db8::9]", 50002), ("192.0.2.10", 50002)]
    with patch.object(remote, "_candidates", AsyncMock(return_value=candidates)):
        await remote.send(MagicMock(), http, "play")
    assert calls == [
        "http://[2001:db8::9]:50002/ctrl-int/1/play",
        "http://192.0.2.10:50002/ctrl-int/1/play",
    ]
    assert remote._address == ("192.0.2.10", 50002)


def test_dacp_id_is_padded_to_16_hex_digits() -> None:
    """shairport-sync drops leading zeros, but the advertised service name keeps all 16."""
    remote = DacpRemote()
    remote.update(dacp_id="ABCDEF012345678")
    assert remote.dacp_id == "0ABCDEF012345678"


async def test_set_volume_without_a_client_report_does_nothing() -> None:
    """With the client's volume unknown, stepping from a guess could blast it: leave it alone."""
    remote = _remote()
    remote.airplay_volume = None
    with patch.object(remote, "send", AsyncMock()) as send:
        await remote.set_volume(MagicMock(), MagicMock(), 60)
    send.assert_not_awaited()


async def test_set_volume_coalesces_to_the_latest_target() -> None:
    """A change arriving during a walk retargets it instead of queueing a second walk."""
    remote = _remote()
    remote.airplay_volume = -30.0
    sent: list[str] = []

    async def _send(_zc: Any, _http: Any, command: str) -> None:
        sent.append(command)
        if len(sent) == 1:
            # the slider moved on while we were walking
            await remote.set_volume(MagicMock(), MagicMock(), 25)
        current = -30.0 if remote.airplay_volume is None else remote.airplay_volume
        remote.airplay_volume = current + (1.875 if command == "volumeup" else -1.875)

    with (
        patch.object(remote, "send", side_effect=_send),
        patch("music_assistant.providers.airplay_receiver.dacp.asyncio.sleep", AsyncMock()),
    ):
        # a second walk queued behind the first would wait on its lock for ever
        await asyncio.wait_for(remote.set_volume(MagicMock(), MagicMock(), 100), timeout=5)
    assert remote.airplay_volume == -22.5, "ends at the newest target (25 %)"


async def test_send_looks_the_service_up_again_when_the_cached_address_fails() -> None:
    """A remembered address that stops answering is replaced within the same command."""
    remote = _remote()
    remote._address = ("192.0.2.10", 50001)

    def _get(url: str, **_kwargs: Any) -> _Response:
        if "50001" in url:
            raise OSError("gone")
        return _Response(200)

    http = MagicMock()
    http.get = MagicMock(side_effect=_get)
    fresh = [("192.0.2.20", 50003)]
    with patch.object(remote, "_candidates", AsyncMock(side_effect=[[remote._address], fresh])):
        await remote.send(MagicMock(), http, "pause")
    assert remote._address == ("192.0.2.20", 50003)


async def test_release_forgets_our_pause() -> None:
    """Once the player lets go of the source, a pause we sent no longer holds it."""
    prov, daemon = _provider_with_daemon()
    daemon.remote_paused = True
    daemon.active_player_id = "player-a"
    daemon.in_use_by_player = None
    await prov.on_source_released("player-a", "player-a")
    assert not daemon.remote_paused
    assert daemon.active_player_id is None


async def test_resume_on_the_client_after_our_pause_restarts_the_player() -> None:
    """Paused from MA, resumed on the phone within the same session: play the source again."""
    prov, daemon = _provider_with_daemon()
    daemon.remote_paused = True
    daemon.stop_called = False
    prov._remote_resume = MagicMock()  # type: ignore[method-assign]
    prov._on_metadata_update(daemon, {"resumed": True})
    prov._remote_resume.assert_called_once_with(daemon)
    prov._remote_resume.reset_mock()
    daemon.remote_paused = False
    prov._on_metadata_update(daemon, {"resumed": True})
    prov._remote_resume.assert_not_called()


async def test_set_volume_stops_when_the_session_ends_mid_walk() -> None:
    """The session ending during a walk (volume forgotten) ends the walk quietly."""
    remote = _remote()
    remote.airplay_volume = -30.0
    sent: list[str] = []

    async def _send(_zc: Any, _http: Any, command: str) -> None:
        sent.append(command)
        if len(sent) == 1:
            remote.clear()

    with (
        patch.object(remote, "send", side_effect=_send),
        patch("music_assistant.providers.airplay_receiver.dacp.asyncio.sleep", AsyncMock()),
    ):
        await remote.set_volume(MagicMock(), MagicMock(), 50)
    # the round already under way finishes its steps, then the walk gives up
    assert sent == ["volumeup"] * 8


async def test_volume_change_does_not_wait_for_the_client() -> None:
    """MA sets the player's own volume only after this hook returns: the walk runs apart."""
    prov, daemon = _provider_with_daemon()
    release = asyncio.Event()

    async def _slow_walk(*_args: Any) -> None:
        await release.wait()

    with patch.object(daemon.dacp, "set_volume", side_effect=_slow_walk):
        await asyncio.wait_for(prov.on_volume_change("player-a", 40), timeout=1)
        release.set()
        await asyncio.sleep(0)


async def test_volume_change_during_a_walk_reaches_the_remote() -> None:
    """A player volume change while a walk runs is handed on (to retarget it), not dropped."""
    prov, daemon = _provider_with_daemon()
    daemon.last_client_volume = 20
    with (
        patch.object(type(daemon.dacp), "volume_echo", new=True),
        patch.object(daemon.dacp, "set_volume", AsyncMock()) as set_volume,
    ):
        await prov.on_volume_change("player-a", 40)
        await asyncio.sleep(0)
    assert set_volume.await_args is not None
    assert set_volume.await_args.args[2] == 40
    assert daemon.last_client_volume is None, "the client's next report sets it again"
