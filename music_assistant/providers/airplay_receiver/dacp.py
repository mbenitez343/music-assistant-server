"""DACP remote control of the AirPlay client streaming to a receiver."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aiohttp import ClientError, ClientTimeout
from zeroconf.asyncio import AsyncServiceInfo

if TYPE_CHECKING:
    from aiohttp import ClientSession
    from zeroconf import Zeroconf

DACP_SERVICE_TYPE = "_dacp._tcp.local."
# iOS ignores or rejects an absolute volume over DACP, so a target is reached with
# volumeup/volumedown steps: 16 of them from -30 dB to 0 dB, -144 dB being mute
AIRPLAY_VOLUME_MIN_DB = -30.0
VOLUME_STEP_DB = 30.0 / 16
VOLUME_STEP_INTERVAL = 0.035
# the client reports every step back, and for this long those reports are ours
VOLUME_ECHO_WINDOW = 1.5
REQUEST_TIMEOUT = 5.0


class DacpUnavailableError(Exception):
    """The AirPlay client did not offer (or no longer offers) a DACP remote."""


@dataclass
class DacpRemote:
    """Remote control of the AirPlay client of one receiver session."""

    dacp_id: str | None = None
    active_remote: str | None = None
    client_ip: str | None = None
    # the client's last reported AirPlay volume in dB
    airplay_volume: float | None = None
    _address: tuple[str, int] | None = None
    _volume_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _volume_echo_until: float = 0.0
    _volume_target: int | None = None

    @property
    def available(self) -> bool:
        """Return whether the current session has offered a DACP remote."""
        return bool(self.dacp_id and self.active_remote)

    @property
    def volume_echo(self) -> bool:
        """Return whether client volume reports are currently echoes of our own volume walk."""
        return self._volume_lock.locked() or time.monotonic() < self._volume_echo_until

    def update(
        self,
        dacp_id: str | None = None,
        active_remote: str | None = None,
        client_ip: str | None = None,
    ) -> None:
        """
        Record session identifiers announced by the client.

        :param dacp_id: The DACP id (``daid``) of the session.
        :param active_remote: The Active-Remote token (``acre``) of the session.
        :param client_ip: The client's IP address (``clip``).
        """
        if dacp_id is not None:
            # a 64-bit id: shairport-sync passes it on without leading zeros, while the
            # client advertises iTunes_Ctrl_ with all 16 hex digits
            dacp_id = dacp_id.strip().upper().zfill(16)
            if dacp_id != self.dacp_id:
                self.dacp_id = dacp_id
                self._address = None
        if active_remote is not None:
            self.active_remote = active_remote
        if client_ip is not None:
            self.client_ip = client_ip

    def clear(self) -> None:
        """Forget the session (the client disconnected)."""
        self.dacp_id = None
        self.active_remote = None
        self.client_ip = None
        self.airplay_volume = None
        self._address = None

    async def send(self, zeroconf: Zeroconf, http: ClientSession, command: str) -> None:
        """
        Send one DACP command (e.g. ``play``, ``pause``, ``nextitem``) to the client.

        :param zeroconf: Zeroconf instance used to find the client's DACP service.
        :param http: HTTP session used for the request.
        :param command: The DACP command name.
        """
        if not self.available:
            raise DacpUnavailableError("the AirPlay client offered no remote control")
        # the client advertises several addresses and not all of them answer (a stale
        # IPv6 one was seen with iOS). Use the first that connects and remember it, and
        # look the service up again when the remembered one stops answering
        error: Exception | None = None
        remembered, self._address = self._address, None
        if remembered:
            error = await self._request(http, remembered, command)
            if error is None:
                self._address = remembered
                return
        for address in await self._lookup(zeroconf):
            error = await self._request(http, address, command)
            if error is None:
                self._address = address
                return
        raise DacpUnavailableError(f"DACP {command} failed: {error}")

    async def set_volume(self, zeroconf: Zeroconf, http: ClientSession, volume: int) -> None:
        """
        Walk the client's volume to the given level with relative steps.

        :param zeroconf: Zeroconf instance used to find the client's DACP service.
        :param http: HTTP session used for the requests.
        :param volume: Target volume level (0-100).
        """
        # a walk in progress picks up the newest target instead of queueing another
        self._volume_target = volume
        if self._volume_lock.locked():
            return
        async with self._volume_lock:
            try:
                # a few rounds to correct missed steps, and more while the target moves
                for _ in range(8):
                    volume = self._volume_target
                    # unknown until the client reports it (and forgotten when the session
                    # ends mid-walk): stepping from a guess could blast it to full volume
                    current = self.airplay_volume
                    if current is None or volume is None:
                        return
                    steps = volume_steps(current, volume)
                    if steps == 0:
                        if volume == self._volume_target:
                            return
                        continue
                    command = "volumeup" if steps > 0 else "volumedown"
                    start = time.monotonic()
                    for index in range(abs(steps)):
                        # a steady cadence: the request time is part of the interval
                        delay = start + index * VOLUME_STEP_INTERVAL - time.monotonic()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        await self.send(zeroconf, http, command)
                    # let the client report where it ended up before re-checking
                    await asyncio.sleep(0.5)
            finally:
                self._volume_echo_until = time.monotonic() + VOLUME_ECHO_WINDOW

    async def _request(
        self, http: ClientSession, address: tuple[str, int], command: str
    ) -> Exception | None:
        """
        Send one DACP command to one address of the client.

        :param http: HTTP session used for the request.
        :param address: The (host, port) to send it to.
        :param command: The DACP command name.
        :return: The connection error, or None when the client accepted the command.
        """
        host, port = address
        url = f"http://{host}:{port}/ctrl-int/1/{command}"
        headers = {"Active-Remote": str(self.active_remote)}
        try:
            async with http.get(
                url, headers=headers, timeout=ClientTimeout(total=REQUEST_TIMEOUT)
            ) as resp:
                if resp.status not in (200, 204):
                    raise DacpUnavailableError(f"DACP {command} answered HTTP {resp.status}")
        except (OSError, TimeoutError, ClientError) as err:
            return err
        return None

    async def _lookup(self, zeroconf: Zeroconf) -> list[tuple[str, int]]:
        """Return the addresses of the client's DACP service, in the order to try them."""
        name = f"iTunes_Ctrl_{self.dacp_id}.{DACP_SERVICE_TYPE}"
        info = AsyncServiceInfo(DACP_SERVICE_TYPE, name)
        if not await info.async_request(zeroconf, REQUEST_TIMEOUT * 1000) or not info.port:
            raise DacpUnavailableError(f"DACP service {name} not found")
        port = info.port
        # IPv4 first, then the address the client streams from, then the rest
        ordered = sorted(info.parsed_addresses(), key=lambda a: (":" in a, a != self.client_ip))
        if not ordered:
            raise DacpUnavailableError(f"DACP service {name} has no address")
        return [(f"[{a}]" if ":" in a else a, port) for a in ordered]


def volume_steps(current_db: float, target: int) -> int:
    """
    Return the number of volume steps (positive = up) from the client's volume to a target.

    :param current_db: The client's AirPlay volume in dB (-144 when muted).
    :param target: The target volume level (0-100).
    """
    # the clamp also maps mute (-144 dB) to the bottom of the range
    current_db = max(AIRPLAY_VOLUME_MIN_DB, min(0.0, current_db))
    target_db = AIRPLAY_VOLUME_MIN_DB + 30.0 * max(0, min(100, target)) / 100
    return round((target_db - current_db) / VOLUME_STEP_DB)
