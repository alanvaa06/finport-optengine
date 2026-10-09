"""An on-disk cache, so iterating on a strategy does not re-download the world.

A walk-forward sweep re-reads the same panel dozens of times. Without a cache
that is dozens of downloads, a rate limit, and a wait long enough that people
stop iterating. With one it is a single fetch and a Parquet read.

Three properties the design is built around:

* **Opt-in.** No directory, no cache. The library never writes to a user's
  filesystem because they called a function.
* **Keyed by what affects the data.** The key comes from
  :meth:`~optimization_engine.ingest.spec.IngestRequest.fingerprint`, which
  covers the universe, window, interval, fields, provider and currency — and
  deliberately excludes worker count and cache settings, which change how the
  fetch runs rather than what it returns.
* **Never poisonous.** A corrupt, unreadable or half-written entry is a cache
  miss, not an exception. An interrupted run cannot leave a truncated entry
  that a later run trusts.

One entry is one file, and that is load-bearing rather than incidental. The
obvious layout — a directory per entry holding a Parquet per field — cannot be
published atomically: ``rename`` onto a directory that already exists fails,
so publishing means removing the old one first, and two writers racing through
that gap leave one of them with ``Directory not empty`` and readers with a
window where the entry does not exist at all. ``os.replace`` on a *file* has
neither problem. So an entry is a Zip holding exactly what the directory would
have held, written to a temporary name beside it and moved into place in one
step: a reader sees the old entry or the new one, a second writer simply wins,
and a reader already mid-read keeps the file it opened.

Inside the Zip the frames are plain NumPy arrays rather than Parquet, and that
is a dependency decision rather than a taste one. A cache is core behaviour —
it turns on the moment a directory is named — so it must not require a package
the project does not depend on. Parquet needs ``pyarrow``; NumPy is already a
hard dependency, and a panel is always float64 values on a ``DatetimeIndex``,
which ``.npy`` round-trips bit for bit. The index and the column names travel
in the manifest beside them.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import shutil
import tempfile
import time
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from optimization_engine.ingest import fields as F
from optimization_engine.ingest.panel import PricePanel, SeriesMeta

_LOG = logging.getLogger(__name__)

_MANIFEST = "manifest.json"

#: Bumped when the on-disk layout changes. An entry written by a different
#: version is a miss, never a parse attempt — version 1 was a directory per
#: entry, which is why version 2 exists.
_FORMAT_VERSION = 2

#: Suffix that marks a complete entry. A file only carries it at the instant
#: it is whole, so a half-written one can never be mistaken for a hit.
_SUFFIX = ".panel"

#: Bare arrays do not self-compress the way Parquet does, so the Zip does it.
_COMPRESSION = zipfile.ZIP_DEFLATED

#: How hard :meth:`PanelCache.store` tries to publish an entry, and how long
#: it waits between attempts (linear: 10 ms, 20 ms, ...). This exists for
#: Windows, where ``os.replace`` is refused with ``PermissionError``
#: (``WinError 5``) while any other process holds the target open — a reader
#: mid-:meth:`~PanelCache.load`, or a second writer of the same key. POSIX
#: renames straight over an open file, so there the first attempt always wins
#: and neither the sleep nor the loop is ever reached.
_REPLACE_ATTEMPTS = 5
_REPLACE_BACKOFF_SECONDS = 0.010

#: The reader's side of the same refusal. While ``os.replace`` is publishing
#: over an entry on Windows, opening that entry raises ``PermissionError``
#: (``Errno 13``) for the instant the rename holds it, so :meth:`PanelCache.load`
#: retries the open (linear: 5 ms, 10 ms, ...) instead of reporting an entry
#: that exists as a miss. On POSIX the open is never refused this way; a
#: genuinely unreadable entry costs these few waits once, then is a miss.
_OPEN_ATTEMPTS = 5
_OPEN_BACKOFF_SECONDS = 0.005


@dataclass(frozen=True)
class CacheEntry:
    """What a cache lookup found, for reporting."""

    key: str
    path: Path
    age_seconds: float
    fields: tuple[str, ...]
    #: Run-level notes the fetch that wrote the entry attached to it — a
    #: series whose currency was never declared is still undeclared on a
    #: warm run, and saying so only on the cold one hid it for the TTL.
    notes: tuple[str, ...] = ()

    @property
    def age_label(self) -> str:
        """How old this entry is, in the largest unit that stays readable.

        Returns:
            ``"just now"`` under a minute, then minutes, hours, and days. Meant for
            a provider panel or a CLI line, not for arithmetic — use
            :attr:`age_seconds` for that.
        """
        minutes = self.age_seconds / 60.0
        if minutes < 1:
            return "just now"
        if minutes < 60:
            return f"{minutes:.0f} min ago"
        hours = minutes / 60.0
        if hours < 24:
            return f"{hours:.1f} h ago"
        return f"{hours / 24.0:.1f} days ago"


class PanelCache:
    """Reads and writes :class:`PricePanel` objects under a directory.

    Each entry is one Zip file named by the request fingerprint, holding a
    NumPy array per field plus a JSON manifest with the index, the column
    names, the provenance and the write time. The single file is what makes
    publishing atomic; NumPy is what keeps the entry free of any dependency
    the project does not already have.
    """

    def __init__(self, directory: str | Path, ttl_seconds: int = 24 * 60 * 60) -> None:
        """Point the cache at a directory.

        Args:
            directory: Where entries are written. Created on first write, not here.
            ttl_seconds: How long an entry stays fresh. Anything older is a miss.
                Defaults to 24 hours.
        """
        self.directory = Path(directory)
        self.ttl_seconds = int(ttl_seconds)

    def path_for(self, key: str) -> Path:
        """Where the entry for a fingerprint lives.

        Args:
            key: A request fingerprint.

        Returns:
            The Zip file's path, whether or not it exists.
        """
        return self.directory / f"{key}{_SUFFIX}"

    def load(self, key: str) -> tuple[PricePanel, CacheEntry] | None:
        """Return a cached panel, or ``None`` on any miss.

        Args:
            key: The request fingerprint.

        Returns:
            A ``(panel, entry)`` pair, or ``None``. A miss includes: no entry, an
            entry older than the TTL, one written by a different format version,
            and any read failure at all. Nothing here raises — a broken cache must
            degrade to a fetch, never to a stack trace, because the fetch is
            always available and always correct.
        """
        path = self.path_for(key)
        if not path.is_file():
            return None

        try:
            with _open_entry(path) as archive:
                manifest = json.loads(archive.read(_MANIFEST).decode("utf-8"))
                if int(manifest.get("format_version", 0)) != _FORMAT_VERSION:
                    return None

                age = time.time() - float(manifest["written_at"])
                if self.ttl_seconds and age > self.ttl_seconds:
                    return None

                index = _decode_index(manifest)
                columns = list(manifest["identifiers"])
                frames = {
                    name: pd.DataFrame(
                        np.load(io.BytesIO(archive.read(f"{name}.npy"))),
                        index=index,
                        columns=columns,
                    )
                    for name in manifest["fields"]
                }

            meta = {
                identifier: SeriesMeta(
                    identifier=identifier,
                    provider_symbol=record["provider_symbol"],
                    provider=record["provider"],
                    kind=F.InstrumentKind(record["kind"]),
                    currency=record.get("currency"),
                    name=record.get("name"),
                    exchange=record.get("exchange"),
                )
                for identifier, record in manifest.get("meta", {}).items()
            }
            panel = PricePanel.from_frames(frames, meta)
        except Exception as exc:  # corrupt entry, schema drift, partial write
            _LOG.debug("Ignoring unreadable cache entry %s: %s", key, exc)
            return None

        return panel, CacheEntry(
            key=key,
            path=path,
            age_seconds=age,
            fields=tuple(manifest["fields"]),
            notes=tuple(str(note) for note in manifest.get("notes", ())),
        )

    def store(self, key: str, panel: PricePanel, notes: Sequence[str] = ()) -> bool:
        """Write a panel to the cache. Returns whether the entry is in place.

        The entry is built under a temporary name in the same directory — so
        the move below stays on one filesystem — and then moved into place
        with :func:`os.replace`. That call is atomic, which is what lets two
        runs fetch the same request concurrently without either of them
        failing and without a reader ever seeing a partial entry.

        Atomic is not the same as always permitted, though. On Windows
        ``os.replace`` raises ``PermissionError`` while another process holds
        the target open — exactly what a reader mid-:meth:`load` or a second
        writer of this key does — so the publish is retried a few times with a
        short backoff rather than reported as a failed write on the first
        refusal.

        Args:
            key: The request fingerprint to file it under.
            panel: The panel to write.
            notes: Run-level warnings about the panel itself, returned with it
                on every hit as :attr:`CacheEntry.notes`.

        Returns:
            ``True`` when an entry for ``key`` is on disk afterwards, whether
            this call published it or a concurrent writer of the same key won
            the race — the fingerprint covers everything that affects the
            data, so their entry is the one this call would have written.
            ``False`` otherwise, including when every publish was refused and
            the entry left in place is one that was already there when this
            call began — on Windows, an older entry a reader holds open. That
            entry is not this call's result. Also ``False``, without writing,
            for a panel whose index the format cannot carry back. Failures
            are logged and swallowed, because a read-only directory or a full
            disk should slow the next run down, not fail this one.
        """
        target = self.path_for(key)
        handle, staging = -1, ""
        # What was at the target before this call, so a refused publish can
        # tell a racing writer's fresh entry from an old one held open.
        before = _identity(target)
        try:
            if not _decode_index(_encode_index(panel.index)).equals(panel.index):
                _LOG.warning(
                    "Not caching panel %s: its index (%s) does not survive the "
                    "round trip through the cache format",
                    key,
                    panel.index.dtype,
                )
                return False
            self.directory.mkdir(parents=True, exist_ok=True)
            handle, staging = tempfile.mkstemp(
                dir=self.directory, prefix=f".{key}.", suffix=".tmp"
            )
            with os.fdopen(handle, "wb") as raw:
                handle = -1  # now owned by the file object
                self._write_archive(raw, panel, notes)

            # One atomic step. A concurrent writer of the same key simply
            # wins; a reader holding the old file keeps reading it. On
            # Windows an open target makes that step raise instead, so back
            # off briefly and try again — whoever holds the file is reading an
            # entry, not doing something long.
            for attempt in range(1, _REPLACE_ATTEMPTS + 1):
                try:
                    os.replace(staging, target)
                except PermissionError:
                    if attempt == _REPLACE_ATTEMPTS:
                        break
                    time.sleep(_REPLACE_BACKOFF_SECONDS * attempt)
                else:
                    staging = ""  # renamed away; nothing left to clean up
                    return True

            # Every attempt was refused, so this call did not publish. If a
            # *different* entry is there now, a racing writer of the same key
            # published one while we backed off, and by the paragraph above
            # that is this call's own result — a hit, not a lost write. The
            # same entry as before is not: it is an older one that somebody
            # is holding open, which is what refuses the replace on Windows.
            # Note the check is here rather than above ``os.replace``: a
            # write that never got that far (a serialization error, a full
            # disk) must still report False even when an entry exists.
            #
            # service.py reads this bool and says "Not cached" when it is
            # False, so the caller hears about a write that did not land.
            after = _identity(target)
            if after is not None and after != before:
                _LOG.debug(
                    "Could not publish cache entry %s (%d attempts refused), "
                    "but a concurrent writer left one in place",
                    key,
                    _REPLACE_ATTEMPTS,
                )
                return True

            _LOG.warning(
                "Could not cache panel %s: publishing was refused %d times "
                "and %s",
                key,
                _REPLACE_ATTEMPTS,
                "an older entry, held open elsewhere, was left in place"
                if after is not None
                else "no entry is in place",
            )
            return False
        except Exception as exc:
            _LOG.warning("Could not cache panel %s: %s", key, exc)
            return False
        finally:
            # Every exit that did not rename the staging file away has to
            # remove it, including the one where a racing writer published for
            # us. The temporary name never carried the entry suffix, so even
            # if this cleanup fails the leftover cannot be read as a hit.
            if handle != -1:
                os.close(handle)
            if staging:
                with contextlib.suppress(OSError):
                    os.unlink(staging)

    @staticmethod
    def _write_archive(stream, panel: PricePanel, notes: Sequence[str] = ()) -> None:
        """Serialize a panel into an open binary stream as a Zip.

        The index is stored once, as nanoseconds since the epoch, because every
        field frame shares it — :meth:`PricePanel.from_frames` guarantees that,
        and :meth:`PricePanel.validate` enforces it.
        """
        manifest = {
            "format_version": _FORMAT_VERSION,
            "written_at": time.time(),
            "fields": list(panel.frames),
            "identifiers": list(panel.identifiers),
            **_encode_index(panel.index),
            "meta": {
                identifier: {
                    "provider_symbol": record.provider_symbol,
                    "provider": record.provider,
                    "kind": record.kind.value,
                    "currency": record.currency,
                    "name": record.name,
                    "exchange": record.exchange,
                }
                for identifier, record in panel.meta.items()
            },
            "notes": [str(note) for note in notes],
        }
        with zipfile.ZipFile(stream, "w", compression=_COMPRESSION) as archive:
            archive.writestr(_MANIFEST, json.dumps(manifest))
            for name, frame in panel.frames.items():
                buffer = io.BytesIO()
                np.save(buffer, frame.to_numpy(dtype="float64"))
                archive.writestr(f"{name}.npy", buffer.getvalue())

    def clear(self) -> int:
        """Delete every entry. Returns how many were removed.

        Sweeps up three things: current entries, the directories version 1
        wrote, and any temporary file an interrupted run left behind.
        """
        if not self.directory.is_dir():
            return 0
        removed = 0
        for child in self.directory.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
            elif child.suffix == _SUFFIX or child.name.endswith(".tmp"):
                with contextlib.suppress(OSError):
                    child.unlink()
                    removed += 1
        return removed


def _encode_index(index: pd.DatetimeIndex) -> dict[str, object]:
    """The manifest fields that carry a panel's dates.

    The unit travels with the values: pandas builds an index at microsecond
    or nanosecond resolution depending on how it was constructed, and a panel
    that comes back at a different one is not the panel that went in. So
    does the time zone, separately: ``datetime64[us, America/New_York]`` is
    not a dtype NumPy can cast to, and storing it as one made every load of
    a tz-aware panel a miss while ``store`` reported success.
    """
    naive = index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index
    return {
        "index": naive.view("int64").tolist(),
        "index_dtype": str(naive.dtype),
        "index_tz": str(index.tz) if index.tz is not None else None,
    }


def _decode_index(manifest: dict) -> pd.DatetimeIndex:
    """Rebuild the index :func:`_encode_index` wrote."""
    index = pd.DatetimeIndex(
        np.asarray(manifest["index"], dtype="int64").astype(
            manifest.get("index_dtype", "datetime64[ns]")
        ),
        name="date",
    )
    tz = manifest.get("index_tz")
    return index.tz_localize("UTC").tz_convert(tz) if tz else index


def _open_entry(path: Path) -> zipfile.ZipFile:
    """Open an entry, waiting out a publish that is replacing it right now.

    Only ``PermissionError`` is retried — that is the refusal a concurrent
    ``os.replace`` causes on Windows. Anything else, and the last refusal,
    propagates for :meth:`PanelCache.load` to report as a miss. Once open, the
    handle keeps reading the file it got even if a newer entry lands.
    """
    for attempt in range(1, _OPEN_ATTEMPTS):
        try:
            return zipfile.ZipFile(path)
        except PermissionError:
            time.sleep(_OPEN_BACKOFF_SECONDS * attempt)
    return zipfile.ZipFile(path)


def _identity(path: Path) -> tuple[int, int, int] | None:
    """Which file is at ``path``: inode, size and mtime, or ``None``."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


__all__ = ["CacheEntry", "PanelCache"]
