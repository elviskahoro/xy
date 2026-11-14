"""Spatial index reader — the Tier-3 out-of-core query path (dossier §28/§32b).

A companion to the out-of-core store (`_ooc`): where the pyramid answers
*zoomed-out* views in O(tiles), the spatial index answers *zoomed-in* views in
O(points in window) instead of O(N). Points are pre-sorted on disk into a
row-major grid of cells (built by `osmium-rs`'s `osm-sort`); a viewport reads
only the cells it overlaps — one contiguous slice per grid row — so deep zoom
gets sharper *and* cheaper the further you go.

The sorted columns are a derived **f32** cache (§27: canonical stays f64; every
derived buffer is rebuildable). Load is mmap-only — no data enters RAM until a
window touches it, and only that window's cells are paged in.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_MAGIC = b"XYSPIDX1"
_HEADER = 56  # magic(8) + g(4) + pad(4) + x0x1y0y1(32) + n(8)


@dataclass
class SpatialIndex:
    """Read-only view over an `osm-sort` index (`<prefix>.idx/.lon.f32/.lat.f32`)."""

    g: int
    x0: float
    x1: float
    y0: float
    y1: float
    n: int
    offsets: npt.NDArray[np.uint64]  # (g*g + 1,) cumulative cell offsets
    lon: np.memmap  # (n,) f32, sorted by cell
    lat: np.memmap  # (n,) f32, sorted by cell

    @classmethod
    def load(cls, prefix: str) -> SpatialIndex:
        with open(prefix + ".idx", "rb") as f:
            head = f.read(_HEADER)
            if head[:8] != _MAGIC:
                raise ValueError(f"{prefix}.idx is not a spatial index")
            g = struct.unpack_from("<I", head, 8)[0]
            x0, x1, y0, y1 = struct.unpack_from("<dddd", head, 16)
            n = struct.unpack_from("<Q", head, 48)[0]
            offsets = np.fromfile(f, dtype=np.uint64, count=g * g + 1)
        lon = np.memmap(prefix + ".lon.f32", dtype=np.float32, mode="r", shape=(int(n),))
        lat = np.memmap(prefix + ".lat.f32", dtype=np.float32, mode="r", shape=(int(n),))
        return cls(int(g), x0, x1, y0, y1, int(n), offsets, lon, lat)

    def _cell_range(self, lo: float, hi: float, dlo: float, dhi: float) -> tuple[int, int]:
        span = dhi - dlo
        i0 = int((lo - dlo) / span * self.g)
        i1 = int((hi - dlo) / span * self.g)
        return max(0, min(self.g - 1, i0)), max(0, min(self.g - 1, i1))

    def window_count(self, x0: float, x1: float, y0: float, y1: float) -> int:
        """Points in the cells overlapping the window — O(rows), no point reads.
        A cheap upper bound on the exact in-window count (whole-cell overhang),
        used to decide whether an exact windowed bin is affordable."""
        ix0, ix1 = self._cell_range(x0, x1, self.x0, self.x1)
        iy0, iy1 = self._cell_range(y0, y1, self.y0, self.y1)
        g = self.g
        rows = self.offsets[(np.arange(iy0, iy1 + 1) * g)[:, None] + np.array([ix0, ix1 + 1])]
        return int((rows[:, 1] - rows[:, 0]).sum())

    def _gather_f32(
        self, x0: float, x1: float, y0: float, y1: float
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Window points as **f32** (the on-disk dtype) — one contiguous memmap
        slice per grid row, concatenated with no widening. The production
        deep-zoom path bins these directly with the f32 kernel (no f32→f64
        widening copy, which would otherwise dominate the query)."""
        ix0, ix1 = self._cell_range(x0, x1, self.x0, self.x1)
        iy0, iy1 = self._cell_range(y0, y1, self.y0, self.y1)
        g = self.g
        lon_parts, lat_parts = [], []
        for iy in range(iy0, iy1 + 1):
            lo = int(self.offsets[iy * g + ix0])
            hi = int(self.offsets[iy * g + ix1 + 1])
            if hi > lo:
                lon_parts.append(self.lon[lo:hi])
                lat_parts.append(self.lat[lo:hi])
        if not lon_parts:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty
        return np.concatenate(lon_parts), np.concatenate(lat_parts)
