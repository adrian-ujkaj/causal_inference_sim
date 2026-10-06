"""Ecriture CSV par paquets : ouvrir le fichier a chaque ligne coute cher dans la boucle
de controle. Les lignes en attente sont ecrites par close() ou a la sortie du programme."""

from __future__ import annotations

import atexit
import csv
import os
import weakref

_OPEN_BUFFERS: "weakref.WeakSet[CsvBuffer]" = weakref.WeakSet()


class CsvBuffer:
    def __init__(self, path: str, header: list[str], flush_every: int = 400):
        self.path = path
        self.flush_every = int(flush_every)
        self._rows: list[list] = []
        self.closed = False
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)
        _OPEN_BUFFERS.add(self)

    def write(self, row: list) -> None:
        if self.closed:
            return
        self._rows.append(row)
        if len(self._rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerows(self._rows)
        self._rows.clear()

    def close(self) -> None:
        if not self.closed:
            self.flush()
            self.closed = True


@atexit.register
def _flush_all() -> None:
    for buf in list(_OPEN_BUFFERS):
        try:
            buf.close()
        except Exception:
            pass
