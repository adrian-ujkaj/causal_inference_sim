"""
Ecriture CSV bufferisee.

Ouvrir et fermer un fichier a chaque ligne (open(..., "a") dans la boucle de
controle) coute cher : a 80 Hz et 4 drones, c'est plusieurs centaines
d'ouvertures de fichier par seconde simulee. On accumule donc les lignes en
memoire et on les ecrit par paquets.

Les lignes en attente sont ecrites a la fermeture explicite (close), et en
dernier recours a la sortie de l'interpreteur (atexit), pour ne rien perdre si
la simulation est interrompue.
"""

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
