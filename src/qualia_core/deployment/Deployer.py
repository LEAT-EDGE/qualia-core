from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from qualia_core.typing import TYPE_CHECKING
from qualia_core.utils.process import subprocesstee

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    from pathlib import Path  # noqa: TCH003

    from .Deploy import Deploy  # noqa: TCH001

logger = logging.getLogger(__name__)

class Deployer(ABC):
    @abstractmethod
    def deploy(self, tag: str) -> Deploy | None:
        ...

    def _sections_size(self, elffile: Path, size_cmd: str, section_labels: list[str]) -> int | None:
        args = ('-A', '-d', str(elffile))
        logger.info('Running: %s %s', size_cmd, ' '.join(args))
        returncode, outputs = subprocesstee.run(size_cmd, *args)
        if returncode != 0:
            return None
        outputs = outputs[1].decode().splitlines()
        logger.info([line.split() for line in outputs])
        sections = [line for line in outputs if line and any(seclabel == line.split()[0] for seclabel in section_labels)]
        sections_size = [int(line.split()[1]) for line in sections]

        return sum(sections_size)

    def _rom_size(self, elffile: Path, size_cmd: str) -> int | None:
        return self._sections_size(elffile,
                                   size_cmd,
                                   section_labels=['.isr_vector',
                                                   '.text',
                                                   '.rodata',
                                                   '.ARM',
                                                   '.preinit_array',
                                                   '.init_array',
                                                   '.fini_array',
                                                   '.data'])

    def _ram_size(self, elffile: Path, size_cmd: str) -> int | None:
        return self._sections_size(elffile,
                                   size_cmd,
                                   section_labels=['.bss', '.data'])
