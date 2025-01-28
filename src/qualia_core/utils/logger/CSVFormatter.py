from __future__ import annotations

import csv
import io
import logging
import sys

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class CSVFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output)

    @override
    def format(self, record: logging.LogRecord) -> str:
        self.writer.writerow(record.msg)
        data = self.output.getvalue()
        _ = self.output.truncate(0)
        _ = self.output.seek(0)
        return data.strip()
