from __future__ import annotations

import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Final

from qualia_core.typing import TYPE_CHECKING

from .ExperimentTracking import ExperimentTracking

if TYPE_CHECKING:
    from qualia_core.qualia import TrainResult
    from qualia_core.typing import RecursiveConfigDict

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class QualiaDatabase(ExperimentTracking):
    # Latest schema to create a fresh database
    __sql_schema: Final[str] = """
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_id INTEGER,
        timestamp INTEGER,
        name TEXT,
        parameters INTEGER,
        hash TEXT,

        FOREIGN KEY(parent_id) REFERENCES models(id)
    );

    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id INTEGER,
        name TEXT,
        value REAL,

        UNIQUE(model_id, name, value),
        FOREIGN KEY(model_id) REFERENCES models(id)
    );
    """

    # Incremental schema upgrades
    __sql_schema_upgrades: Final[list[str]] = [
        """
        ALTER TABLE models ADD COLUMN parameters INTEGER;
        """,
        """
        ALTER TABLE models ADD COLUMN hash TEXT;
        """,
        """
        ALTER TABLE models ADD COLUMN timestamp INTEGER;
        """,
        """
        ALTER TABLE models ADD COLUMN parent_id REFERENCES models(id);
        """,
        """
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER,
            name TEXT,
            value REAL,

            FOREIGN KEY(model_id) REFERENCES models(id)
        );
        """,
        """
        CREATE UNIQUE INDEX _ ON metrics(model_id, name, value);
        """,
    ]

    __queries: Final[dict[str, str]] = {
        'insert_model': """INSERT INTO models(parent_id, timestamp, name, parameters, hash)
                           VALUES (:parent_id, :timestamp, :name, :parameters, :hash)""",
        'insert_metric': 'INSERT OR IGNORE INTO metrics(model_id, name, value) VALUES (:model_id, :name, :value)',
        'lookup_model_hash': 'SELECT id FROM models WHERE hash = :model_hash ORDER BY timestamp DESC',
        'lookup_model': """SELECT id FROM models
                           WHERE parent_id IS :parent_id AND name = :name AND parameters = :parameters AND hash = :hash
                           ORDER BY timestamp DESC""",
    }

    __con: sqlite3.Connection | None = None
    __cur: sqlite3.Cursor | None = None

    def __init__(self, db_path: str | Path | None = None) -> None:
        super().__init__()
        self.__db_path = Path(db_path) if db_path is not None else Path('out') / 'qualia.db'

    def __create_database(self, path: Path) -> None:
        """Instantiate initial schema if database did not exist."""
        logger.info('Creating new database at %s', path)
        con = sqlite3.connect(path)
        cur = con.cursor()
        _ = cur.execute('PRAGMA foreign_keys = 1')
        _ = cur.executescript(self.__sql_schema)
        self.__set_schema_version(cur, self.__sql_schema_version)
        con.close()

    def __set_schema_version(self, cur: sqlite3.Cursor, version: int) -> None:
        _ = cur.execute(f'PRAGMA user_version = {version}')

    def __get_schema_version(self, cur: sqlite3.Cursor) -> int:
        res = cur.execute('PRAGMA user_version').fetchone()
        return res[0] if res is not None else 0

    def __upgrade_database_schema(self, con: sqlite3.Connection, cur: sqlite3.Cursor) -> None:
        # Perform schema upgrades if needed
        current_version = self.__get_schema_version(cur)
        latest_version = self.__sql_schema_version
        logger.info('Current database schema version: %d, latest schema version: %d', current_version, latest_version)
        for i, sql_schema_upgrade in enumerate(self.__sql_schema_upgrades[current_version:latest_version]):
            new_version = current_version + i + 1
            logger.info('Upgrading database schema to version %d', new_version)
            try:
                _ = cur.execute('BEGIN')  # Begin transaction to only update version number if schema upgrade succeeded
                _ = cur.execute(sql_schema_upgrade)
                self.__set_schema_version(cur, new_version)
                con.commit()
            except sqlite3.Error:
                con.rollback()
                logger.exception('Could not upgrade database schema to version %d', new_version)

    def __lookup_model(self, cur: sqlite3.Cursor, values: dict[str, Any]) -> int | None:
        res = cur.execute(self.__queries['lookup_model'], values).fetchone()
        return res[0] if res is not None else None

    def __lookup_model_hash(self, cur: sqlite3.Cursor, model_hash: str) -> int | None:
        res = cur.execute(self.__queries['lookup_model_hash'], (model_hash, )).fetchone()
        return res[0] if res is not None else None

    @override
    def start(self, name: str | None = None) -> None:
        if not self.__db_path.exists():
            self.__create_database(self.__db_path)

        self.__con = sqlite3.connect(self.__db_path, isolation_level=None)
        self.__cur = self.__con.cursor()
        _ = self.__cur.execute('PRAGMA foreign_keys = 1')

        logger.info('Opened database at %s', self.__db_path)

        self.__upgrade_database_schema(self.__con, self.__cur)

    def log_trainresult(self, trainresult: TrainResult) -> None:
        from qualia_core.learningframework.PyTorch import PyTorch

        if not isinstance(trainresult.framework, PyTorch):
            logger.error('Only PyTorch LearningFramework is supported')
            raise TypeError

        if not self.__con or not self.__cur:
            logger.error('Database not initialized')
            raise RuntimeError

        parent_id = (self.__lookup_model_hash(self.__cur, trainresult.parent_model_hash)
                     if trainresult.parent_model_hash is not None else None)

        # Insert model record
        values = {
            'parent_id': parent_id,
            'timestamp': time.time_ns() // (1000 * 1000 * 1000),  # Unix timestamp in seconds
            'name': trainresult.name,
            'parameters': trainresult.params,
            'hash': trainresult.model_hash,
        }

        # Avoid duplicate rows by looking up if the exact entry (excluding timestamp) already exsits in the database first
        model_id = self.__lookup_model(self.__cur, values)

        if model_id is None:
            _ = self.__cur.execute(self.__queries['insert_model'], values)
            self.__con.commit()

            model_id = self.__cur.lastrowid

        # Insert each metric record
        metrics = [
            {'model_id': model_id,
             'name': name,
             'value': value,
             }
            for name, value in trainresult.metrics.items()
        ]

        _ = self.__cur.executemany(self.__queries['insert_metric'], metrics)
        self.__con.commit()

    @override
    def _hyperparameters(self, params: RecursiveConfigDict) -> None:
        print(f'{params=}')

    @override
    def stop(self) -> None:
        if self.__con:
            self.__con.close()

    @property
    def __sql_schema_version(self) -> int:
        return len(self.__sql_schema_upgrades)
