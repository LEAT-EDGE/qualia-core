from __future__ import annotations

import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import qualia_core.utils.plugin
from qualia_core.typing import TYPE_CHECKING

from .ExperimentTracking import ExperimentTracking

if TYPE_CHECKING:
    from qualia_core.evaluation.Stats import Stats
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
        source TEXT,
        name TEXT,
        value REAL,

        UNIQUE(model_id, source, name, value),
        FOREIGN KEY(model_id) REFERENCES models(id)
    );
    CREATE TABLE IF NOT EXISTS quantization (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id INTEGER,
        bits INTEGER,
        epochs INTEGER,

        UNIQUE(model_id),
        FOREIGN KEY(model_id) REFERENCES models(id)
    );

    CREATE TABLE IF NOT EXISTS plugins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        schema_version INTEGER,

        UNIQUE(name)
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
        CREATE UNIQUE INDEX _ ON metrics(model_id, source, name, value);
        """,
        """
        ALTER TABLE metrics ADD COLUMN source TEXT;
        """,
        """
        CREATE TABLE IF NOT EXISTS quantization (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER,
            bits INTEGER,

            FOREIGN KEY(model_id) REFERENCES models(id)
        );
        """,
        """
        ALTER TABLE quantization ADD COLUMN epochs INTEGER;
        """,
        """
        CREATE UNIQUE INDEX _ ON quantization(model_id);
        """,
        """
        CREATE TABLE IF NOT EXISTS plugins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            schema_version INTEGER,

            UNIQUE(name)
        );
        """,
    ]

    __queries: Final[dict[str, str]] = {
        'get_schema_version': 'PRAGMA user_version',
        'insert_model': """INSERT INTO models(parent_id, timestamp, name, parameters, hash)
                           VALUES (:parent_id, :timestamp, :name, :parameters, :hash)""",
        'insert_metric': 'INSERT OR IGNORE INTO metrics(model_id, source, name, value) VALUES (:model_id, :source, :name, :value)',
        'insert_quantization': 'INSERT OR REPLACE INTO quantization(model_id, bits, epochs) VALUES(:model_id, :bits, :epochs)',
        'lookup_model_hash': 'SELECT id FROM models WHERE hash = :model_hash ORDER BY timestamp DESC',
        'lookup_model_name_and_hash': """SELECT id FROM models
                                         WHERE name = :model_name AND hash = :model_hash ORDER BY timestamp DESC""",
        'lookup_model': """SELECT id FROM models
                           WHERE parent_id IS :parent_id AND name = :name AND parameters = :parameters AND hash = :hash
                           ORDER BY timestamp DESC""",
        'get_models': 'SELECT * from models',
        'get_model': 'SELECT * from models WHERE id = :model_id',
        'get_metrics': 'SELECT * from metrics WHERE model_id = :model_id',
        'get_quantization': 'SELECT * from quantization WHERE model_id = :model_id',
        'get_plugins': 'SELECT * from plugins',
    }

    _con: sqlite3.Connection | None = None
    _cur: sqlite3.Cursor | None = None
    __ref_count: int = 0

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
        res = cur.execute(self.__queries['get_schema_version']).fetchone()
        return res[0] if res is not None else 0

    def _upgrade_database_schema(self, con: sqlite3.Connection, cur: sqlite3.Cursor) -> None:
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
        res = cur.execute(self.__queries['lookup_model_hash'], {'model_hash': model_hash}).fetchone()
        return res[0] if res is not None else None

    def __lookup_model_name_and_hash(self, cur: sqlite3.Cursor, model_name: str, model_hash: str) -> int | None:
        res = cur.execute(self.__queries['lookup_model_name_and_hash'],
                          {'model_name': model_name, 'model_hash': model_hash}).fetchone()
        return res[0] if res is not None else None

    def __get_models(self, cur: sqlite3.Cursor) -> list[dict[str, Any]]:
        return cur.execute(self.__queries['get_models']).fetchall()

    def __get_model(self, cur: sqlite3.Cursor, model_id: int) -> dict[str, Any] | None:
        res = cur.execute(self.__queries['get_model'], {'model_id': model_id}).fetchone()
        return res if res is not None else None

    def __get_metrics(self, cur: sqlite3.Cursor, model_id: int) -> list[dict[str, Any]]:
        return cur.execute(self.__queries['get_metrics'], {'model_id': model_id}).fetchall()

    def __get_quantization(self, cur: sqlite3.Cursor, model_id: int) -> dict[str, Any] | None:
        return cur.execute(self.__queries['get_quantization'], {'model_id': model_id}).fetchone()

    def __get_plugins(self, cur: sqlite3.Cursor) -> list[dict[str, Any]]:
        return cur.execute(self.__queries['get_plugins']).fetchall()

    @override
    def start(self, name: str | None = None) -> None:

        if self._con is not None:
            logger.warning('Database is already opened, incrementing reference count')
            self.__ref_count += 1
            return

        if not self.__db_path.exists():
            self.__create_database(self.__db_path)

        self._con = sqlite3.connect(self.__db_path, isolation_level=None)
        self.__ref_count += 1
        self._con.row_factory = sqlite3.Row
        self._cur = self._con.cursor()
        _ = self._cur.execute('PRAGMA foreign_keys = 1')

        logger.info('Opened database at %s', self.__db_path)

        self._upgrade_database_schema(self._con, self._cur)

    def log_trainresult(self, trainresult: TrainResult) -> int | None:
        from qualia_core.learningframework.PyTorch import PyTorch

        if not isinstance(trainresult.framework, PyTorch):
            logger.error('Only PyTorch LearningFramework is supported')
            return None

        if not self._con or not self._cur:
            logger.error('Database not initialized')
            return None

        parent_id = (self.__lookup_model_hash(self._cur, trainresult.parent_model_hash)
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
        model_id = self.__lookup_model(self._cur, values)

        if model_id is None:
            _ = self._cur.execute(self.__queries['insert_model'], values)
            self._con.commit()

            model_id = self._cur.lastrowid

        # Insert each metric record
        metrics = [
            {'model_id': model_id,
             'source': 'host',
             'name': name,
             'value': value,
             }
            for name, value in trainresult.metrics.items()
        ]

        _ = self._cur.executemany(self.__queries['insert_metric'], metrics)
        self._con.commit()

        return model_id

    def log_stats(self, model_name: str, model_hash: str, stats: Stats) -> None:
        if not self._con or not self._cur:
            logger.error('Database not initialized')
            return

        model_id = self.__lookup_model_name_and_hash(self._cur, model_name, model_hash)

        if model_id is None:
            logger.warning('Could not find model in database, target evaluation results will not be recorded (name=%s, hash=%s)',
                           model_name, model_hash)

        # Insert each metric record
        metrics = [
            {'model_id': model_id,
             'source': 'target',
             'name': name,
             'value': value,
             }
            for name, value in stats.metrics.items()
        ]

        # Also add the Stats fields avg_time, rom_size, ram_size
        metrics.extend({'model_id': model_id,
                            'source': 'target',
                            'name': name,
                            'value': getattr(stats, name),
                            } for name in ('avg_time', 'ram_size', 'rom_size'))

        _ = self._cur.executemany(self.__queries['insert_metric'], metrics)
        self._con.commit()

    def log_quantization(self, model_id: int, bits: int, epochs: int) -> None:
        if not self._con or not self._cur:
            logger.error('Database not initialized')
            return

        _ = self._cur.execute(self.__queries['insert_quantization'], {'model_id': model_id, 'bits': bits, 'epochs': epochs})
        self._con.commit()

    def get_plugins(self) -> list[dict[str, Any]]:
        if not self._con or not self._cur:
            logger.error('Database not initialized')
            return []

        return self.__get_plugins(self._cur)

    @override
    def _hyperparameters(self, params: RecursiveConfigDict) -> None:
        pass

    @override
    def stop(self) -> None:
        if self.__ref_count > 1:
            logger.info('Decrementing reference count')
            self.__ref_count -= 1
            return

        if self._con:
            self._con.close()
            logger.info('Database closed')

            self.__ref_count = 0

    def _print_models(self, models: list[dict[str, Any]]) -> None:
        if not models:
            print('No model in database')
            return

        pad_id = max(len(str(max(m['id'] for m in models))), len('ID'))
        pad_name = max(*(len(m['name']) for m in models), len('Name'))
        pad_hash = max(*(len(m['hash']) for m in models), len('Hash'))
        pad_date = max(len(str(datetime.fromtimestamp(0, tz=timezone.utc))), len('Date'))
        pad_parameters = max(len(str(max(m['parameters'] for m in models))), len('Parameters'))
        pad_parent_id = max(len(str(max(m['parent_id'] if m['parent_id'] is not None else 0 for m in models))), len('Parent'))

        header = f'{"ID": <{pad_id}} | '
        header += f'{"Name": <{pad_name}} | '
        header += f'{"Hash": <{pad_hash}} | '
        header += f'{"Date": <{pad_date}} | '
        header += f'{"Parameters": <{pad_parameters}} | '
        header += f'{"Parent": <{pad_parent_id}}'
        print(header)
        print('—' * len(header))

        for model in models:
            date = str(datetime.fromtimestamp(model["timestamp"], tz=timezone.utc))
            print(f'{model["id"]: <{pad_id}} | ', end='')
            print(f'{model["name"]: <{pad_name}} | ', end='')
            print(f'{model["hash"]: <{pad_hash}} | ', end='')
            print(f'{date: <{pad_date}} | ', end='')
            print(f'{model["parameters"]: <{pad_parameters}} | ', end='')
            print(f'{model["parent_id"] or "": <{pad_parent_id}}')


    def _print_model(self, model: dict[str, Any]) -> None:
        print(f'Model id:         {model["id"]}')
        print(f'Model name:       {model["name"]}')
        print(f'Model hash:       {model["hash"]}')
        print(f'Model date:       {datetime.fromtimestamp(model["timestamp"], tz=timezone.utc)}')
        print(f'Model parameters: {model["parameters"]}')
        print(f'Parent model id:  {model["parent_id"]}')

    def __print_quantization(self, quantization: dict[str, Any]) -> None:
        print('Quantization:')
        print(f'    Bits:   {quantization["bits"]}')
        print(f'    Epochs: {quantization["epochs"]}', end='')
        if quantization['epochs']:
            print(' (QAT)')
        else:
            print(' (PTQ)')

    def __print_metrics(self, metrics: list[dict[str, Any]]) -> None:
        max_name_length = 0
        metrics_by_source: dict[str, list[dict[str, Any]]] = {}
        for metric in metrics:
            metrics_by_source.setdefault(metric['source'], []).append(metric)
            max_name_length = max(max_name_length, len(metric['name']))

        print('Metrics:')
        for source_name, source in metrics_by_source.items():
            print(f'    Source: {source_name}')

            for metric in source:
                print(f'        {metric["name"]}: {" " * (max_name_length - len(metric["name"]))}{metric["value"]}')

    def __handle_list_command(self, subcommand: str, *args: str) -> None:
        if subcommand == 'models':
            self.__handle_list_model_command()
        else:
            logger.error('Invalid subcommand %s', subcommand)

    def __handle_list_model_command(self) -> None:
        if self._cur is None:
            logger.error('Database not initialized')
            return

        models = self.__get_models(self._cur)

        self._print_models(models)

    def __handle_show_model_command(self, *args: str) -> None:
        if len(args) < 1:
            logger.error('Model hash required')
            return

        if self._cur is None:
            logger.error('Database not initialized')
            return

        model_id = self.__lookup_model_hash(self._cur, args[0])
        if model_id is None:
            logger.error('Model hash %s not found', args[0])
            return

        while model_id is not None:
            model = self.__get_model(self._cur, model_id)
            if model is None:
                logger.error('Model %d not found', model_id)
                return

            self._print_model(model)

            quantization = self.__get_quantization(self._cur, model_id)
            if quantization:
                self.__print_quantization(quantization)

            metrics = self.__get_metrics(self._cur, model_id)
            self.__print_metrics(metrics)

            print()

            model_id = model['parent_id']
            if model_id is not None:
                print('Parent model')

    def __handle_show_command(self, subcommand: str, *args: str) -> None:
        if subcommand == 'model':
            self.__handle_show_model_command(*args)
        else:
            logger.error('Invalid subcommand %s', subcommand)

    def handle_command(self, command: str, *args: str) -> None:
        if command == 'list':
            if len(args) < 1:
                logger.error('Subcommand required')
                return

            self.__handle_list_command(*args)
        elif command == 'show':
            if len(args) < 1:
                logger.error('Subcommand required')
                return

            self.__handle_show_command(*args)
        elif command == 'help':
            self.print_cli_help()
        else:
            logger.error('Invalid command %s', command)
            self.print_cli_help()

    @classmethod
    def print_cli_help(cls) -> None:
        print('Usage: {sys.argv[0]} <command> <args>', file=sys.stderr)
        print('    command:')
        print('        - help')
        print('        - show')

    @classmethod
    def cli(cls) -> None:
        from qualia_core.utils.logger.setup_root_logger import setup_root_logger

        # We main not be called from qualia_core.main:main so always setup logging to show logger.info()
        setup_root_logger(colored=True)

        if len(sys.argv) < 2:
            cls.print_cli_help()
            return

        qualia_database = cls()

        qualia_database.start()

        # Instantiate QualiaDatabase from plugin if available
        plugins = qualia_database.get_plugins()
        if plugins:
            qualia = qualia_core.utils.plugin.load_plugins([plugin['name'] for plugin in plugins])
            if qualia.experimenttracking:
                logger.info('Reloading QualiaDatabase from plugin %s', qualia.experimenttracking.QualiaDatabase.__name__)
                qualia_database.stop()
                qualia_database: QualiaDatabase = qualia.experimenttracking.QualiaDatabase.QualiaDatabase()
                qualia_database.start()

        qualia_database.handle_command(*sys.argv[1:])

        qualia_database.stop()

    @property
    def __sql_schema_version(self) -> int:
        return len(self.__sql_schema_upgrades)

    @property
    def logger(self) -> None:
        return None
