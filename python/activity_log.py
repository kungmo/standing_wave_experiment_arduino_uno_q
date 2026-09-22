"""Research-grade SQLite persistence for the Arduino UNO Q standing-wave lab."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 3
MAX_PAYLOAD_BYTES = 64 * 1024
MAX_EVENTS_PER_BATCH = 25
MAX_ACTIVITY_VALUE_LENGTH = 4_000
MAX_CHAT_MESSAGE_LENGTH = 100_000
TERMINAL_STATUSES = {"completed", "stopped", "reset", "start_failed", "abandoned", "imported"}
# An experiment_id in one of these states must never receive new measurements.
CLOSED_STATUSES = {"completed", "reset", "imported", "abandoned"}


def local_now() -> str:
    """Return the computer's current local time with its UTC offset."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _text(value: Any, limit: int) -> str:
    value = "" if value is None else str(value)
    return value if len(value) <= limit else value[:limit]


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _id(value: Any, name: str) -> str:
    value = _text(value, 160).strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


class ActivityLog:
    """Thread-safe research store shared by browser and MCU synchronization."""

    def __init__(self, app_root: str | os.PathLike[str] | None = None) -> None:
        self.app_root = Path(app_root or Path(__file__).resolve().parent.parent).resolve()
        self.data_dir = self.app_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Keep the established filename used by the application and its documentation.
        self.db_path = self.data_dir / "standing_wave_activity.sqlite3"
        self._active_lock = threading.RLock()
        self._active_session_id: str | None = None
        self._active_experiment_id: str | None = None
        self._cycle_cache_lock = threading.Lock()
        self._cycle_cache: dict[str, set[int]] = {}
        self._initialize()
        self._restore_active_experiment()

    def _connect(self, path: Path | None = None) -> sqlite3.Connection:
        connection = sqlite3.connect(str(path or self.db_path), timeout=8.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 8000")
        connection.execute("PRAGMA foreign_keys = ON")
        # EXTRA is SQLite's most conservative rollback-journal setting: it also
        # synchronizes the directory after the journal is removed at commit.
        # Temporary query data stays in RAM instead of creating extra eMMC files.
        connection.execute("PRAGMA synchronous = EXTRA")
        connection.execute("PRAGMA temp_store = MEMORY")
        return connection

    @contextmanager
    def _db(self, path: Path | None = None):
        connection = self._connect(path)
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
        return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}

    def _initialize(self) -> None:
        with self._db() as connection:
            # Do not modify or partly upgrade an existing legacy database.  Local-time
            # schema v3 intentionally starts with a new DB so UTC and local timestamps
            # can never be mixed in columns whose names have different meanings.
            existing_tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            research_tables = {
                "activity_sessions", "activity_events", "experiments", "measurements",
                "experiment_transitions", "chat_messages", "analysis_results",
            }
            if existing_tables & research_tables:
                schema_version = None
                if "log_metadata" in existing_tables:
                    row = connection.execute(
                        "SELECT value FROM log_metadata WHERE key='schema_version'"
                    ).fetchone()
                    schema_version = str(row[0]) if row else None
                if schema_version != str(SCHEMA_VERSION):
                    raise RuntimeError(
                        "기존 연구 DB의 스키마가 현재 로컬 시간대 스키마 v3과 다릅니다. "
                        "자동 마이그레이션은 수행하지 않습니다. 앱을 종료한 뒤 "
                        "data/standing_wave_activity.sqlite3를 백업하고 다른 위치로 옮기거나 "
                        "삭제한 다음 다시 실행하십시오."
                    )

            # DELETE is SQLite's crash-safe rollback-journal mode. Unlike WAL it has no
            # persistent -wal/-shm companions: after each commit only the main DB remains.
            # A short-lived -journal file can exist during a write and is intentionally
            # retained as the recovery mechanism for process or power failure.
            journal_mode = str(connection.execute("PRAGMA journal_mode = DELETE").fetchone()[0]).lower()
            if journal_mode != "delete":
                raise RuntimeError(f"SQLite journal mode change failed: {journal_mode}")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS log_metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS activity_sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at_server_local TEXT NOT NULL,
                    started_at_client_local TEXT,
                    ended_at_server_local TEXT,
                    ended_at_client_local TEXT,
                    last_seen_server_local TEXT NOT NULL,
                    app_version TEXT, language TEXT, user_agent TEXT, screen_json TEXT
                );
                CREATE TABLE IF NOT EXISTS activity_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_uuid TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    experiment_id TEXT,
                    server_time_local TEXT NOT NULL,
                    client_time_local TEXT,
                    elapsed_ms INTEGER,
                    event_type TEXT NOT NULL,
                    target_id TEXT, target_label TEXT, value_text TEXT,
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES activity_sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    created_at_server_local TEXT NOT NULL,
                    started_at_server_local TEXT,
                    ended_at_server_local TEXT,
                    updated_at_server_local TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stop_reason TEXT,
                    last_cycle INTEGER NOT NULL DEFAULT 0,
                    frequency_hz REAL,
                    distance_per_cycle_cm REAL,
                    max_cycles INTEGER,
                    first_measure_at_start INTEGER,
                    tube_length_cm REAL,
                    tube_type TEXT,
                    temperature_c REAL,
                    data_mode TEXT,
                    start_state_json TEXT NOT NULL DEFAULT '{}',
                    end_state_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES activity_sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    cycle INTEGER NOT NULL,
                    measured_at_local TEXT,
                    saved_at_server_local TEXT NOT NULL,
                    estimated_position_cm REAL,
                    avg_peak_to_peak_v REAL NOT NULL,
                    std_peak_to_peak_v REAL NOT NULL,
                    clipped INTEGER NOT NULL,
                    boot_id INTEGER,
                    raw_json TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(experiment_id, cycle),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
                    FOREIGN KEY (session_id) REFERENCES activity_sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS experiment_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transition_uuid TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    server_time_local TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    state_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
                    FOREIGN KEY (session_id) REFERENCES activity_sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_uuid TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    experiment_id TEXT,
                    server_time_local TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    model TEXT,
                    context_points INTEGER,
                    FOREIGN KEY (session_id) REFERENCES activity_sessions(session_id),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                );
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_uuid TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    experiment_id TEXT,
                    server_time_local TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    ok INTEGER NOT NULL,
                    wavelength_cm REAL,
                    r2 REAL,
                    result_json TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES activity_sessions(session_id),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                );
                """
            )
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_activity_session_id ON activity_events(session_id, id);
                CREATE INDEX IF NOT EXISTS idx_activity_experiment_id ON activity_events(experiment_id, id);
                CREATE INDEX IF NOT EXISTS idx_activity_time ON activity_events(server_time_local);
                CREATE INDEX IF NOT EXISTS idx_activity_type ON activity_events(event_type);
                CREATE INDEX IF NOT EXISTS idx_experiments_session ON experiments(session_id, created_at_server_local);
                CREATE INDEX IF NOT EXISTS idx_measurements_keys ON measurements(session_id, experiment_id, cycle);
                CREATE INDEX IF NOT EXISTS idx_transition_keys ON experiment_transitions(session_id, experiment_id, id);
                CREATE INDEX IF NOT EXISTS idx_chat_keys ON chat_messages(session_id, experiment_id, id);

                DROP VIEW IF EXISTS activity_timeline;
                CREATE VIEW activity_timeline AS
                SELECT e.id, e.server_time_local, e.client_time_local, e.elapsed_ms,
                       e.session_id, e.experiment_id, e.event_type, e.target_id,
                       e.target_label, e.value_text, e.detail_json,
                       s.started_at_server_local AS session_started_at_server_local,
                       s.ended_at_server_local AS session_ended_at_server_local,
                       s.app_version, s.language
                FROM activity_events e JOIN activity_sessions s USING(session_id);

                DROP VIEW IF EXISTS research_measurements;
                CREATE VIEW research_measurements AS
                SELECT m.session_id, m.experiment_id, e.status AS experiment_status,
                       e.started_at_server_local AS experiment_started_at_server_local,
                       e.ended_at_server_local AS experiment_ended_at_server_local,
                       e.stop_reason, m.cycle, m.measured_at_local,
                       m.saved_at_server_local, m.estimated_position_cm,
                       m.avg_peak_to_peak_v, m.std_peak_to_peak_v, m.clipped,
                       m.boot_id, e.frequency_hz, e.distance_per_cycle_cm,
                       e.max_cycles, e.first_measure_at_start, e.tube_length_cm,
                       e.tube_type, e.temperature_c, e.data_mode
                FROM measurements m JOIN experiments e USING(experiment_id);

                DROP VIEW IF EXISTS chat_transcript;
                CREATE VIEW chat_transcript AS
                SELECT c.id, c.server_time_local, c.session_id, c.experiment_id,
                       c.role, c.message, c.model, c.context_points,
                       e.status AS experiment_status
                FROM chat_messages c LEFT JOIN experiments e USING(experiment_id);

                DROP VIEW IF EXISTS experiment_overview;
                CREATE VIEW experiment_overview AS
                SELECT e.*,
                       (SELECT COUNT(*) FROM measurements m WHERE m.experiment_id=e.experiment_id) AS measurement_count,
                       (SELECT COUNT(*) FROM experiment_transitions t WHERE t.experiment_id=e.experiment_id) AS transition_count,
                       (SELECT COUNT(*) FROM chat_messages c WHERE c.experiment_id=e.experiment_id) AS chat_message_count,
                       (SELECT COUNT(*) FROM activity_events a WHERE a.experiment_id=e.experiment_id) AS activity_event_count
                FROM experiments e;

                DROP VIEW IF EXISTS experiment_state_timeline;
                CREATE VIEW experiment_state_timeline AS
                SELECT id, server_time_local, session_id, experiment_id, status, reason, state_json
                FROM experiment_transitions;
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO log_metadata(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )
            current_local = datetime.now().astimezone()
            offset = current_local.strftime("%z")
            if len(offset) == 5:
                offset = f"{offset[:3]}:{offset[3:]}"
            connection.executemany(
                "INSERT OR REPLACE INTO log_metadata(key,value) VALUES(?,?)",
                (
                    ("local_timezone", current_local.tzname() or ""),
                    ("local_utc_offset", offset),
                ),
            )

    def _restore_active_experiment(self) -> None:
        with self._db() as connection:
            row = connection.execute(
                "SELECT session_id,experiment_id FROM experiments WHERE status='running' "
                "ORDER BY COALESCE(started_at_server_local,created_at_server_local) DESC LIMIT 1"
            ).fetchone()
        if row:
            with self._active_lock:
                self._active_session_id = str(row["session_id"])
                self._active_experiment_id = str(row["experiment_id"])

    @staticmethod
    def _config(config: dict[str, Any] | None) -> dict[str, Any]:
        config = config if isinstance(config, dict) else {}
        first = config.get("first_measure_at_start")
        return {
            "frequency_hz": _float(config.get("frequency_hz")),
            "distance_per_cycle_cm": _float(config.get("distance_per_cycle_cm")),
            "max_cycles": _int(config.get("max_cycles")),
            "first_measure_at_start": None if first is None else int(bool(first)),
            "tube_length_cm": _float(config.get("tube_length_cm")),
            "tube_type": _text(config.get("tube_type"), 80),
            "temperature_c": _float(config.get("temperature_c")),
            "data_mode": _text(config.get("data_mode"), 40),
        }

    def _ensure_session(self, db, session_id: str, now: str, client_time: str = "", info=None) -> None:
        info = info if isinstance(info, dict) else {}
        screen = json.dumps(
            {"screen": info.get("screen"), "viewport": info.get("viewport")},
            ensure_ascii=False, separators=(",", ":"),
        ) if info else ""
        db.execute(
            """INSERT INTO activity_sessions(
                   session_id,started_at_server_local,started_at_client_local,last_seen_server_local,
                   app_version,language,user_agent,screen_json)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(session_id) DO UPDATE SET
                   last_seen_server_local=excluded.last_seen_server_local,
                   app_version=CASE WHEN excluded.app_version<>'' THEN excluded.app_version ELSE activity_sessions.app_version END,
                   language=CASE WHEN excluded.language<>'' THEN excluded.language ELSE activity_sessions.language END,
                   user_agent=CASE WHEN excluded.user_agent<>'' THEN excluded.user_agent ELSE activity_sessions.user_agent END,
                   screen_json=CASE WHEN excluded.screen_json<>'' THEN excluded.screen_json ELSE activity_sessions.screen_json END""",
            (session_id, now, client_time, now, _text(info.get("app_version"),120),
             _text(info.get("language"),40), _text(info.get("user_agent"),1000), screen),
        )

    def _ensure_experiment(self, db, session_id: str, experiment_id: str, now: str, config=None, status="prepared") -> None:
        c = self._config(config)
        db.execute(
            """INSERT INTO experiments(
                   experiment_id,session_id,created_at_server_local,updated_at_server_local,status,
                   frequency_hz,distance_per_cycle_cm,max_cycles,first_measure_at_start,
                   tube_length_cm,tube_type,temperature_c,data_mode)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(experiment_id) DO UPDATE SET
                   updated_at_server_local=excluded.updated_at_server_local,
                   frequency_hz=COALESCE(excluded.frequency_hz,experiments.frequency_hz),
                   distance_per_cycle_cm=COALESCE(excluded.distance_per_cycle_cm,experiments.distance_per_cycle_cm),
                   max_cycles=COALESCE(excluded.max_cycles,experiments.max_cycles),
                   first_measure_at_start=COALESCE(excluded.first_measure_at_start,experiments.first_measure_at_start),
                   tube_length_cm=COALESCE(excluded.tube_length_cm,experiments.tube_length_cm),
                   tube_type=CASE WHEN excluded.tube_type<>'' THEN excluded.tube_type ELSE experiments.tube_type END,
                   temperature_c=COALESCE(excluded.temperature_c,experiments.temperature_c),
                   data_mode=CASE WHEN excluded.data_mode<>'' THEN excluded.data_mode ELSE experiments.data_mode END""",
            (experiment_id, session_id, now, now, status, c["frequency_hz"],
             c["distance_per_cycle_cm"], c["max_cycles"], c["first_measure_at_start"],
             c["tube_length_cm"], c["tube_type"], c["temperature_c"], c["data_mode"]),
        )

    def ingest_json(self, payload: str) -> dict[str, Any]:
        if len(payload.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise ValueError("payload is too large")
        try:
            events = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc.msg}") from exc
        if not isinstance(events, list) or not 1 <= len(events) <= MAX_EVENTS_PER_BATCH:
            raise ValueError(f"batch must contain 1..{MAX_EVENTS_PER_BATCH} events")
        events = [self._normalize_event(item) for item in events]
        inserted = duplicates = 0
        now = local_now()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                for e in events:
                    info = e["detail"] if e["event_type"] == "page_open" else {}
                    self._ensure_session(db, e["session_id"], now, e["client_time_local"], info)
                    cur = db.execute(
                        """INSERT OR IGNORE INTO activity_events(
                               event_uuid,session_id,experiment_id,server_time_local,client_time_local,
                               elapsed_ms,event_type,target_id,target_label,value_text,detail_json)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (e["event_uuid"],e["session_id"],e["experiment_id"],now,e["client_time_local"],
                         e["elapsed_ms"],e["event_type"],e["target_id"],e["target_label"],
                         e["value_text"],json.dumps(e["detail"],ensure_ascii=False,separators=(",",":"))),
                    )
                    inserted += int(cur.rowcount == 1)
                    duplicates += int(cur.rowcount != 1)
                    if e["event_type"] == "page_open":
                        db.execute("UPDATE activity_sessions SET ended_at_server_local=NULL,ended_at_client_local=NULL WHERE session_id=?",
                                   (e["session_id"],))
                    if e["event_type"] == "page_hide":
                        db.execute("UPDATE activity_sessions SET ended_at_server_local=?,ended_at_client_local=? WHERE session_id=?",
                                   (now,e["client_time_local"],e["session_id"]))
                db.commit()
            except Exception:
                db.rollback()
                raise
        return {"ok":True,"inserted":inserted,"duplicates":duplicates}

    def _normalize_event(self, item: Any) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError("each event must be an object")
        detail = item.get("detail") if isinstance(item.get("detail"), dict) else {}
        if len(json.dumps(detail,ensure_ascii=False).encode("utf-8")) > 16*1024:
            raise ValueError("detail is too large")
        return {
            "event_uuid":_id(item.get("event_uuid"),"event_uuid"),
            "session_id":_id(item.get("session_id"),"session_id"),
            "experiment_id":_text(item.get("experiment_id"),160).strip() or None,
            "client_time_local":_text(item.get("client_time_local"),80),
            "elapsed_ms":max(0,_int(item.get("elapsed_ms")) or 0),
            "event_type":_id(item.get("event_type"),"event_type")[:80],
            "target_id":_text(item.get("target_id"),160),
            "target_label":_text(item.get("target_label"),240),
            "value_text":_text(item.get("value_text"),MAX_ACTIVITY_VALUE_LENGTH),
            "detail":detail,
        }

    def prepare_experiment(self, session_id: str, experiment_id: str, config: dict[str, Any], status="prepared") -> None:
        session_id, experiment_id, now = _id(session_id,"session_id"), _id(experiment_id,"experiment_id"), local_now()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                self._ensure_session(db,session_id,now)
                self._ensure_experiment(db,session_id,experiment_id,now,config,status)
                db.commit()
            except Exception:
                db.rollback(); raise

    def _reuse_block_reason(self, db, experiment_id: str, state: dict[str, Any], require_fresh: bool = False) -> str | None:
        row = db.execute("SELECT status FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
        if row is None:
            return None
        if require_fresh and (row["status"] != "prepared" or db.execute(
            "SELECT 1 FROM measurements WHERE experiment_id=? LIMIT 1", (experiment_id,)
        ).fetchone()):
            return "fresh_experiment_required"
        if row["status"] in CLOSED_STATUSES:
            return f"experiment_already_{row['status']}"
        boot_id = _int(state.get("boot_id"))
        if boot_id is not None and db.execute(
            "SELECT 1 FROM measurements WHERE experiment_id=? AND boot_id IS NOT NULL AND boot_id<>? LIMIT 1",
            (experiment_id, boot_id),
        ).fetchone():
            return "experiment_from_previous_mcu_boot"
        return None

    def start_experiment(self, session_id: str, experiment_id: str, config: dict[str, Any], state=None,
                         require_fresh: bool = False) -> str:
        """Start or resume an experiment and return the experiment_id actually used.

        A closed experiment (completed/reset/imported/abandoned) or one measured under a
        different MCU boot is never reopened; a fresh UUID is issued instead so that old
        rows cannot be overwritten by new cycle numbers.
        """
        session_id, experiment_id, now = _id(session_id,"session_id"), _id(experiment_id,"experiment_id"), local_now()
        state = state if isinstance(state, dict) else {}
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                self._ensure_session(db,session_id,now)
                replaced_from, blocked = None, self._reuse_block_reason(db, experiment_id, state, require_fresh)
                if blocked:
                    replaced_from, experiment_id = experiment_id, str(uuid.uuid4())
                self._ensure_experiment(db,session_id,experiment_id,now,config,"running")
                db.execute("""UPDATE experiments SET status='running',
                              started_at_server_local=COALESCE(started_at_server_local,?),
                              ended_at_server_local=NULL,stop_reason=NULL,
                              start_state_json=CASE WHEN start_state_json='{}' THEN ? ELSE start_state_json END,
                              updated_at_server_local=?
                              WHERE experiment_id=?""",
                           (now,json.dumps(state,ensure_ascii=False),now,experiment_id))
                transition_state = dict(state)
                if replaced_from:
                    transition_state["replaced_experiment_id"] = replaced_from
                    transition_state["replacement_reason"] = blocked
                db.execute("""INSERT INTO experiment_transitions(
                              transition_uuid,session_id,experiment_id,server_time_local,status,reason,state_json)
                              VALUES(?,?,?,?,?,?,?)""",
                           (str(uuid.uuid4()),session_id,experiment_id,now,"running",
                            "measurement_start" if not replaced_from else "measurement_start_new_id",
                            json.dumps(transition_state,ensure_ascii=False,separators=(",",":"))))
                db.commit()
            except Exception:
                db.rollback(); raise
        with self._active_lock:
            self._active_session_id, self._active_experiment_id = session_id, experiment_id
        return experiment_id

    def active_ids(self) -> tuple[str | None, str | None]:
        with self._active_lock:
            return self._active_session_id, self._active_experiment_id

    def record_measurements_for_active(self, rows: Iterable[dict[str,Any]], config: dict[str,Any], state=None) -> int:
        session_id, experiment_id = self.active_ids()
        if not session_id or not experiment_id:
            return 0
        state = state if isinstance(state,dict) else {}
        rows = [dict(row) for row in rows if isinstance(row,dict)]
        dx = float(config.get("distance_per_cycle_cm",0.0) or 0.0)
        first = bool(config.get("first_measure_at_start",True))
        imported = str(config.get("data_mode","live")) == "imported"
        with self._cycle_cache_lock:
            cached = self._cycle_cache.get(experiment_id)
            cached = None if cached is None else set(cached)
        candidates = []
        for row in rows:
            cycle, avg, std = _int(row.get("rev")), _float(row.get("avg_v")), _float(row.get("std_v"))
            if cycle is None or cycle < 1 or avg is None or std is None:
                continue
            candidates.append((cycle, avg, std, row))
        saved = 0
        if cached is not None and all(c[0] in cached for c in candidates):
            # Nothing new: no connection, no file access, no flash write.
            stored = cached
        else:
            with self._db() as db:
                known = {int(r[0]) for r in db.execute("SELECT cycle FROM measurements WHERE experiment_id=?", (experiment_id,))}
                pending, seen = [], set(known)
                for cycle, avg, std, row in candidates:
                    if cycle in seen:
                        continue
                    seen.add(cycle)
                    pending.append((cycle, avg, std, row))
                if pending:
                    now = local_now()
                    db.execute("BEGIN IMMEDIATE")
                    try:
                        self._ensure_session(db,session_id,now)
                        self._ensure_experiment(db,session_id,experiment_id,now,config,"running")
                        for cycle, avg, std, row in pending:
                            x_cm = _float(row.get("x_cm")) if imported else ((cycle-(1 if first else 0))*dx if dx>0 else None)
                            # A completed cycle is immutable: keep the first saved row and its time.
                            cur = db.execute("""INSERT INTO measurements(
                                              experiment_id,session_id,cycle,measured_at_local,saved_at_server_local,
                                              estimated_position_cm,avg_peak_to_peak_v,std_peak_to_peak_v,clipped,boot_id,raw_json)
                                          VALUES(?,?,?,?,?,?,?,?,?,?,?)
                                          ON CONFLICT(experiment_id,cycle) DO NOTHING""",
                                       (experiment_id,session_id,cycle,_text(row.get("time"),80),now,x_cm,avg,std,
                                        int(bool(row.get("clipped"))),None if imported else _int(state.get("boot_id")),
                                        json.dumps(row,ensure_ascii=False,separators=(",",":"))))
                            saved += int(cur.rowcount == 1)
                        db.execute("UPDATE experiments SET last_cycle=MAX(last_cycle,(SELECT COALESCE(MAX(cycle),0) FROM measurements WHERE experiment_id=?)),"
                                   "updated_at_server_local=? WHERE experiment_id=?", (experiment_id,now,experiment_id))
                        db.commit()
                    except Exception:
                        db.rollback(); raise
                stored = {int(r[0]) for r in db.execute("SELECT cycle FROM measurements WHERE experiment_id=?", (experiment_id,))}
            with self._cycle_cache_lock:
                self._cycle_cache[experiment_id] = set(stored)
        stored_max = max(stored, default=0)
        max_cycles = _int(state.get("max_cycles")) or _int(config.get("max_cycles")) or 0
        last_rev = _int(state.get("last_result_rev")) or 0
        # Finish only after every completed MCU cycle is really in the DB. Otherwise the
        # final cycle, fetched later in the same sync pass, would be silently dropped.
        if (not imported and not bool(state.get("running")) and max_cycles > 0
                and last_rev >= max_cycles and stored_max >= last_rev):
            self.finish_experiment(session_id,experiment_id,"completed","maximum_cycle_reached",state)
        return saved

    def finish_experiment(self, session_id: str, experiment_id: str, status: str, reason: str, state=None) -> None:
        session_id, experiment_id = _id(session_id,"session_id"), _id(experiment_id,"experiment_id")
        if status not in TERMINAL_STATUSES:
            raise ValueError("invalid terminal experiment status")
        state, now = (state if isinstance(state,dict) else {}), local_now()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                current = db.execute("SELECT status FROM experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
                current_status = str(current["status"]) if current else ""
                self._ensure_session(db,session_id,now)
                self._ensure_experiment(db,session_id,experiment_id,now,{},status)
                # Pressing reset (or stop) after a run has already ended must not erase the
                # time, reason and MCU state describing how that run actually ended.
                already_ended = current_status in TERMINAL_STATUSES and status in {"reset", current_status}
                if already_ended:
                    db.execute("UPDATE experiments SET updated_at_server_local=? WHERE experiment_id=?", (now, experiment_id))
                else:
                    db.execute("""UPDATE experiments SET status=?,stop_reason=?,ended_at_server_local=?,
                                  updated_at_server_local=?,end_state_json=?
                                  WHERE experiment_id=?""",
                               (status,_text(reason,240),now,now,
                                json.dumps(state,ensure_ascii=False,separators=(",",":")),experiment_id))
                db.execute("""INSERT INTO experiment_transitions(
                              transition_uuid,session_id,experiment_id,server_time_local,status,reason,state_json)
                              VALUES(?,?,?,?,?,?,?)""",
                           (str(uuid.uuid4()),session_id,experiment_id,now,status,_text(reason,240),
                            json.dumps(state,ensure_ascii=False,separators=(",",":"))))
                db.commit()
            except Exception:
                db.rollback(); raise
        with self._active_lock:
            if self._active_experiment_id == experiment_id:
                self._active_session_id = self._active_experiment_id = None
        with self._cycle_cache_lock:
            self._cycle_cache.pop(experiment_id, None)

    def record_chat_exchange(self, session_id: str, experiment_id: str | None, question: str,
                             answer: str, model: str, context_points: int, config=None) -> None:
        self._record_chat(session_id,experiment_id,(("user",question),("assistant",answer)),model,context_points,config)

    def record_chat_error(self, session_id: str, experiment_id: str | None, question: str,
                          error: str, model: str, config=None) -> None:
        self._record_chat(session_id,experiment_id,(("user",question),("error",error)),model,None,config)

    def _record_chat(self, session_id, experiment_id, messages, model, context_points, config) -> None:
        session_id, experiment_id, now = _id(session_id,"session_id"), _text(experiment_id,160).strip() or None, local_now()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                self._ensure_session(db,session_id,now)
                if experiment_id:
                    self._ensure_experiment(db,session_id,experiment_id,now,config,"prepared")
                for role,message in messages:
                    db.execute("""INSERT INTO chat_messages(message_uuid,session_id,experiment_id,
                                  server_time_local,role,message,model,context_points) VALUES(?,?,?,?,?,?,?,?)""",
                               (str(uuid.uuid4()),session_id,experiment_id,now,role,
                                _text(message,MAX_CHAT_MESSAGE_LENGTH),_text(model,200),context_points))
                db.commit()
            except Exception:
                db.rollback(); raise

    def record_analysis(self, session_id: str, experiment_id: str | None, analysis_type: str,
                        result: dict[str,Any], config=None) -> None:
        session_id, experiment_id, now = _id(session_id,"session_id"), _text(experiment_id,160).strip() or None, local_now()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                self._ensure_session(db,session_id,now)
                if experiment_id:
                    self._ensure_experiment(db,session_id,experiment_id,now,config,"prepared")
                db.execute("""INSERT INTO analysis_results(analysis_uuid,session_id,experiment_id,
                              server_time_local,analysis_type,ok,wavelength_cm,r2,result_json)
                              VALUES(?,?,?,?,?,?,?,?,?)""",
                           (str(uuid.uuid4()),session_id,experiment_id,now,_text(analysis_type,80),
                            int(bool(result.get("ok"))),_float(result.get("wavelength_cm")),_float(result.get("r2")),
                            json.dumps(result,ensure_ascii=False,separators=(",",":"))))
                db.commit()
            except Exception:
                db.rollback(); raise

    def status(self) -> dict[str,Any]:
        tables = ("activity_events","activity_sessions","experiments","measurements","experiment_transitions","chat_messages","analysis_results")
        with self._db() as db:
            counts = {table:int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}
            journal_mode = str(db.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            synchronous = int(db.execute("PRAGMA synchronous").fetchone()[0])
        current_local = datetime.now().astimezone()
        offset = current_local.strftime("%z")
        if len(offset) == 5:
            offset = f"{offset[:3]}:{offset[3:]}"
        return {"ok":True,"total_events":counts["activity_events"],"total_sessions":counts["activity_sessions"],
                "total_experiments":counts["experiments"],"total_measurements":counts["measurements"],
                "total_transitions":counts["experiment_transitions"],
                "total_chat_messages":counts["chat_messages"],"total_analysis_results":counts["analysis_results"],
                "database":str(self.db_path.relative_to(self.app_root)),"schema_version":SCHEMA_VERSION,
                "local_timezone":current_local.tzname() or "","local_utc_offset":offset,
                "journal_mode":journal_mode,
                "synchronous":{0:"OFF",1:"NORMAL",2:"FULL",3:"EXTRA"}.get(synchronous,str(synchronous))}
