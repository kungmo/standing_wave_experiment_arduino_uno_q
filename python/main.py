# Open Standing Wave Lab v4.2 - temperature metadata + grounded experiment chatbot
import csv
import json
import os
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI
from arduino.app_bricks.cloud_llm import CloudLLM
from activity_log_api import activity_log, register_activity_log_routes


PHASE_NAMES = {
    0: "stopped",
    1: "moving",
    2: "settling",
    3: "measuring",
}

USERDATA_HOME = Path("/home/arduino")
DATA_DIR = (USERDATA_HOME if USERDATA_HOME.exists() and os.access(USERDATA_HOME, os.W_OK) else Path.home()) / ".uno_q_standing_wave"
DATA_JSON = DATA_DIR / "measurements.json"
DATA_CSV = DATA_DIR / "measurements.csv"
CONFIG_JSON = DATA_DIR / "config.json"
SESSION_JSON = DATA_DIR / "session.json"
ARCHIVE_DIR = DATA_DIR / "archive"

MAX_CYCLES_LIMIT = 1000

DEFAULT_CONFIG = {
    "frequency_hz": 0.0,
    "distance_per_cycle_cm": 0.0,
    "max_cycles": 110,
    "first_measure_at_start": True,
    "tube_length_cm": 0.0,
    "tube_type": "open",
    "temperature_c": None,
    "data_mode": "live",
}

bridge_lock = threading.RLock()
data_lock = threading.RLock()
sync_lock = threading.RLock()  # Also allows reset/stop to force a final synchronization safely.
max_cycles_lock = threading.RLock()  # Prevent config/background races while applying scan settings.
import_lock = threading.RLock()
measurements = []
config = dict(DEFAULT_CONFIG)
session = {"boot_id": None}
import_buffer = []
import_meta = None

# -----------------------------
# LLM chatbot (separate layer)
# -----------------------------
CHAT_MODEL_ID = "google:gemini-3.5-flash-lite"
CHAT_HISTORY_TURNS = 6
chat_lock = threading.Lock()
chat_sessions_lock = threading.RLock()
chat_sessions = {}
fit_cache_lock = threading.RLock()
fit_cache = None


def local_now(timespec="seconds"):
    """Return the computer's current local time with its UTC offset."""
    return datetime.now().astimezone().isoformat(timespec=timespec)


def clean_research_ids(research_session_id="", experiment_id=""):
    """Return bounded browser/experiment UUID strings, or (None, None) for an old UI."""
    research_session_id = str(research_session_id or "").strip()[:160]
    experiment_id = str(experiment_id or "").strip()[:160]
    if not research_session_id or not experiment_id:
        return None, None
    return research_session_id, experiment_id


def research_warning(operation, exc):
    # Research persistence must never stop the physical apparatus or hide its result.
    print(f"RESEARCH DB WARNING [{operation}]: {exc}")


def prepare_research_experiment(research_session_id, experiment_id, status="prepared"):
    ids = clean_research_ids(research_session_id, experiment_id)
    if ids[0]:
        try:
            activity_log.prepare_experiment(ids[0], ids[1], dict(config), status=status)
        except Exception as exc:
            research_warning("prepare_experiment", exc)


def finish_research_experiment(research_session_id, experiment_id, status, reason, state, require_active=False):
    # The experiment the MCU is actually running is the active one, no matter which
    # browser tab (possibly holding a stale experiment_id) pressed the button.
    ids = activity_log.active_ids()
    if not (ids[0] and ids[1]):
        if require_active:
            return
        ids = clean_research_ids(research_session_id, experiment_id)
    if ids[0] and ids[1]:
        try:
            activity_log.finish_experiment(ids[0], ids[1], status, reason, state)
        except Exception as exc:
            research_warning("finish_experiment", exc)


def start_research_experiment(research_session_id, experiment_id, state):
    """Return the experiment_id actually recorded (a new one if the old id was closed)."""
    ids = clean_research_ids(research_session_id, experiment_id)
    if not ids[0]:
        return None
    try:
        return activity_log.start_experiment(ids[0], ids[1], dict(config), state)
    except Exception as exc:
        research_warning("start_experiment", exc)
        return None


CHAT_SYSTEM_PROMPT = """
<시스템_지침>
<역할>정상파를 모르는 학생의 실험을 보조하는 챗봇</역할>
<장치의_개요>이 장치는 Arduino UNO Q, 스텝모터-실-도르래, 마이크 센서를 이용해 관 내부의 음향 정상파 압력 진폭 분포를 위치에 따라 자동 측정하여 실시간으로 그래프를 그려 주며, 학생의 질의에는 실험과 실시간 측정값을 근거로 답한다.</장치의_개요>
<사실의_우선순위>
- 매 요청의 '현재 실험 스냅샷'을 그 질문 시점의 최신 실험 사실로 취급하고 최근 대화보다 우선한다.
- 최근 대화에는 과거 질문과 답변만 있으며 과거의 실험 스냅샷은 포함되지 않는다. 수치적 사실은 항상 현재 실험 스냅샷을 기준으로 한다.
- 이 장치가 학습 효과나 학생의 성취도를 높였다고 실험 데이터 없이 단정하지 않는다.
- 파장, 마디 간격, R², 음속 등 Python이 결정론적으로 계산한 값이 있으면 '계산된 결과'로서 정확히 인용한다. 다만 계산값의 물리적 신뢰성은 측정 구간과 모드에 따라 별도로 평가하며, 특히 기본진동의 단일 lobe처럼 파장 식별이 약한 경우 계산값을 곧바로 참값으로 단정하지 않는다. 계산값과 사용자의 추측이 충돌하면 사용자의 전제를 그대로 따르지 말고 근거를 들어 설명한다.
- fitting 결과가 없거나 현재 데이터만으로 판단하기 어려운 것은 추측하지 말고 불확실하다고 명시한다.
- 특정 회차, 위치, Vpp, 표준편차, clipping 여부를 언급할 때는 반드시 현재 스냅샷의 실제 행을 확인한다. 존재하지 않는 회차나 값을 만들어내지 않는다.
</사실의_우선순위>
<측정_데이터와_clipping_또는_외란>
- clipping 여부의 유일한 기준은 현재 측정 데이터의 `clipped` 필드이다. Vpp가 크거나 약 3 V 부근이라는 이유만으로 clipping이라고 새로 판정하지 않는다.
- `clipped=1`인 점만 clipping 의심점이라고 부른다. 이 점들은 현재 Python fitting에서 제외된다.
- `clipped=0`인데 표준편차가 크거나 주변 추세에서 갑자기 벗어난 점은 주변 소음, 말소리, 기계적 외란 등으로 인한 이상점일 가능성을 언급할 수 있으나 원인을 단정하지 않는다.
- 현재 fitting은 clipping flag가 있는 점만 제외하고 나머지 점을 동일 가중의 제곱오차(SSE)로 맞춘다. 표준편차 가중, robust loss, 자동 이상점 제거를 사용한다고 말하지 않는다. 일부 이상점이 있어도 전체 주기 구조가 충분하면 전역 fitting 결과가 안정적일 수 있다고 설명한다.
- R²는 현재 모델이 관측된 진폭 형상을 얼마나 잘 설명하는지를 나타내는 지표이지, 파장 자체의 정확성이나 물리적 모수의 식별 가능성을 보증하는 값은 아니다.
- R²를 '정확도'라고 부르지 않는다. 학생에게는 '계산한 곡선이 측정값의 모양을 얼마나 잘 따라가는지 나타내는 값'이라고 설명한다.
</측정_데이터와_clipping_또는_외란>
<정상파_해석>
- 서로 이웃한 같은 종류의 지점 사이 거리, 즉 마디와 다음 마디 사이 또는 배와 다음 배 사이의 거리는 Δx = λ/2이다. 따라서 λ = 2Δx를 사용할 수 있다.
- 이론값과 측정값이 두 배 정도 차이가 난다면 학생이 마디와 배를 하나씩 선택한 거리인 λ/4를 잰 것일 수 있으므로, 이를 λ/2로 잘못 계산하지 않았는지 되묻는다.
- 진동수 f가 주어지면 v = fλ를 사용할 수 있다.
- 이 장치의 Vpp는 마이크 출력의 압력 진폭에 대응한다. 개관의 열린 끝은 이상적으로 압력 마디에 가깝고, 막힌 끝은 압력 배에 가깝다.
- 모드(기본진동, 2배진동, 3배진동 등)는 관 종류, 관 길이, 진동수, 실험실 온도, 실제 측정된 공간 패턴, fitting 결과를 함께 고려해 판단한다. 단순히 L/(λ/2)를 가장 가까운 정수로 반올림하는 것만으로 확정하지 않는다.
- 개관의 기본진동처럼 관 내부 측정 구간에 하나의 압력 진폭 lobe만 주로 보이고 양 끝의 실제 압력 마디가 끝단보정 때문에 관 바깥쪽에 위치할 수 있는 경우, 자유로운 sinusoidal fitting만으로 λ를 정밀하게 식별하기 어렵다. 이 경우 높은 R²만으로 λ와 음속이 정확하다고 단정하지 말고, 관 길이·진동수·온도·끝단보정 가능성을 함께 검토한다.
- 한쪽이 막힌 관의 기본진동도 관 내부에서 대략 1/4파장만 관측되므로, 경계조건과 끝단보정을 무시한 자유 fitting의 λ 추정에는 같은 종류의 식별 한계가 있을 수 있다.
- 반대로 관 내부에 둘 이상의 명확한 압력 마디 또는 반복 lobe가 나타나는 고차 모드에서는 공간 주기에서 λ를 직접 제약할 수 있으므로 파장 추정이 일반적으로 더 강건하다.
- 끝단보정의 크기를 정량적으로 계산하려면 관의 반지름/직경 등 추가 정보가 필요하다. 그 정보가 스냅샷에 없으면 보정량을 임의로 만들지 않는다.
- 대칭적인 개관에서 양 끝의 끝단보정이 비슷하다면 끝단보정 자체만으로 중앙 압력 배가 한쪽으로 이동한다고 단정하지 않는다. 중앙 위치의 이동은 시작점/위치 보정, 비대칭 경계조건, 외란 등 다른 가능성과 함께 논의한다.
- 공명 진동수가 아닌 진동수라고 해서 정상파가 생기지 않는다고 단정하지 않는다.
- 관의 다른 공명 진동수에서는 마디와 배의 개수가 다른 정상파 모드가 나타날 수 있다.
- 공명 진동수에서 많이 벗어나면 정상파가 약하거나 그래프의 반복 모양이 불분명할 수 있다고 설명한다.
- fitting을 사용하면 파장을 반드시 더 정확하게 구할 수 있다고 말하지 않는다. 측정점 전체를 이용해 파장을 추정하는 방법이라고 설명한다.
- Vpp의 pp는 peak-to-peak이며, 한 측정 구간에서 신호의 최댓값과 최솟값의 차이라고 설명한다.
- Vpp가 크면 이 장치에서 측정한 마이크 신호의 변화 폭이 크다는 뜻이다. 이를 실제 음압이나 사람이 느끼는 소리의 크기와 완전히 같다고 단정하지 않는다.
</정상파_해석>
<온도와_음속>
- 실험실 온도가 입력되어 있으면 20℃ 같은 임의의 상온값 대신 입력된 온도를 기준으로 이론 음속과 비교한다.
- 필요하면 건조 공기의 근사식 v ≈ 331.3 + 0.606T (m/s, T는 ℃)를 사용할 수 있으나, 습도·기압 등을 보정하지 않은 근사임을 필요할 때 밝힌다.
- 온도가 미입력이라면 임의의 온도를 가정하여 정확한 오차율을 제시하지 않는다.
</온도와_음속>
<실험_장치의_특성>
- 추정 위치는 실-도르래 보정값으로 얻은 값이므로 절대 위치라고 단정하지 않는다.
- 측정 데이터 초기화는 물리적 원점 복귀가 아니다. 새 실험 전에는 마이크를 START 표시선에 수동으로 맞추고 시작 위치를 확인한다.
</실험_장치의_특성>
<학습자_수준>
- 대상은 정상파를 처음 접하는 고등학교 1학년 학생이다.
- 학생은 파장 기호 λ, 마디, 배, 위상, fitting, R²의 의미를 아직 배우지 않았다.
- 학생의 지적 능력을 낮게 평가하거나 어린아이를 대하듯 말하지 않는다.
- 전문용어를 처음 사용할 때는 일상적인 표현을 먼저 제시하고, 그 뒤에 용어를 괄호로 알려 준다.
  예: "그래프에서 소리가 가장 약한 지점(마디)"
</학습자_수준>
<학생에게_보이는_응답_방식>
- 먼저 질문에 대한 답을 쉬운 말로 한두 문장 안에 제시한다.
- 현재 그래프나 측정값에서 직접 확인할 수 있는 근거를 한두 개만 보여 준다.
- 전문용어와 수식은 질문에 답하는 데 꼭 필요할 때만 사용한다.
- 수식을 사용할 때는 기호의 뜻을 먼저 설명하거나 바로 뒤에 설명한다.
- 한 답변에서 새로운 개념을 너무 많이 소개하지 않는다.
- 가능하면 마지막에 학생이 그래프에서 확인할 한 가지를 알려 준다.
</학생에게_보이는_응답_방식>
</시스템_지침>
""".strip()


def build_cloud_llm():
    # CloudLLM retrieves the provider credential from the App Lab Brick configuration.
    # The Arduino Cloud LLM Brick accepts a raw model identifier, so the requested
    # Gemini model can be pinned without constructing a separate Google client.
    # Intentionally DO NOT call .with_memory(): every chat request is self-contained.
    # This prevents the full experiment snapshot from being stored repeatedly in
    # CloudLLM's message history. We keep only trimmed raw Q/A text ourselves.
    return CloudLLM(
        model=CHAT_MODEL_ID,
        system_prompt=CHAT_SYSTEM_PROMPT,
        temperature=None,
        timeout=45,
    )


chat_llm = None
chat_llm_error = None
try:
    chat_llm = build_cloud_llm()
except Exception as exc:
    # AI is optional: a CloudLLM configuration error must not stop measurement/fitting.
    chat_llm_error = str(exc)
    print(f"CHATBOT INIT WARNING: {chat_llm_error}")


def normalize_measurements(rows):
    """Return one valid measurement per cycle, sorted by cycle.

    If an older version stored duplicate cycles, the last occurrence wins.
    """
    by_rev = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            rev = int(row.get("rev", 0))
        except (TypeError, ValueError):
            continue
        if rev < 1:
            continue
        clean = dict(row)
        clean["rev"] = rev
        by_rev[rev] = clean
    return [by_rev[rev] for rev in sorted(by_rev)]


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path, payload):
    ensure_data_dir()
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def load_persistent_state():
    global measurements, config, session
    ensure_data_dir()

    try:
        loaded = json.loads(DATA_JSON.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            measurements = normalize_measurements(loaded)
    except Exception:
        measurements = []

    try:
        loaded = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            # v3.13 and earlier always moved once before recording rev 1. Preserve the
            # coordinate mapping of an existing legacy dataset when upgrading. New/empty
            # experiments use the new default: rev 1 at the confirmed start position.
            had_first_mode = "first_measure_at_start" in loaded
            config.update(loaded)
            if not had_first_mode and measurements:
                config["first_measure_at_start"] = False
            if config.get("data_mode") not in ("live", "imported"):
                config["data_mode"] = "live"
    except Exception:
        pass

    try:
        loaded = json.loads(SESSION_JSON.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            session.update(loaded)
    except Exception:
        pass


def save_session():
    with data_lock:
        atomic_write_json(SESSION_JSON, session)


def normalize_temperature_c(value):
    """Return a validated optional laboratory temperature in degrees Celsius."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in ("none", "null", "nan"):
        return None
    temperature = float(text)
    if not math.isfinite(temperature) or temperature < -80.0 or temperature > 100.0:
        raise ValueError("실험실 온도는 -80~100 ℃ 범위로 입력하십시오.")
    return temperature


def archive_current_measurements(reason):
    global measurements
    with data_lock:
        if not measurements:
            return None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = ARCHIVE_DIR / f"measurements_{stamp}.json"
        payload = {
            "archived_at": local_now(),
            "reason": str(reason),
            "config": dict(config),
            "measurements": [dict(row) for row in measurements],
        }
        atomic_write_json(path, payload)
        measurements = []
    save_measurements()
    return path


def reconcile_mcu_session(state):
    """Keep the current graph tied to one MCU boot/session.

    Linux data intentionally survives browser refreshes and app restarts, while the MCU's
    RAM result buffer does not survive an MCU reboot. If those two lifetimes diverge, old
    cycle numbers can collide with a new scan (rev 1, 2, ...). Archive the old current
    dataset instead of silently treating new rev numbers as duplicates.
    """
    boot_id = int(state.get("boot_id", 0) or 0)
    if boot_id <= 0:
        return False

    previous_boot = session.get("boot_id")
    with data_lock:
        max_known_rev = max((int(row.get("rev", 0)) for row in measurements), default=0)
    mcu_rev = int(state.get("last_result_rev", 0) or 0)

    should_archive = False
    reason = None

    if previous_boot is None:
        # First run after upgrading from a version that had no session file. Keep data only
        # when it plausibly matches the MCU's current result buffer.
        if max_known_rev and max_known_rev != mcu_rev:
            should_archive = True
            reason = "session_tracking_enabled_mismatch"
    elif int(previous_boot) != boot_id:
        if max_known_rev:
            should_archive = True
            reason = "mcu_reboot"
    elif max_known_rev > mcu_rev:
        # Same Linux process/session record but MCU data were cleared or rolled back.
        should_archive = True
        reason = "mcu_result_counter_rollback"

    if should_archive:
        archive_current_measurements(reason)
        active_session_id, active_experiment_id = activity_log.active_ids()
        if active_session_id and active_experiment_id:
            try:
                activity_log.finish_experiment(active_session_id, active_experiment_id, "abandoned", reason, state)
            except Exception as exc:
                research_warning("abandon_experiment", exc)

    if previous_boot != boot_id:
        session["boot_id"] = boot_id
        save_session()

    return should_archive


def invalidate_fit_cache():
    global fit_cache
    with fit_cache_lock:
        fit_cache = None


def save_config():
    invalidate_fit_cache()
    with data_lock:
        atomic_write_json(CONFIG_JSON, config)


def save_measurements():
    invalidate_fit_cache()
    with data_lock:
        measurements[:] = normalize_measurements(measurements)
        atomic_write_json(DATA_JSON, measurements)
        ensure_data_dir()
        temp_path = DATA_CSV.with_suffix(".csv.tmp")
        with temp_path.open("w", newline="", encoding="utf-8-sig") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "time",
                    "cycle",
                    "estimated_position_cm",
                    "avg_peak_to_peak_v",
                    "std_peak_to_peak_v",
                    "clipped",
                    "frequency_hz",
                    "distance_per_cycle_cm",
                    "first_measure_at_start",
                    "max_cycles",
                    "tube_length_cm",
                    "tube_type",
                    "temperature_c",
                ]
            )
            dx = float(config.get("distance_per_cycle_cm", 0.0) or 0.0)
            frequency = float(config.get("frequency_hz", 0.0) or 0.0)
            first_at_start = configured_first_measure_at_start()
            max_cycles = configured_max_cycles()
            tube_length_cm = float(config.get("tube_length_cm", 0.0) or 0.0)
            tube_type = str(config.get("tube_type", "unknown") or "unknown")
            temperature_c = normalize_temperature_c(config.get("temperature_c"))
            imported = is_imported_mode()
            for item in measurements:
                if imported and item.get("x_cm") is not None:
                    x_cm = item.get("x_cm")
                elif dx > 0:
                    x_cm = (int(item["rev"]) - (1 if first_at_start else 0)) * dx
                else:
                    x_cm = ""
                writer.writerow(
                    [
                        item.get("time", ""),
                        item["rev"],
                        x_cm,
                        item["avg_v"],
                        item["std_v"],
                        int(bool(item["clipped"])),
                        frequency,
                        dx,
                        int(bool(first_at_start)),
                        max_cycles,
                        tube_length_cm,
                        tube_type,
                        "" if temperature_c is None else temperature_c,
                    ]
                )
        temp_path.replace(DATA_CSV)


def bridge_call(name, *args):
    with bridge_lock:
        return Bridge.call(name, *args)


def read_mcu_status():
    running = int(bridge_call("get_running"))
    phase = int(bridge_call("get_phase"))
    revnum = int(bridge_call("get_revnum"))
    step_index = int(bridge_call("get_step_index"))
    measure_index = int(bridge_call("get_measure_index"))
    data_seq = int(bridge_call("get_data_seq"))
    last_result_rev = int(bridge_call("get_last_result_rev"))
    last_avg_mv = int(bridge_call("get_last_avg_mv"))
    last_std_mv = int(bridge_call("get_last_std_mv"))
    last_clipped = int(bridge_call("get_last_clipped"))
    max_cycles = int(bridge_call("get_max_cycles"))
    first_measure_at_start = int(bridge_call("get_first_measure_at_start"))
    max_result_capacity = int(bridge_call("get_max_result_capacity"))
    position_ready = int(bridge_call("get_position_ready"))
    jog_direction = int(bridge_call("get_jog_direction"))
    boot_id = int(bridge_call("get_boot_id"))

    return {
        "running": bool(running),
        "phase": phase,
        "phase_name": PHASE_NAMES.get(phase, "unknown"),
        "revnum": revnum,
        "step_index": step_index,
        "measure_index": measure_index,
        "data_seq": data_seq,
        "last_result_rev": last_result_rev,
        "last_avg_mv": last_avg_mv,
        "last_avg_v": last_avg_mv / 1000.0,
        "last_std_mv": last_std_mv,
        "last_std_v": last_std_mv / 1000.0,
        "last_clipped": bool(last_clipped),
        "max_cycles": max_cycles,
        "first_measure_at_start": bool(first_measure_at_start),
        "max_result_capacity": max_result_capacity,
        "position_ready": bool(position_ready),
        "jog_direction": jog_direction,
        "boot_id": boot_id,
    }


def configured_max_cycles():
    try:
        value = int(config.get("max_cycles", 110) or 110)
    except (TypeError, ValueError):
        value = 110
    return max(1, min(MAX_CYCLES_LIMIT, value))


def configured_first_measure_at_start():
    return bool(config.get("first_measure_at_start", True))


def is_imported_mode():
    return str(config.get("data_mode", "live")) == "imported"


def sync_configured_max_cycles_to_mcu(state):
    """Apply persisted live-scan settings to the MCU while it is idle.

    Imported CSV data are intentionally analysis-only and must never rewrite the physical
    MCU scan state.
    """
    if is_imported_mode():
        return state

    with max_cycles_lock:
        desired_max = configured_max_cycles()
        desired_first = 1 if configured_first_measure_at_start() else 0
        current_max = int(state.get("max_cycles", 110) or 110)
        current_first = 1 if bool(state.get("first_measure_at_start", True)) else 0

        if current_max == desired_max and current_first == desired_first:
            return state
        if state.get("running") or int(state.get("phase", 0) or 0) != 0 or int(state.get("jog_direction", 0) or 0) != 0:
            return state
        if desired_max < int(state.get("last_result_rev", 0) or 0):
            return state

        updated = dict(state)
        if current_max != desired_max:
            updated["max_cycles"] = int(bridge_call("set_max_cycles", desired_max))

        # The first-point mode cannot be changed after results exist because it changes the
        # physical sequence and the rev->position mapping.
        if current_first != desired_first and int(state.get("last_result_rev", 0) or 0) == 0:
            updated["first_measure_at_start"] = bool(int(bridge_call("set_first_measure_at_start", desired_first)))
        return updated


def get_result_from_mcu(rev):
    avg_mv = int(bridge_call("get_result_avg_mv", int(rev)))
    std_mv = int(bridge_call("get_result_std_mv", int(rev)))
    clipped = bool(int(bridge_call("get_result_clipped", int(rev))))
    return {
        "time": local_now(),
        "rev": int(rev),
        "avg_v": avg_mv / 1000.0,
        "std_v": std_mv / 1000.0,
        "clipped": clipped,
    }


def sync_new_results(state=None):
    # api_status(), api_data(), and background_sync_loop() can all call this function.
    # Serialize the whole synchronization pass so two callers cannot both decide that
    # the same cycle is missing and append it twice.
    with sync_lock:
        if is_imported_mode():
            return False
        if state is None:
            state = read_mcu_status()

        # Backfill Linux-persisted points before session reconciliation can archive or
        # clear them after an MCU reboot. This closes the narrow crash window between
        # measurements.json being written and the research transaction completing.
        try:
            with data_lock:
                persisted_rows = [dict(row) for row in measurements]
                persisted_config = dict(config)
            if persisted_rows:
                activity_log.record_measurements_for_active(persisted_rows, persisted_config, state)
        except Exception as exc:
            research_warning("backfill_measurements", exc)

        reconcile_mcu_session(state)
        target_rev = int(state.get("last_result_rev", 0))
        if target_rev <= 0:
            return False

        changed = False
        for rev in range(1, target_rev + 1):
            with data_lock:
                already_present = any(int(row.get("rev", 0)) == rev for row in measurements)
            if already_present:
                continue

            item = get_result_from_mcu(rev)

            # Defensive second check: another operation (for example reset/session
            # reconciliation) may have changed the list while the Bridge call ran.
            with data_lock:
                if any(int(row.get("rev", 0)) == rev for row in measurements):
                    continue
                measurements.append(item)
                measurements[:] = normalize_measurements(measurements)
            changed = True

        if changed:
            save_measurements()

        # Upsert every completed point into the research DB on every pass. Repeating
        # this is intentional: if a transient DB error occurred, the next 0.5 s pass
        # repairs the missing rows, while UNIQUE(experiment_id, cycle) prevents copies.
        try:
            with data_lock:
                research_rows = [dict(row) for row in measurements]
                research_config = dict(config)
            activity_log.record_measurements_for_active(research_rows, research_config, state)
        except Exception as exc:
            research_warning("record_measurements", exc)
        return changed


def measurement_payload():
    dx = float(config.get("distance_per_cycle_cm", 0.0) or 0.0)
    first_at_start = configured_first_measure_at_start()
    imported = is_imported_mode()
    with data_lock:
        source = normalize_measurements(measurements)
        result = []
        for item in source:
            row = dict(item)
            if imported and row.get("x_cm") is not None:
                try:
                    row["x_cm"] = float(row["x_cm"])
                except (TypeError, ValueError):
                    row["x_cm"] = None
            else:
                if dx > 0:
                    offset = 1 if first_at_start else 0
                    row["x_cm"] = (int(row["rev"]) - offset) * dx
                else:
                    row["x_cm"] = None
            result.append(row)
        return result


def solve_linear_two_parameter(z_values, y_values):
    n = len(y_values)
    sum_z = sum(z_values)
    sum_y = sum(y_values)
    sum_zz = sum(z * z for z in z_values)
    sum_zy = sum(z * y for z, y in zip(z_values, y_values))
    det = n * sum_zz - sum_z * sum_z
    if abs(det) < 1e-12:
        return None
    baseline = (sum_y * sum_zz - sum_z * sum_zy) / det
    amplitude = (n * sum_zy - sum_z * sum_y) / det
    return baseline, amplitude


def fit_for_lambda_phase(x_values, y_values, wavelength_cm, phase):
    if wavelength_cm <= 0:
        return None
    k = 2.0 * math.pi / wavelength_cm
    z_values = [abs(math.sin(k * x + phase)) for x in x_values]
    solved = solve_linear_two_parameter(z_values, y_values)
    if solved is None:
        return None
    baseline, amplitude = solved
    if amplitude < 0:
        return None
    predicted = [baseline + amplitude * z for z in z_values]
    sse = sum((y - p) ** 2 for y, p in zip(y_values, predicted))
    return sse, baseline, amplitude, predicted


def fit_standing_wave():
    data = [row for row in measurement_payload() if row.get("x_cm") is not None and not row.get("clipped")]
    if len(data) < 8:
        return {"ok": False, "error": "보정된 비클리핑 데이터가 8개 이상 필요합니다."}

    x_values = [float(row["x_cm"]) for row in data]
    y_values = [float(row["avg_v"]) for row in data]
    x_span = max(x_values) - min(x_values)
    if x_span <= 0:
        return {"ok": False, "error": "위치 범위가 0입니다. 이동거리 보정값을 확인하십시오."}

    sorted_x = sorted(set(x_values))
    spacings = [b - a for a, b in zip(sorted_x, sorted_x[1:]) if b > a]
    dx = sorted(spacings)[len(spacings) // 2] if spacings else x_span / max(1, len(x_values) - 1)

    min_lambda = max(4.0 * dx, x_span / 25.0)
    max_lambda = max(min_lambda * 1.5, x_span * 2.5)

    best = None
    coarse_lambda_count = 320
    coarse_phase_count = 90
    lambda_step = (max_lambda - min_lambda) / max(1, coarse_lambda_count - 1)

    for li in range(coarse_lambda_count):
        wavelength = min_lambda + li * lambda_step
        for pi in range(coarse_phase_count):
            phase = math.pi * pi / coarse_phase_count
            result = fit_for_lambda_phase(x_values, y_values, wavelength, phase)
            if result is None:
                continue
            sse, baseline, amplitude, predicted = result
            if best is None or sse < best[0]:
                best = (sse, wavelength, phase, baseline, amplitude, predicted)

    if best is None:
        return {"ok": False, "error": "정상파 모델 fitting에 실패했습니다."}

    _, coarse_wavelength, coarse_phase, _, _, _ = best
    refine_lambda_half = max(lambda_step * 2.0, coarse_wavelength * 0.01)
    refine_phase_half = math.pi / coarse_phase_count * 2.0

    for li in range(121):
        wavelength = max(
            min_lambda,
            coarse_wavelength - refine_lambda_half + 2.0 * refine_lambda_half * li / 120.0,
        )
        for pi in range(81):
            phase = coarse_phase - refine_phase_half + 2.0 * refine_phase_half * pi / 80.0
            phase %= math.pi
            result = fit_for_lambda_phase(x_values, y_values, wavelength, phase)
            if result is None:
                continue
            sse, baseline, amplitude, predicted = result
            if sse < best[0]:
                best = (sse, wavelength, phase, baseline, amplitude, predicted)

    sse, wavelength, phase, baseline, amplitude, predicted = best
    mean_y = sum(y_values) / len(y_values)
    sst = sum((y - mean_y) ** 2 for y in y_values)
    r2 = 1.0 - sse / sst if sst > 1e-12 else None

    frequency = float(config.get("frequency_hz", 0.0) or 0.0)
    sound_speed = frequency * wavelength / 100.0 if frequency > 0 else None

    fit_points = [
        {"x_cm": x, "predicted_v": p}
        for x, p in zip(x_values, predicted)
    ]

    return {
        "ok": True,
        "wavelength_cm": wavelength,
        "node_spacing_cm": wavelength / 2.0,
        "baseline_v": baseline,
        "amplitude_v": amplitude,
        "phase_rad": phase,
        "r2": r2,
        "frequency_hz": frequency if frequency > 0 else None,
        "sound_speed_m_s": sound_speed,
        "used_points": len(data),
        "excluded_clipped_points": len(measurement_payload()) - len(data),
        "fit_points": fit_points,
        "model": "Vpp(x) = baseline + amplitude*|sin(2*pi*x/lambda + phase)|",
    }


def api_jog(direction: int = 0, detail: int = 1):
    """Low-speed manual motor jog for taking up string slack before an experiment.

    +1 follows the normal measurement direction, -1 reverses it, and 0 stops/releases.
    The MCU also has a 1.5 s watchdog, so the browser must refresh a non-zero command.
    Heartbeat requests use detail=0 to avoid repeatedly reading the full MCU status.
    """
    try:
        if is_imported_mode():
            return {"ok": False, "error": "CSV 불러오기 분석 모드에서는 JOG를 사용할 수 없습니다. 측정 데이터 초기화 후 새 실험을 준비하십시오."}
        direction = int(direction)
        detail = int(detail)
        if direction not in (-1, 0, 1):
            raise ValueError("JOG 방향은 -1, 0, 1 중 하나여야 합니다.")

        applied = int(bridge_call("set_jog_direction", direction))
        if direction != 0 and applied != direction:
            state = read_mcu_status() if detail else None
            return {
                "ok": False,
                "error": "JOG를 시작할 수 없습니다. 자동 측정이 정지되어 있고 측정 데이터가 비어 있는지 확인하십시오.",
                "state": state,
                "jog_direction": applied,
            }

        if detail:
            time.sleep(0.01)
            state = read_mcu_status()
            return {"ok": True, "state": state, "jog_direction": applied}
        return {"ok": True, "jog_direction": applied}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def api_start(research_session_id="", experiment_id=""):
    try:
        if is_imported_mode():
            state = read_mcu_status()
            return {"ok": False, "error": "CSV 불러오기 분석 모드입니다. 새 측정을 시작하려면 측정 데이터 초기화를 먼저 실행하십시오.", "state": state}
        state = sync_configured_max_cycles_to_mcu(read_mcu_status())
        if int(state.get("jog_direction", 0) or 0) != 0:
            return {"ok": False, "error": "장력 조절 모터를 먼저 정지한 뒤 측정을 시작하십시오.", "state": state}
        if int(state.get("max_cycles", 0)) != configured_max_cycles():
            return {"ok": False, "error": "설정한 마지막 회차를 MCU에 적용하지 못했습니다. 장치 상태를 확인하십시오.", "state": state}
        if bool(state.get("first_measure_at_start")) != configured_first_measure_at_start():
            return {"ok": False, "error": "설정한 첫 데이터 위치 방식을 MCU에 적용하지 못했습니다. 데이터가 비어 있는지 확인하십시오.", "state": state}

        used_experiment_id = start_research_experiment(research_session_id, experiment_id, state)
        if used_experiment_id:
            experiment_id = used_experiment_id
        bridge_call("set_run", 1)
        state = read_mcu_status()
        if state["running"]:
            return {"ok": True, "state": state, "research_experiment_id": used_experiment_id}
        if not state["position_ready"]:
            finish_research_experiment(research_session_id, experiment_id, "start_failed", "position_not_ready", state)
            return {
                "ok": False,
                "error": "시작 위치가 확인되지 않았거나 회차-위치 대응이 무효입니다. 데이터 초기화 후 마이크를 시작 표시선으로 수동 복귀시키고 '시작 위치 확인'을 누르십시오.",
                "state": state,
                "research_experiment_id": used_experiment_id,
            }
        if state["last_result_rev"] >= state["max_cycles"]:
            finish_research_experiment(research_session_id, experiment_id, "start_failed", "maximum_cycle_already_reached", state)
            return {
                "ok": False,
                "error": "최대 측정 횟수에 도달했습니다. 결과를 저장한 뒤 새 실험을 준비하십시오.",
                "state": state,
                "research_experiment_id": used_experiment_id,
            }
        finish_research_experiment(research_session_id, experiment_id, "start_failed", "mcu_did_not_start", state)
        return {"ok": False, "error": "MCU가 측정 상태로 전환되지 않았습니다.", "state": state, "research_experiment_id": used_experiment_id}
    except Exception as exc:
        ids = clean_research_ids(research_session_id, experiment_id)
        if ids[0] and activity_log.active_ids() == ids:
            finish_research_experiment(
                research_session_id, experiment_id, "start_failed",
                f"start_exception: {exc}", locals().get("state", {}),
            )
        return {"ok": False, "error": str(exc)}


def api_stop(research_session_id="", experiment_id=""):
    try:
        state_before_stop = read_mcu_status()
        bridge_call("set_jog_direction", 0)
        bridge_call("set_run", 0)
        time.sleep(0.05)
        state = read_mcu_status()
        sync_new_results(state)
        completed = int(state.get("last_result_rev", 0) or 0) >= int(state.get("max_cycles", 0) or 0) > 0
        end_state = dict(state)
        end_state["state_before_stop"] = state_before_stop
        finish_research_experiment(
            research_session_id,
            experiment_id,
            "completed" if completed else "stopped",
            "maximum_cycle_reached" if completed else "user_stop",
            end_state,
            require_active=True,
        )
        return {"ok": True, "state": state}
    except Exception as exc:
        if activity_log.active_ids()[1]:
            failure_state = dict(locals().get("state", {}) or {})
            failure_state["state_before_stop"] = locals().get("state_before_stop", {})
            finish_research_experiment(
                research_session_id, experiment_id, "stopped",
                f"stop_exception: {exc}", failure_state,
            )
        return {"ok": False, "error": str(exc)}


def api_confirm_start():
    try:
        if is_imported_mode():
            return {"ok": False, "error": "CSV 불러오기 분석 모드입니다. 새 실험을 준비하려면 측정 데이터 초기화를 먼저 실행하십시오.", "state": read_mcu_status()}
        before = read_mcu_status()
        if int(before.get("jog_direction", 0) or 0) != 0:
            return {"ok": False, "error": "장력 조절 모터를 정지한 뒤 시작 위치를 확인하십시오.", "state": before}
        accepted = int(bridge_call("confirm_start_position"))
        time.sleep(0.02)
        state = read_mcu_status()
        if not accepted:
            if state["last_result_rev"] > 0:
                error = "기존 측정 데이터가 남아 있어 새 시작 위치를 설정할 수 없습니다. 필요한 데이터를 먼저 저장한 뒤 '측정 데이터 초기화'를 실행하십시오."
            elif state["running"]:
                error = "측정 중에는 시작 위치를 설정할 수 없습니다."
            else:
                error = "시작 위치 확인 요청이 거부되었습니다. 장치 상태를 확인하십시오."
            return {"ok": False, "error": error, "state": state}
        return {"ok": True, "state": state}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def api_reset(research_session_id="", experiment_id=""):
    global measurements
    try:
        # Do not let the background synchronizer re-add an old MCU result while reset is
        # clearing both the MCU buffer and the Linux-side current dataset.
        with sync_lock:
            state_before_stop = read_mcu_status()
            bridge_call("set_jog_direction", 0)
            bridge_call("set_run", 0)
            time.sleep(0.05)
            before_reset = read_mcu_status()
            # Pull the last completed MCU result before reset_all erases its buffer.
            # An interrupted, unfinished cycle has no final Vpp, but its phase/step/
            # measure_index remain preserved in experiments.end_state_json.
            sync_new_results(before_reset)
            completed = int(before_reset.get("last_result_rev", 0) or 0) >= int(before_reset.get("max_cycles", 0) or 0) > 0
            end_state = dict(before_reset)
            end_state["state_before_stop"] = state_before_stop
            finish_research_experiment(
                research_session_id,
                experiment_id,
                "completed" if completed else "reset",
                "reset_after_completion" if completed else "user_reset",
                end_state,
            )
            bridge_call("reset_all")
            state = None
            for _ in range(30):
                time.sleep(0.05)
                state = read_mcu_status()
                if (
                    not state["running"]
                    and state["revnum"] == 0
                    and state["data_seq"] == 0
                    and state["last_result_rev"] == 0
                ):
                    break
            with data_lock:
                measurements = []
                config["data_mode"] = "live"
            save_config()
            save_measurements()
        return {"ok": True, "state": state, "config": dict(config)}
    except Exception as exc:
        if activity_log.active_ids()[1]:
            failure_state = dict(locals().get("before_reset", {}) or {})
            failure_state["state_before_stop"] = locals().get("state_before_stop", {})
            finish_research_experiment(
                research_session_id, experiment_id, "stopped",
                f"reset_exception: {exc}", failure_state,
            )
        return {"ok": False, "error": str(exc)}


def api_status():
    try:
        state = sync_configured_max_cycles_to_mcu(read_mcu_status())
        sync_new_results(state)
        return {"ok": True, "state": state, "data_mode": config.get("data_mode", "live")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def api_data():
    try:
        state = sync_configured_max_cycles_to_mcu(read_mcu_status())
        sync_new_results(state)
        return {
            "ok": True,
            "state": state,
            "config": dict(config),
            "measurements": measurement_payload(),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def api_config(frequency_hz: float = 0.0, distance_per_cycle_cm: float = 0.0, max_cycles: int = 110, first_measure_at_start: int = 1, tube_length_cm: float = 0.0, tube_type: str = "open", temperature_c="", research_session_id="", experiment_id=""):
    try:
        frequency_hz = max(0.0, float(frequency_hz))
        distance_per_cycle_cm = max(0.0, float(distance_per_cycle_cm))
        tube_length_cm = max(0.0, float(tube_length_cm))
        temperature_c = normalize_temperature_c(temperature_c)
        tube_type = str(tube_type or "unknown").strip().lower()
        if tube_type not in ("open", "one_end_closed", "unknown"):
            raise ValueError("관 종류는 open, one_end_closed, unknown 중 하나여야 합니다.")
        max_cycles = int(max_cycles)
        first_measure_at_start = 1 if int(first_measure_at_start) else 0
        if max_cycles < 1 or max_cycles > MAX_CYCLES_LIMIT:
            raise ValueError(f"마지막 회차는 1~{MAX_CYCLES_LIMIT} 사이의 정수여야 합니다.")

        # Imported data are analysis-only. Settings may still be edited to re-analyze the
        # loaded dataset, but must not alter the physical MCU state.
        if is_imported_mode():
            with data_lock:
                config["frequency_hz"] = frequency_hz
                config["distance_per_cycle_cm"] = distance_per_cycle_cm
                config["max_cycles"] = max_cycles
                config["first_measure_at_start"] = bool(first_measure_at_start)
                config["tube_length_cm"] = tube_length_cm
                config["tube_type"] = tube_type
                config["temperature_c"] = temperature_c
            save_config()
            save_measurements()
            prepare_research_experiment(research_session_id, experiment_id, status="imported")
            return {"ok": True, "config": dict(config), "measurements": measurement_payload(), "state": read_mcu_status()}

        with max_cycles_lock:
            state = read_mcu_status()
            if state["running"] or int(state.get("phase", 0)) != 0:
                raise ValueError("측정 중에는 실험 설정을 변경할 수 없습니다.")
            if int(state.get("jog_direction", 0) or 0) != 0:
                raise ValueError("장력 조절 모터를 정지한 뒤 실험 설정을 저장하십시오.")
            if max_cycles < int(state.get("last_result_rev", 0)):
                raise ValueError(f"이미 {state['last_result_rev']}회차까지 측정되어 마지막 회차를 그보다 작게 설정할 수 없습니다.")
            current_first = 1 if bool(state.get("first_measure_at_start", True)) else 0
            if int(state.get("last_result_rev", 0)) > 0 and first_measure_at_start != current_first:
                raise ValueError("측정 데이터가 있는 동안에는 첫 데이터 위치 방식을 변경할 수 없습니다. 새 실험에서 설정하십시오.")

            applied = int(bridge_call("set_max_cycles", max_cycles))
            if applied != max_cycles:
                raise RuntimeError(f"MCU가 마지막 회차 {max_cycles} 설정을 받아들이지 않았습니다. 현재 값: {applied}")
            applied_first = int(bridge_call("set_first_measure_at_start", first_measure_at_start))
            if applied_first != first_measure_at_start:
                raise RuntimeError("MCU가 첫 데이터 위치 설정을 받아들이지 않았습니다. 데이터가 비어 있는지 확인하십시오.")

            with data_lock:
                config["frequency_hz"] = frequency_hz
                config["distance_per_cycle_cm"] = distance_per_cycle_cm
                config["max_cycles"] = max_cycles
                config["first_measure_at_start"] = bool(first_measure_at_start)
                config["tube_length_cm"] = tube_length_cm
                config["tube_type"] = tube_type
                config["temperature_c"] = temperature_c
                config["data_mode"] = "live"
        save_config()
        save_measurements()
        prepare_research_experiment(research_session_id, experiment_id, status="prepared")
        state = read_mcu_status()
        return {"ok": True, "config": dict(config), "measurements": measurement_payload(), "state": state}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _clean_import_row(row):
    if not isinstance(row, dict):
        raise ValueError("CSV 행 형식이 올바르지 않습니다.")
    rev = int(row.get("rev", 0))
    if rev < 1 or rev > MAX_CYCLES_LIMIT:
        raise ValueError(f"CSV 회차는 1~{MAX_CYCLES_LIMIT} 범위여야 합니다.")
    avg_v = float(row.get("avg_v", 0.0))
    std_v = max(0.0, float(row.get("std_v", 0.0) or 0.0))
    clipped_raw = row.get("clipped", False)
    clipped = str(clipped_raw).strip().lower() in ("1", "true", "yes", "y", "의심") if not isinstance(clipped_raw, bool) else clipped_raw
    x_raw = row.get("x_cm")
    x_cm = None if x_raw in (None, "") else float(x_raw)
    return {
        "time": str(row.get("time", "") or ""),
        "rev": rev,
        "avg_v": avg_v,
        "std_v": std_v,
        "clipped": bool(clipped),
        "x_cm": x_cm,
    }


def api_import_begin(frequency_hz: float = 0.0, distance_per_cycle_cm: float = 0.0, max_cycles: int = 110, first_measure_at_start: int = 1, tube_length_cm: float = 0.0, tube_type: str = "unknown", temperature_c="", total_rows: int = 0, research_session_id="", experiment_id=""):
    global import_buffer, import_meta
    try:
        state = read_mcu_status()
        if state.get("running") or int(state.get("phase", 0) or 0) != 0 or int(state.get("jog_direction", 0) or 0) != 0:
            raise ValueError("측정 또는 JOG 동작 중에는 CSV를 불러올 수 없습니다.")
        total_rows = int(total_rows)
        if total_rows < 1 or total_rows > MAX_CYCLES_LIMIT:
            raise ValueError(f"CSV 측정점 수는 1~{MAX_CYCLES_LIMIT}개 범위여야 합니다.")
        max_cycles = max(int(max_cycles), total_rows)
        max_cycles = min(MAX_CYCLES_LIMIT, max_cycles)
        tube_type = str(tube_type or "unknown").strip().lower()
        if tube_type not in ("open", "one_end_closed", "unknown"):
            tube_type = "unknown"
        with import_lock:
            import_buffer = []
            import_meta = {
                "frequency_hz": max(0.0, float(frequency_hz)),
                "distance_per_cycle_cm": max(0.0, float(distance_per_cycle_cm)),
                "max_cycles": max_cycles,
                "first_measure_at_start": bool(int(first_measure_at_start)),
                "tube_length_cm": max(0.0, float(tube_length_cm)),
                "tube_type": tube_type,
                "temperature_c": normalize_temperature_c(temperature_c),
                "total_rows": total_rows,
                "research_session_id": str(research_session_id or "")[:160],
                "experiment_id": str(experiment_id or "")[:160],
            }
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def api_import_chunk(payload: str = ""):
    global import_buffer
    try:
        rows = json.loads(str(payload or "[]"))
        if not isinstance(rows, list):
            raise ValueError("CSV 전송 데이터가 배열이 아닙니다.")
        cleaned = [_clean_import_row(row) for row in rows]
        with import_lock:
            if import_meta is None:
                raise ValueError("CSV 불러오기가 시작되지 않았습니다.")
            import_buffer.extend(cleaned)
            if len(import_buffer) > MAX_CYCLES_LIMIT:
                raise ValueError("CSV 데이터가 최대 저장 개수를 초과했습니다.")
            received = len(import_buffer)
        return {"ok": True, "received": received}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def api_import_commit():
    global measurements, import_buffer, import_meta
    try:
        with import_lock:
            if import_meta is None:
                raise ValueError("CSV 불러오기가 시작되지 않았습니다.")
            meta = dict(import_meta)
            rows = normalize_measurements(import_buffer)
            if len(rows) != int(meta["total_rows"]):
                raise ValueError(f"CSV 행 수가 맞지 않습니다. 예상 {meta['total_rows']}개, 수신 {len(rows)}개")
            import_buffer = []
            import_meta = None

        with data_lock:
            measurements = rows
            config["frequency_hz"] = meta["frequency_hz"]
            config["distance_per_cycle_cm"] = meta["distance_per_cycle_cm"]
            config["max_cycles"] = max(int(meta["max_cycles"]), max((int(r["rev"]) for r in rows), default=1))
            config["first_measure_at_start"] = bool(meta["first_measure_at_start"])
            config["tube_length_cm"] = float(meta.get("tube_length_cm", 0.0) or 0.0)
            config["tube_type"] = str(meta.get("tube_type", "unknown") or "unknown")
            config["temperature_c"] = normalize_temperature_c(meta.get("temperature_c"))
            config["data_mode"] = "imported"
        save_config()
        save_measurements()
        research_ids = clean_research_ids(meta.get("research_session_id"), meta.get("experiment_id"))
        imported_experiment_id = None
        if research_ids[0]:
            try:
                imported_state = read_mcu_status()
                imported_experiment_id = activity_log.start_experiment(
                    research_ids[0], research_ids[1], dict(config), imported_state, require_fresh=True)
                activity_log.record_measurements_for_active(rows, dict(config), imported_state)
                activity_log.finish_experiment(research_ids[0], imported_experiment_id, "imported", "experiment_bundle_import", imported_state)
            except Exception as exc:
                research_warning("record_imported_experiment", exc)
        return {
            "ok": True,
            "research_experiment_id": imported_experiment_id,
            "config": dict(config),
            "measurements": measurement_payload(),
            "state": read_mcu_status(),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def api_import_cancel():
    global import_buffer, import_meta
    with import_lock:
        import_buffer = []
        import_meta = None
    return {"ok": True}


def api_fit(research_session_id="", experiment_id=""):
    global fit_cache
    try:
        result = fit_standing_wave()
        with fit_cache_lock:
            fit_cache = dict(result) if isinstance(result, dict) else None
        ids = clean_research_ids(research_session_id, experiment_id)
        if ids[0]:
            try:
                activity_log.record_analysis(ids[0], ids[1], "curve_fit", result, dict(config))
            except Exception as exc:
                research_warning("record_analysis", exc)
        return result
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
        ids = clean_research_ids(research_session_id, experiment_id)
        if ids[0]:
            try:
                activity_log.record_analysis(ids[0], ids[1], "curve_fit", result, dict(config))
            except Exception as log_exc:
                research_warning("record_analysis_error", log_exc)
        return result


def compact_measurement_context():
    """Create a compact, lossless text snapshot of all current measurement points.

    The scan end point is user-configurable (up to 1000 stored points). Sending the current
    points as compact CSV-like rows avoids retrieval ambiguity and uses far fewer tokens than
    repeating verbose labels on every measurement.
    """
    rows = measurement_payload()
    lines = ["cycle,x_cm,Vpp_mean_V,Vpp_sd_V,clipped"]
    for row in rows:
        rev = int(row.get("rev", 0))
        x = row.get("x_cm")
        avg = float(row.get("avg_v", 0.0) or 0.0)
        std = float(row.get("std_v", 0.0) or 0.0)
        clipped = 1 if bool(row.get("clipped")) else 0
        x_text = "" if x is None else f"{float(x):.3f}"
        lines.append(f"{rev},{x_text},{avg:.4f},{std:.4f},{clipped}")
    return rows, lines if rows else []


def get_fit_for_chat():
    global fit_cache
    with fit_cache_lock:
        cached = dict(fit_cache) if isinstance(fit_cache, dict) else None
    if cached is not None:
        return cached

    # Compute once on demand only when the deterministic fitter has enough calibrated data.
    rows = measurement_payload()
    dx = float(config.get("distance_per_cycle_cm", 0.0) or 0.0)
    usable = [row for row in rows if not row.get("clipped")]
    if dx <= 0 or len(usable) < 8:
        return {"ok": False, "error": "아직 fitting을 위한 보정된 비클리핑 데이터가 충분하지 않습니다."}

    try:
        result = fit_standing_wave()
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    with fit_cache_lock:
        fit_cache = dict(result) if isinstance(result, dict) else None
    return result


def _xml_text(value):
    """Return XML-safe element text while discarding XML 1.0 control characters."""
    text = "" if value is None else str(value)
    text = "".join(
        char for char in text
        if char in "\t\n\r"
        or 0x20 <= ord(char) <= 0xD7FF
        or 0xE000 <= ord(char) <= 0xFFFD
        or 0x10000 <= ord(char) <= 0x10FFFF
    )
    return xml_escape(text, {'"': "&quot;", "'": "&apos;"})


def build_experiment_context():
    rows, data_lines = compact_measurement_context()
    frequency = float(config.get("frequency_hz", 0.0) or 0.0)
    dx = float(config.get("distance_per_cycle_cm", 0.0) or 0.0)
    max_cycles = configured_max_cycles()
    first_at_start = configured_first_measure_at_start()
    tube_length_cm = float(config.get("tube_length_cm", 0.0) or 0.0)
    tube_type = str(config.get("tube_type", "unknown") or "unknown")
    temperature_c = normalize_temperature_c(config.get("temperature_c"))
    tube_type_ko = {"open": "개관(양쪽 열림)", "one_end_closed": "한쪽이 막힌 관", "unknown": "미지정"}.get(tube_type, "미지정")
    clipped_count = sum(1 for row in rows if row.get("clipped"))
    positions = [float(row["x_cm"]) for row in rows if row.get("x_cm") is not None]
    fit = get_fit_for_chat()

    context_lines = [
        '<현재_실험_스냅샷 최신성="현재_요청_시점" 제공_횟수="이_요청에서_1회">',
        "  <장치_정보>",
        "    <장치>Arduino UNO Q 기반 음향 정상파 자동 스캔 장치</장치>",
        f"    <관_종류>{_xml_text(tube_type_ko)}</관_종류>",
        (f'    <관_길이 단위="cm">{tube_length_cm:.3f}</관_길이>'
         if tube_length_cm > 0 else '    <관_길이 상태="미입력" />'),
        (f'    <실험실_온도 단위="℃">{temperature_c:.2f}</실험실_온도>'
         if temperature_c is not None else '    <실험실_온도 상태="미입력" />'),
        "    <이동_기구>스텝모터와 실/도르래로 마이크를 한 방향으로 이동하며 새 실험 전에는 수동으로 복귀</이동_기구>",
        "  </장치_정보>",
        "  <측정_설정>",
        "    <빠른_스캔>모터 20 RPM, 이동 후 안정화 0.5 s, 200 ms Vpp window 10회, 최대값과 최소값을 1개씩 제외한 8개의 평균과 표준편차</빠른_스캔>",
        (f'    <설정_주파수 단위="Hz">{frequency:.3f}</설정_주파수>'
         if frequency > 0 else '    <설정_주파수 상태="미입력" />'),
        (f'    <회차당_추정_이동거리 단위="cm">{dx:.4f}</회차당_추정_이동거리>'
         if dx > 0 else '    <회차당_추정_이동거리 상태="미보정">그래프 x축은 회차</회차당_추정_이동거리>'),
        (f'    <측정_위치_범위 단위="cm"><최솟값>{min(positions):.3f}</최솟값><최댓값>{max(positions):.3f}</최댓값></측정_위치_범위>'
         if positions else '    <측정_위치_범위 상태="측정값_없음" />'),
        f"    <마지막_회차>{max_cycles}</마지막_회차>",
        f"    <첫_데이터_위치>{'확인된 시작 위치(1회차 = 0 cm)' if first_at_start else '1회 이동 후'}</첫_데이터_위치>",
        f"    <데이터_출처>{'불러온 실험 데이터' if is_imported_mode() else '현재 MCU 실시간 측정 데이터'}</데이터_출처>",
        f"    <저장된_측정점_수>{len(rows)}</저장된_측정점_수>",
        f"    <clipping_의심점_수>{clipped_count}</clipping_의심점_수>",
        "  </측정_설정>",
        "  <fitting_방법>",
        "    <제외_기준>clipped=1인 점만 제외</제외_기준>",
        "    <오차_함수>나머지 비클리핑 점 전체에 동일 가중치를 적용한 제곱오차합(SSE)</오차_함수>",
        "    <사용하지_않는_방법>표준편차 가중, robust loss, 자동 이상점 제거</사용하지_않는_방법>",
        "  </fitting_방법>",
    ]

    if isinstance(fit, dict) and fit.get("ok"):
        context_lines.extend(
            [
                '  <fitting_결과 상태="성공">',
                f'    <파장 단위="cm">{float(fit["wavelength_cm"]):.4f}</파장>',
                f'    <마디_간격 단위="cm">{float(fit["node_spacing_cm"]):.4f}</마디_간격>',
                (f'    <R_제곱>{float(fit["r2"]):.5f}</R_제곱>'
                 if fit.get("r2") is not None else '    <R_제곱 상태="계산_불가" />'),
                f'    <baseline 단위="V">{float(fit["baseline_v"]):.4f}</baseline>',
                f'    <amplitude 단위="V">{float(fit["amplitude_v"]):.4f}</amplitude>',
                f'    <사용점_수>{int(fit["used_points"])}</사용점_수>',
                f'    <제외된_clipping_점_수>{int(fit["excluded_clipped_points"])}</제외된_clipping_점_수>',
            ]
        )
        if fit.get("sound_speed_m_s") is not None:
            context_lines.append(f'    <계산된_음속 단위="m/s">{float(fit["sound_speed_m_s"]):.3f}</계산된_음속>')
        context_lines.extend(
            [
                f"    <모델>{_xml_text(fit.get('model', ''))}</모델>",
                "  </fitting_결과>",
            ]
        )
    else:
        fit_error = fit.get("error", "미실행") if isinstance(fit, dict) else "미실행"
        context_lines.append(f'  <fitting_결과 상태="없음"><사유>{_xml_text(fit_error)}</사유></fitting_결과>')

    if data_lines:
        context_lines.extend(
            [
                '  <측정_데이터 형식="CSV" 제공_횟수="이_요청에서_1회">',
                _xml_text("\n".join(data_lines)),
                "  </측정_데이터>",
            ]
        )
    else:
        context_lines.append('  <측정_데이터 상태="없음" />')

    context_lines.append("</현재_실험_스냅샷>")

    return "\n".join(context_lines)


def get_chat_history_text(session_id):
    with chat_sessions_lock:
        history = list(chat_sessions.get(session_id, []))
    if not history:
        return f'<최근_대화 상태="없음" 최대_턴="{CHAT_HISTORY_TURNS}" />'
    parts = [f'<최근_대화 최대_턴="{CHAT_HISTORY_TURNS}" 용도="질문_의도와_연속성_파악">']
    for index, item in enumerate(history[-CHAT_HISTORY_TURNS * 2 :], start=1):
        role = "학생" if item.get("role") == "user" else "챗봇"
        parts.append(f'  <메시지 순서="{index}" 역할="{role}">{_xml_text(item.get("content", ""))}</메시지>')
    parts.append("</최근_대화>")
    return "\n".join(parts)


def append_chat_history(session_id, role, content):
    # Keep the original text in memory. XML escaping happens only when the next
    # request serializes the history, so stored Q/A and the research log stay raw.
    with chat_sessions_lock:
        history = chat_sessions.setdefault(session_id, [])
        history.append({"role": role, "content": str(content)})
        max_items = CHAT_HISTORY_TURNS * 2
        if len(history) > max_items:
            del history[:-max_items]


def api_chat(question="", session_id="default", research_session_id="", experiment_id=""):
    global chat_llm, chat_llm_error
    question = str(question or "").strip()
    session_id = str(session_id or "default").strip()[:80] or "default"
    if not question:
        return {"ok": False, "error": "질문을 입력하십시오."}
    if len(question) > 2000:
        return {"ok": False, "error": "질문이 너무 깁니다. 2000자 이내로 입력하십시오."}

    if chat_llm is None:
        # Retry in case Brick configuration became available after application startup.
        try:
            chat_llm = build_cloud_llm()
            chat_llm_error = None
        except Exception as exc:
            chat_llm_error = str(exc)
            return {"ok": False, "error": chat_llm_error, "model": CHAT_MODEL_ID}

    try:
        # IMPORTANT CONTEXT DESIGN:
        # 1) build a fresh experiment snapshot exactly once for this request;
        # 2) history contains ONLY prior raw questions/answers, never old snapshots;
        # 3) CloudLLM built-in memory is disabled, so this assembled prompt is not
        #    accumulated inside the Brick across turns.
        experiment_context = build_experiment_context()
        history_text = get_chat_history_text(session_id)
        prompt = f"""<챗봇_요청>
{experiment_context}

{history_text}

<학생_질문>{_xml_text(question)}</학생_질문>

<과제>
  <사실_기준>현재 실험 스냅샷을 수치적 사실의 기준으로 사용한다.</사실_기준>
  <최근_대화_사용_범위>질문의 의도와 연속성을 이해하는 데만 사용한다.</최근_대화_사용_범위>
  <응답_기준>시스템 프롬프트의 <학습자_수준>과 <학생에게_보이는_응답_방식>을 적용하여 답한다.</응답_기준>
</과제>
</챗봇_요청>"""

        with chat_lock:
            answer = chat_llm.chat(prompt)
        answer = str(answer or "").strip()
        if not answer:
            raise RuntimeError("LLM이 빈 응답을 반환했습니다.")

        # Store ONLY Q/A text. The experiment snapshot/prompt is deliberately not stored.
        append_chat_history(session_id, "user", question)
        append_chat_history(session_id, "assistant", answer)
        context_points = len(measurement_payload())
        research_ids = clean_research_ids(research_session_id, experiment_id)
        if research_ids[0]:
            try:
                activity_log.record_chat_exchange(
                    research_ids[0], research_ids[1], question, answer,
                    CHAT_MODEL_ID, context_points, dict(config),
                )
            except Exception as exc:
                research_warning("record_chat_exchange", exc)
        return {
            "ok": True,
            "answer": answer,
            "model": CHAT_MODEL_ID,
            "context_points": context_points,
            "history_turn_limit": CHAT_HISTORY_TURNS,
            "context_mode": "fresh_snapshot_once_per_request",
        }
    except Exception as exc:
        print(f"CHATBOT FAILED: {exc}")
        research_ids = clean_research_ids(research_session_id, experiment_id)
        if research_ids[0]:
            try:
                activity_log.record_chat_error(
                    research_ids[0], research_ids[1], question, str(exc),
                    CHAT_MODEL_ID, dict(config),
                )
            except Exception as log_exc:
                research_warning("record_chat_error", log_exc)
        return {"ok": False, "error": str(exc), "model": CHAT_MODEL_ID}


def api_chat_clear(session_id="default"):
    session_id = str(session_id or "default").strip()[:80] or "default"
    with chat_sessions_lock:
        chat_sessions.pop(session_id, None)
    return {"ok": True}


def api_chat_restore_message(session_id="default", role="", content=""):
    """Restore one raw Q/A message from an exported chat CSV.

    Only raw user/assistant text is accepted. Experiment snapshots are not part of the
    exported chat history, so importing a bundle does not duplicate measurement context.
    """
    try:
        session_id = str(session_id or "default").strip()[:80] or "default"
        role = str(role or "").strip().lower()
        content = str(content or "")
        if role not in ("user", "assistant"):
            raise ValueError("role은 user 또는 assistant여야 합니다.")
        if not content:
            raise ValueError("복원할 대화 내용이 비어 있습니다.")
        if len(content) > 12000:
            raise ValueError("복원할 단일 대화 메시지가 너무 깁니다.")
        append_chat_history(session_id, role, content)
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def api_fit_restore(payload=""):
    """Restore the compact deterministic fitting record saved in an experiment bundle."""
    global fit_cache
    try:
        restored = json.loads(str(payload or "{}"))
        if not isinstance(restored, dict):
            raise ValueError("fitting 복원 데이터가 객체가 아닙니다.")
        with fit_cache_lock:
            fit_cache = dict(restored)
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def api_chat_info():
    return {
        "ok": True,
        "model": CHAT_MODEL_ID,
        "ready": chat_llm is not None,
        "error": chat_llm_error,
        "provider": "Arduino CloudLLM (LangChain-backed)",
        "memory": "CloudLLM memory disabled; server keeps only last 6 Q/A turns",
        "context_mode": "fresh experiment snapshot exactly once per request",
    }


def background_sync_loop():
    while True:
        try:
            if not is_imported_mode():
                state = sync_configured_max_cycles_to_mcu(read_mcu_status())
                sync_new_results(state)
        except Exception:
            pass
        time.sleep(0.5)


load_persistent_state()
# Rewrite persisted JSON/CSV once at startup so duplicate cycles left by older versions
# are cleaned even before the next measurement arrives.
save_measurements()

ui = WebUI()
ui.expose_api("GET", "/api/start", api_start)
ui.expose_api("GET", "/api/jog", api_jog)
ui.expose_api("GET", "/api/stop", api_stop)
ui.expose_api("GET", "/api/confirm-start", api_confirm_start)
ui.expose_api("GET", "/api/reset", api_reset)
ui.expose_api("GET", "/api/status", api_status)
ui.expose_api("GET", "/api/data", api_data)
ui.expose_api("GET", "/api/config", api_config)
ui.expose_api("GET", "/api/import-begin", api_import_begin)
ui.expose_api("GET", "/api/import-chunk", api_import_chunk)
ui.expose_api("GET", "/api/import-commit", api_import_commit)
ui.expose_api("GET", "/api/import-cancel", api_import_cancel)
ui.expose_api("GET", "/api/fit", api_fit)
ui.expose_api("GET", "/api/fit-restore", api_fit_restore)
ui.expose_api("GET", "/api/chat", api_chat)
ui.expose_api("GET", "/api/chat-clear", api_chat_clear)
ui.expose_api("GET", "/api/chat-restore-message", api_chat_restore_message)
ui.expose_api("GET", "/api/chat-info", api_chat_info)

# Store browser interactions and research records directly in one durable SQLite DB.
# This app registers its existing endpoints with the literal "/api/..." prefix,
# so the activity routes must follow the same convention.
register_activity_log_routes(ui, routes_include_api_prefix=True)

threading.Thread(target=background_sync_loop, daemon=True).start()
App.run()
