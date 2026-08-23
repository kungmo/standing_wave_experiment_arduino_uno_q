# Open Standing Wave Lab v3.17 - full experiment bundle export/import + CloudLLM chatbot
import csv
import json
import os
import math
import threading
import time
from datetime import datetime
from pathlib import Path

from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI
from arduino.app_bricks.cloud_llm import CloudLLM


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
    "data_mode": "live",
}

bridge_lock = threading.RLock()
data_lock = threading.RLock()
sync_lock = threading.Lock()  # Serialize MCU->Linux result synchronization.
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

CHAT_SYSTEM_PROMPT = """
너는 Open Standing Wave Lab의 실험 보조 챗봇이다.
이 장치는 Arduino UNO Q, 스텝모터-실-도르래, 마이크 센서를 이용해 관 내부의 음향 정상파를 위치에 따라 자동 측정한다.

반드시 다음 원칙을 지켜라.
- 답변은 기본적으로 한국어로 한다. 사용자가 다른 언어로 질문하면 그 언어를 따를 수 있다.
- 매 요청에 포함되는 '현재 실험 스냅샷'은 그 질문 시점의 최신 자료이며, 최근 대화보다 우선한다.
- 파장, 마디 간격, 음속 등 이미 Python이 결정론적으로 계산한 값이 있으면 다시 추측해서 다른 값을 만들지 않는다.
- fitting 결과가 없거나 데이터가 부족하면 없다고 분명히 말한다.
- 정상파의 기본 관계는 인접 마디 간격 Δx = λ/2, 따라서 λ = 2Δx 이다. 주파수 f가 주어지면 v = fλ를 사용할 수 있다.
- 추정 위치는 실-도르래 보정값으로 얻은 값이므로 절대 위치라고 단정하지 않는다.
- clipping 표시점은 신뢰도가 낮을 수 있으며 fitting에서는 제외된다.
- 이 장치의 학습 효과나 학생 성취 향상을 실험 데이터 없이 단정하지 않는다.
- 측정 데이터 초기화는 물리적 원점 복귀가 아니다. 새 실험 전에는 마이크를 START 표시선에 수동으로 맞추고 시작 위치를 확인한다.
- 최근 대화에는 과거 질문/답변만 들어 있고, 과거 실험 데이터 스냅샷은 저장되지 않는다. 현재 스냅샷만 실험 사실의 기준으로 사용한다.
- 답변은 실험실에서 바로 읽기 좋게 간결하되, 물리 설명이 필요한 질문에는 식과 근거를 포함한다.
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


def archive_current_measurements(reason):
    global measurements
    with data_lock:
        if not measurements:
            return None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = ARCHIVE_DIR / f"measurements_{stamp}.json"
        payload = {
            "archived_at": datetime.now().isoformat(timespec="seconds"),
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
                ]
            )
            dx = float(config.get("distance_per_cycle_cm", 0.0) or 0.0)
            frequency = float(config.get("frequency_hz", 0.0) or 0.0)
            first_at_start = configured_first_measure_at_start()
            max_cycles = configured_max_cycles()
            tube_length_cm = float(config.get("tube_length_cm", 0.0) or 0.0)
            tube_type = str(config.get("tube_type", "unknown") or "unknown")
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
        "time": datetime.now().isoformat(timespec="seconds"),
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


def api_start():
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
        bridge_call("set_run", 1)
        state = read_mcu_status()
        if state["running"]:
            return {"ok": True, "state": state}
        if not state["position_ready"]:
            return {
                "ok": False,
                "error": "시작 위치가 확인되지 않았거나 회차-위치 대응이 무효입니다. 데이터 초기화 후 마이크를 시작 표시선으로 수동 복귀시키고 '시작 위치 확인'을 누르십시오.",
                "state": state,
            }
        if state["last_result_rev"] >= state["max_cycles"]:
            return {
                "ok": False,
                "error": "최대 측정 횟수에 도달했습니다. 결과를 저장한 뒤 새 실험을 준비하십시오.",
                "state": state,
            }
        return {"ok": False, "error": "MCU가 측정 상태로 전환되지 않았습니다.", "state": state}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def api_stop():
    try:
        bridge_call("set_jog_direction", 0)
        bridge_call("set_run", 0)
        time.sleep(0.05)
        state = read_mcu_status()
        sync_new_results(state)
        return {"ok": True, "state": state}
    except Exception as exc:
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


def api_reset():
    global measurements
    try:
        # Do not let the background synchronizer re-add an old MCU result while reset is
        # clearing both the MCU buffer and the Linux-side current dataset.
        with sync_lock:
            bridge_call("set_jog_direction", 0)
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


def api_config(frequency_hz: float = 0.0, distance_per_cycle_cm: float = 0.0, max_cycles: int = 110, first_measure_at_start: int = 1, tube_length_cm: float = 0.0, tube_type: str = "open"):
    try:
        frequency_hz = max(0.0, float(frequency_hz))
        distance_per_cycle_cm = max(0.0, float(distance_per_cycle_cm))
        tube_length_cm = max(0.0, float(tube_length_cm))
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
            save_config()
            save_measurements()
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
                config["data_mode"] = "live"
        save_config()
        save_measurements()
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


def api_import_begin(frequency_hz: float = 0.0, distance_per_cycle_cm: float = 0.0, max_cycles: int = 110, first_measure_at_start: int = 1, tube_length_cm: float = 0.0, tube_type: str = "unknown", total_rows: int = 0):
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
                "total_rows": total_rows,
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
            config["data_mode"] = "imported"
        save_config()
        save_measurements()
        return {
            "ok": True,
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


def api_fit():
    global fit_cache
    try:
        result = fit_standing_wave()
        with fit_cache_lock:
            fit_cache = dict(result) if isinstance(result, dict) else None
        return result
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


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


def build_experiment_context():
    rows, data_lines = compact_measurement_context()
    frequency = float(config.get("frequency_hz", 0.0) or 0.0)
    dx = float(config.get("distance_per_cycle_cm", 0.0) or 0.0)
    max_cycles = configured_max_cycles()
    first_at_start = configured_first_measure_at_start()
    tube_length_cm = float(config.get("tube_length_cm", 0.0) or 0.0)
    tube_type = str(config.get("tube_type", "unknown") or "unknown")
    tube_type_ko = {"open": "개관(양쪽 열림)", "one_end_closed": "한쪽이 막힌 관", "unknown": "미지정"}.get(tube_type, "미지정")
    clipped_count = sum(1 for row in rows if row.get("clipped"))
    fit = get_fit_for_chat()

    context_lines = [
        "장치: Arduino UNO Q 기반 음향 정상파 자동 스캔 장치",
        f"사용 관 종류 = {tube_type_ko}",
        f"사용 관 길이 = {tube_length_cm:.3f} cm" if tube_length_cm > 0 else "사용 관 길이 = 미입력",
        "기구: 스텝모터 + 실/도르래로 마이크를 한 방향으로 이동, 새 실험 전 수동 복귀",
        "현재 빠른 스캔 설정: 모터 20 RPM, 이동 후 안정화 0.5 s, 200 ms Vpp window 10회, 최대/최소 1개씩 제외 후 8개 평균/표준편차",
        f"설정 주파수 f = {frequency:.3f} Hz" if frequency > 0 else "설정 주파수 f = 미입력",
        f"1회 이동당 추정 거리 = {dx:.4f} cm" if dx > 0 else "1회 이동당 추정 거리 = 미보정(그래프 x축은 회차)",
        f"설정 마지막 회차 = {max_cycles}회",
        f"첫 데이터 위치 = {'확인된 시작 위치(1회차 = 0 cm)' if first_at_start else '1회 이동 후'}",
        f"데이터 출처 = {'불러온 실험 데이터' if is_imported_mode() else '현재 MCU 실시간 측정 데이터'}",
        f"현재 저장된 측정점 = {len(rows)}개, clipping 의심점 = {clipped_count}개",
    ]

    if isinstance(fit, dict) and fit.get("ok"):
        context_lines.extend(
            [
                "결정론적 Python fitting 결과:",
                f"- wavelength λ = {float(fit['wavelength_cm']):.4f} cm",
                f"- node spacing λ/2 = {float(fit['node_spacing_cm']):.4f} cm",
                f"- R^2 = {float(fit['r2']):.5f}" if fit.get("r2") is not None else "- R^2 = 없음",
                f"- baseline = {float(fit['baseline_v']):.4f} V",
                f"- amplitude = {float(fit['amplitude_v']):.4f} V",
                f"- used points = {int(fit['used_points'])}, excluded clipped = {int(fit['excluded_clipped_points'])}",
            ]
        )
        if fit.get("sound_speed_m_s") is not None:
            context_lines.append(f"- sound speed v = {float(fit['sound_speed_m_s']):.3f} m/s")
        context_lines.append(f"- model = {fit.get('model', '')}")
    else:
        context_lines.append(f"결정론적 Python fitting 결과: 없음 ({fit.get('error', '미실행') if isinstance(fit, dict) else '미실행'})")

    if data_lines:
        context_lines.append("현재 측정 데이터 전체(CSV; 이 요청에서 1회만 제공):")
        context_lines.extend(data_lines)
    else:
        context_lines.append("현재 측정 데이터 전체: 아직 없음")

    return "\n".join(context_lines)


def get_chat_history_text(session_id):
    with chat_sessions_lock:
        history = list(chat_sessions.get(session_id, []))
    if not history:
        return "(이전 대화 없음)"
    parts = []
    for item in history[-CHAT_HISTORY_TURNS * 2 :]:
        role = "사용자" if item.get("role") == "user" else "보조자"
        parts.append(f"{role}: {item.get('content', '')}")
    return "\n".join(parts)


def append_chat_history(session_id, role, content):
    with chat_sessions_lock:
        history = chat_sessions.setdefault(session_id, [])
        history.append({"role": role, "content": str(content)})
        max_items = CHAT_HISTORY_TURNS * 2
        if len(history) > max_items:
            del history[:-max_items]


def api_chat(question="", session_id="default"):
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
        prompt = f"""[현재 실험 스냅샷 - 최신 자료, 이 요청에서 한 번만 제공]
{experiment_context}

[최근 대화 - 최대 {CHAT_HISTORY_TURNS}턴, 질문/답변 텍스트만 포함]
{history_text}

[현재 사용자 질문]
{question}

현재 실험 스냅샷을 수치적 사실의 기준으로 사용하고, 최근 대화는 질문의 의도와 연속성을 이해하는 데만 사용하여 답하라."""

        with chat_lock:
            answer = chat_llm.chat(prompt)
        answer = str(answer or "").strip()
        if not answer:
            raise RuntimeError("LLM이 빈 응답을 반환했습니다.")

        # Store ONLY Q/A text. The experiment snapshot/prompt is deliberately not stored.
        append_chat_history(session_id, "user", question)
        append_chat_history(session_id, "assistant", answer)
        return {
            "ok": True,
            "answer": answer,
            "model": CHAT_MODEL_ID,
            "context_points": len(measurement_payload()),
            "history_turn_limit": CHAT_HISTORY_TURNS,
            "context_mode": "fresh_snapshot_once_per_request",
        }
    except Exception as exc:
        print(f"CHATBOT FAILED: {exc}")
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

threading.Thread(target=background_sync_loop, daemon=True).start()
App.run()
