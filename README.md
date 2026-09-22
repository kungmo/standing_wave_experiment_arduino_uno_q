# Open Standing Wave Lab (v4.6)

Arduino UNO Q 기반의 **교육용 음향 정상파(Standing Wave) 자동 측정 및 AI 탐구 보조 실험 장치**입니다.

관(Tube) 내부에서 마이크 센서를 스텝 모터와 실-도르래 기구로 정밀하게 이동시키며 위치별 음압 진폭($V_{pp}$)을 자동으로 스캔합니다. 측정된 데이터는 실시간 웹 인터페이스에 시각화되며, 물리 법칙에 기반한 사인 곡선 맞춤(Curve Fitting), 그리고 학생의 자기주도적 탐구를 지원하는 생성형 AI(Gemini) 챗봇이 유기적으로 연동되어 있습니다.

---

## 🌟 주요 특징

1. **자동화된 공간 스캔 및 실시간 시각화**
   * 스텝 모터(28BYJ-48 + ULN2003)와 실-도르래 기구를 이용해 마이크를 일정 간격으로 정밀 이동.
   * 각 지점마다 안정화 대기 후 $200\text{ ms}$ 구간을 10회 샘플링하여 평균 $V_{pp}$, 표준편차, 클리핑(Clipping) 여부를 실시간 도출.
   * 웹 브라우저에서 실시간 산점도 및 오차 막대, 맞춤 곡선을 즉시 확인.

2. **이원화된 파장($\lambda$) 및 음속($v$) 분석**
   * **수동 탐구 모드**: 학생이 그래프 상에서 직접 마디(Node) 두 개 또는 배(Antinode) 두 개를 클릭하여 거리 $\Delta x$의 2배($\lambda = 2\Delta x$)로 파장을 계산.
   * **전역 곡선 맞춤(Fitting)**: 클리핑 이상점을 제외한 비선형 최소제곱 모델($V_{pp}(x) = \text{baseline} + A|\sin(kx + \phi)|$)을 적용하여 결정계수($R^2$) 및 음속($v = f\lambda$) 자동 산출.

3. **실험 데이터 근거형(Grounded) AI 챗봇 (Gemini 연동)**
   * Arduino CloudLLM Brick을 통해 Gemini(`google:gemini-3.5-flash-lite`) 연동.
   * 학생의 질문 시점의 최신 실험 조건(관 종류, 관 길이, 온도, 측정값, fitting 결과)을 단일 XML 스냅샷으로 주입하여 환각(Hallucination) 없이 실제 데이터를 근거로 답변.
   * 고등학교 1학년 눈높이에 맞춘 소크라테스식 발문 및 오개념(예: 마디와 배 사이 $\lambda/4$를 $\lambda/2$로 오인) 교정 지침 내장.

4. **연구용 완전 기록 DB (Crash-Safe SQLite)**
   * 브라우저 상의 모든 UI 조작(클릭, 설정 변경, 수동 점 선택)과 챗봇 대화 내용, 측정 데이터를 eMMC 내부 SQLite DB(`standing_wave_activity.sqlite3`)에 실시간 원자적(Atomic) 트랜잭션으로 기록.
   * 전원 강제 차단에도 복구 가능한 Rollback-Journal(`DELETE` 모드, `synchronous=EXTRA`) 적용.

5. **실험 데이터 세트 불러오기(Offline Mode) 지원**
   * 실제 물리 장치 연결 없이도 과거에 측정한 ZIP 실험 번들(측정 데이터, 챗봇 대화, fitting 결과)을 브라우저에 드래그하여 오프라인 데이터 분석 및 챗봇 실습 가능.

---

## 🛠️ 하드웨어 구성 및 부품 목록 (BOM)

### 1. 주요 부품 목록

| 구분 | 품목 명칭 | 권장 사양 / 비고 | 수량 |
| :--- | :--- | :--- | :--- |
| **메인 컨트롤러** | Arduino UNO Q | Qualcomm Linux AP + STM32 MCU 하이브리드 보드 | 1개 |
| **액추에이터** | 28BYJ-48 스텝 모터 | 5V 기어드 스테퍼 모터 | 1개 |
| **모터 드라이버** | ULN2003 드라이버 보드 | 4상 달링턴 드라이버 모듈 | 1개 |
| **센서** | 아날로그 마이크 모듈 | MAX4466 또는 MAX9814 (0~3.3V 아날로그 출력) | 1개 |
| **음향 발생기** | 스피커 / 음향 발생 장치 | 신호 발생기 앱 또는 소형 앰프 스피커 (단일 주파수 발생용) | 1개 |
| **음향 관 (공명관)** | 투명 아크릴 관 | 내경 30~50 mm, 길이 80~100 cm | 1개 |
| **기구부** | 실, 도르래, 마이크 카트 | 나일론 실 또는 연신율이 낮은 낚싯줄, 3D 프린팅 슬라이더 | 1세트 |
| **전원** | 5V/2A 이상 USB 전원 | 아두이노 및 모터 전원 공급 | 1개 |

### 2. 결선도 (Pin Wiring)

```text
[Arduino UNO Q]                   [ULN2003 Driver]
Digital Pin 8   ──────────────────> IN1
Digital Pin 9   ──────────────────> IN2
Digital Pin 10  ──────────────────> IN3
Digital Pin 11  ──────────────────> IN4
5V (or Ext 5V)  ──────────────────> 5V-12V (+)
GND             ──────────────────> GND (-)

[Arduino UNO Q]                   [Microphone Sensor]
Analog Pin A0   <─────────────────  OUT / Analog Out (0 ~ 3.3V)
3.3V            ──────────────────> VCC
GND             ──────────────────> GND
```

> ⚠️ **주의사항**:
> * 마이크 출력 신호는 Arduino UNO Q ADC 입력 허용 전압 범위인 **0 ~ 3.3V**를 넘지 않아야 합니다. 증폭도가 너무 크면 $V_{pp}$가 클리핑되므로 모듈의 가변저항을 조정해 주십시오.
> * 모터 동작 시 전원 노이즈로 인한 보드 리셋을 방지하기 위해 충분한 용량의 5V 전원을 사용하십시오.

---

## 📂 파일 및 폴더 구조

```text
├── assets/
│   └── index.html                         # 프론트엔드 단일 웹 애플리케이션
├── python/
│   ├── main.py                            # 백엔드 코어 (Bridge 통신, WebUI API, Fitting, 챗봇)
│   ├── activity_log.py                    # 연구용 SQLite DB 영속화 엔진
│   └── activity_log_api.py                # 브라우저 사용자 활동 로깅 엔드포인트
├── data/                                  # 앱 실행 시 자동 생성
│   └── standing_wave_activity.sqlite3     # 사용자 행동 및 측정 연구 데이터베이스
└── README.md                              # 프로젝트 설명서
```

---

## 🚀 빠른 시작 가이드 (Quick Start)

본 프로젝트는 **Arduino App Lab / App Bricks** 아키텍처를 기반으로 구동됩니다.

### 1단계: MCU 스케치 업로드
1. Arduino IDE를 열고 아두이노 우노 Q의 STM32 MCU에 전용 정상파 측정 펌웨어 스케치를 업로드합니다.
2. 펌웨어는 모터 스텝 제어, 안정화 지연, ADC 반복 샘플링 및 Bridge RPC 함수(`set_run`, `get_running`, `get_result_avg_mv`, `set_jog_direction` 등)를 처리합니다.

### 2단계: Python 환경 및 앱 배치
1. Arduino UNO Q의 Linux 파일 시스템(`/home/arduino`)에 저장소 파일들을 복사합니다.
2. 내장된 Python 3 환경에서 App Bricks 패키지가 구성되어 있는지 확인합니다.
   * `sqlite3`, `math`, `csv`는 Python 표준 라이브러리입니다.
   * `arduino.app_utils`, `arduino.app_bricks.web_ui`, `arduino.app_bricks.cloud_llm` 라이브러리가 기본 탑재되어 있습니다.

### 3단계: AI 챗봇 API 키 설정
1. Arduino Cloud 또는 App Lab 설정 화면에서 **Gemini API Key**를 등록합니다.
2. `main.py`의 `CHAT_MODEL_ID`는 기본값으로 `"google:gemini-3.5-flash-lite"`가 지정되어 있습니다.

### 4단계: 애플리케이션 실행
터미널에서 애플리케이션을 실행합니다:
```bash
python3 python/main.py
```
실행 후 동일 로컬 네트워크의 PC 또는 태블릿 브라우저에서 `http://<UNO_Q_IP_주소>:<포트>`로 접속합니다.

---

## 📖 실험 진행 방법

1. **실험 조건 입력 (1. 실험 설정)**
   * 스피커로 관 내부에 인가하는 음원의 **공명 진동수($f$)**, **관 길이**, **관 종류(개관/폐관)**, **실험실 온도($^\circ\text{C}$)**를 입력합니다.
   * 마이크 1회 이동당 이동 거리 보정값(예: 1회당 0.5 cm)을 자로 측정하여 입력합니다.
   * `설정 저장`을 클릭하여 설정을 저장합니다.

2. **장력 조절 및 시작 위치 정렬**
   * **실 장력 조절(JOG)**: `▶ 정방향` / `◀ 역방향` 버튼으로 스텝 모터를 미세 구동하여 느슨해진 실의 장력을 맞춥니다.
   * 마이크 카트를 손으로 되감아 관 입구의 **START 기준선(0 cm)**에 정확히 맞춥니다.
   * `시작 위치 확인` 버튼을 눌러 위치 기준점을 초기화합니다.

3. **자동 측정 시작 (2. 장치 제어)**
   * `측정 시작`을 클릭하면 마이크가 단계별로 전진하며 소리 진폭을 측정합니다.
   * **실시간 정상파 그래프**에 점과 오차 막대가 순차적으로 기록됩니다.
   * 측정이 최대 회차에 도달하거나 `정지`를 누르면 측정이 완료되고 모터 전원이 안전하게 차단됩니다.

4. **파장 분석 및 AI 질의**
   * **수동 분석**: 그래프에서 가장 소리가 작은 지점(마디) 2곳 또는 가장 큰 지점(배) 2곳을 클릭하여 마디 간격과 파장을 확인합니다.
   * **곡선 맞춤**: `파장 계산하기`를 클릭하여 전체 비클리핑 측정점에 최적화된 이론 파장, 음속, $R^2$을 확인합니다.
   * **챗봇 탐구**: 우측 하단 💬 버튼을 눌러 챗봇에게 "지금 그래프에서 나타나는 정상파는 몇 배 진동인가요?", "계산된 음속과 이론 음속에 차이가 나는 이유는 무엇인가요?" 등을 질문합니다.

5. **실험 세트 내보내기 및 공유**
   * `실험 세트 저장 (ZIP)` 버튼을 클릭하면 원본 측정값 CSV, 챗봇 대화록 CSV, 피팅 결과 JSON이 하나의 압축 파일로 다운로드됩니다.
   * 다른 PC나 하드웨어가 없는 환경에서도 `실험 세트 불러오기 (ZIP)`를 통해 저장된 실험을 그대로 불러와 분석할 수 있습니다.

---

## 🔬 연구용 데이터베이스 (SQLite) 구조

모든 활동 내역은 `data/standing_wave_activity.sqlite3` 파일에 로컬 시간대(ISO 8601, UTC 오프셋 포함)로 안전하게 기록됩니다. **DBeaver** 등의 SQLite 도구로 열어 분석할 수 있습니다.

| 테이블 / 뷰 | 설명 | 주요 필드 |
| :--- | :--- | :--- |
| `activity_sessions` | 사용자 브라우저 접속 세션 | `session_id`, `started_at_server_local`, `user_agent`, `screen_json` |
| `experiments` | 개별 실험 시행 메타데이터 | `experiment_id`, `status`, `frequency_hz`, `tube_length_cm`, `temperature_c` |
| `measurements` | 회차별 측정 원시 데이터 | `cycle`, `estimated_position_cm`, `avg_peak_to_peak_v`, `std_peak_to_peak_v`, `clipped` |
| `chat_messages` | 학생-챗봇 간의 전체 대화 기록 | `role` (user/assistant/error), `message`, `model`, `context_points` |
| `analysis_results` | 실행된 Curve Fitting 결과 | `wavelength_cm`, `r2`, `sound_speed_m_s`, `result_json` |
| `activity_events` | 450ms 단위 브라우저 UI 조작 로그 | `event_type`, `target_id`, `value_text`, `elapsed_ms` |
| `research_measurements` *(View)* | 실험 설정과 측정값을 결합한 통합 분석 뷰 | 논문 및 통계 분석용 최적화 뷰 |
| `chat_transcript` *(View)* | 실험별 대화 흐름 분석용 뷰 | 학생 질문 의도 및 챗봇 응답 평가 뷰 |

---

## 🛡️ 개인정보 및 라이선스 안내

* **데이터 보호**: 학생의 이름이나 학번 등이 질문 창에 입력될 경우 연구 DB에 평문으로 저장됩니다. 교육 연구 목적으로 활용 시 참여 학생 동의 절차를 준수하고, 공유 전 DB 파일을 초기화하십시오.
* **라이선스**: 본 프로젝트는 오픈소스 교육 및 학술 연구 목적으로 자유롭게 수정, 배포 및 활용할 수 있습니다.
