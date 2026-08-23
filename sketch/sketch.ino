#include <Arduino.h>
#include <Arduino_RouterBridge.h>
#include <Stepper.h>
#include <math.h>

// Open Standing Wave Lab - Arduino UNO Q controller, v3.14 (initial-position measurement + CSV analysis mode)
//
// Mechanical policy for the string-and-pulley apparatus:
// - The motor performs the forward scan and a low-speed manual JOG used only before a scan.
// - Whenever the apparatus is stopped, D8..D11 are driven LOW so the coils are released.
// - There is NO automatic homing/limit switch.
// - Before a new experiment, the user manually rewinds the pulley to the marked start
//   position and then confirms that position in the web UI.
//
// Measurement sequence:
//   1) optionally measure the first point at the confirmed start position (default)
//      OR move 197 steps first (legacy mode)
//   2) wait 500 ms before each measurement set
//   3) take 10 peak-to-peak measurements, 200 ms each
//   4) discard one maximum and one minimum
//   5) calculate the mean and sample standard deviation of the remaining 8 values
//
// If STOP occurs in the middle of a cycle, the cycle-to-position mapping is no longer
// trustworthy. positionReady is then cleared and the scan cannot resume until the user
// resets the data, manually returns the microphone to the start mark, and confirms it.

const int stepsPerRevolution = 197;
const int motorSpeedRpm = 20;
const int manualJogSpeedRpm = 10;
const unsigned long jogWatchdogMs = 1500;

// Result storage has a fixed RAM capacity, while the actual scan end point is user-configurable.
// 1000 cycles supports, for example, a 1 m tube scanned at 0.1 cm/cycle.
const int MAX_RESULT_CAPACITY = 1000;
volatile int maxCycles = 110;
volatile int firstMeasureAtStart = 1;  // 1: rev 1 at x=0, 0: move once before rev 1.

Stepper myStepper(stepsPerRevolution, 11, 9, 10, 8);

const int MOTOR_IN1_PIN = 8;
const int MOTOR_IN2_PIN = 9;
const int MOTOR_IN3_PIN = 10;
const int MOTOR_IN4_PIN = 11;

const unsigned long settleDelayMs = 500;
const unsigned long sampleWindowMs = 200;
const int measurementCount = 10;
const float adcFullScaleVolts = 3.3f;
const float adcCounts = 1024.0f;

// Samples this close to either ADC rail are marked as possible clipping.
const unsigned int clipLowCount = 5;
const unsigned int clipHighCount = 1018;

volatile int runRequested = 0;
volatile int resetRequested = 0;
volatile int phase = 0;  // 0 stopped, 1 moving, 2 settling, 3 measuring
volatile int revnum = 0;
volatile int stepIndex = 0;
volatile int measureIndex = 0;
volatile int dataSeq = 0;
volatile int lastResultRev = 0;
volatile int lastAvgMilliVolts = 0;
volatile int lastStdMilliVolts = 0;
volatile int lastClipped = 0;
volatile int positionReady = 0;  // User-confirmed start reference + intact cycle-position mapping.
volatile int jogDirection = 0;    // -1 reverse, 0 stopped, +1 forward (scan direction).
unsigned long jogLastCommandMs = 0;
int bootId = 1;  // Changes on MCU reboot so Linux can distinguish experiment sessions.

unsigned long settleStartMs = 0;
unsigned long windowStartMs = 0;
unsigned int signalMax = 0;
unsigned int signalMin = 1023;
bool windowHasSample = false;
bool measurementSetClipped = false;
float volts[measurementCount];

// 1-based result storage. Index 0 is unused.
int resultAvgMilliVolts[MAX_RESULT_CAPACITY + 1];
int resultStdMilliVolts[MAX_RESULT_CAPACITY + 1];
uint8_t resultClipped[MAX_RESULT_CAPACITY + 1];

void releaseMotor();

int set_run(int on) {
  if (on == 0) {
    runRequested = 0;
    return 0;
  }

  // Never start an automatic scan while the manual tension JOG is active.
  if (jogDirection != 0) {
    runRequested = 0;
    return 0;
  }

  if (positionReady && lastResultRev < maxCycles && !resetRequested) {
    runRequested = 1;
  } else {
    runRequested = 0;
  }
  return runRequested;
}

int confirm_start_position() {
  // A new zero/reference may only be confirmed for an empty data set.
  // This prevents a manually rewound microphone from being appended to old coordinates.
  if (runRequested || resetRequested || lastResultRev > 0 || phase != 0 || jogDirection != 0) {
    return 0;
  }
  positionReady = 1;
  return 1;
}

int set_jog_direction(int direction) {
  // direction: +1 = normal scan direction, -1 = reverse, 0 = stop/release.
  if (direction == 0) {
    jogDirection = 0;
    myStepper.setSpeed(motorSpeedRpm);
    releaseMotor();
    return 0;
  }

  if (direction != 1 && direction != -1) return jogDirection;

  // JOG is only for preparing an empty experiment. Moving the microphone after results
  // exist would destroy the cycle-to-position relationship of those stored data.
  if (runRequested || resetRequested || phase != 0 || lastResultRev > 0) return jogDirection;

  positionReady = 0;  // Any manual motor movement requires a fresh start-position confirmation.
  jogDirection = direction;
  jogLastCommandMs = millis();
  myStepper.setSpeed(manualJogSpeedRpm);
  return jogDirection;
}

int get_jog_direction() { return jogDirection; }

int reset_all() {
  runRequested = 0;
  jogDirection = 0;
  resetRequested = 1;
  return 1;
}

int get_running() { return runRequested; }
int get_phase() { return phase; }
int get_revnum() { return revnum; }
int get_step_index() { return stepIndex; }
int get_measure_index() { return measureIndex; }
int get_data_seq() { return dataSeq; }
int get_last_result_rev() { return lastResultRev; }
int get_last_avg_mv() { return lastAvgMilliVolts; }
int get_last_std_mv() { return lastStdMilliVolts; }
int get_last_clipped() { return lastClipped; }
int set_max_cycles(int requested) {
  // Do not change the scan end point while a cycle is active. Also never allow the
  // new end point to fall behind data that have already been completed.
  if (runRequested || resetRequested || phase != 0 || jogDirection != 0) return maxCycles;
  if (requested < 1 || requested > MAX_RESULT_CAPACITY) return maxCycles;
  if (requested < lastResultRev) return maxCycles;
  maxCycles = requested;
  return maxCycles;
}

int get_max_cycles() { return maxCycles; }
int set_first_measure_at_start(int enabled) {
  // This changes the physical scan sequence and coordinate mapping, so only allow it
  // for an empty, idle experiment.
  if (runRequested || resetRequested || phase != 0 || jogDirection != 0 || lastResultRev > 0) {
    return firstMeasureAtStart;
  }
  firstMeasureAtStart = enabled ? 1 : 0;
  return firstMeasureAtStart;
}
int get_first_measure_at_start() { return firstMeasureAtStart; }
int get_max_result_capacity() { return MAX_RESULT_CAPACITY; }
int get_position_ready() { return positionReady; }
int get_boot_id() { return bootId; }

int get_result_avg_mv(int rev) {
  if (rev < 1 || rev > MAX_RESULT_CAPACITY || rev > lastResultRev) return -1;
  return resultAvgMilliVolts[rev];
}

int get_result_std_mv(int rev) {
  if (rev < 1 || rev > MAX_RESULT_CAPACITY || rev > lastResultRev) return -1;
  return resultStdMilliVolts[rev];
}

int get_result_clipped(int rev) {
  if (rev < 1 || rev > MAX_RESULT_CAPACITY || rev > lastResultRev) return -1;
  return resultClipped[rev] ? 1 : 0;
}

void releaseMotor() {
  digitalWrite(MOTOR_IN1_PIN, LOW);
  digitalWrite(MOTOR_IN2_PIN, LOW);
  digitalWrite(MOTOR_IN3_PIN, LOW);
  digitalWrite(MOTOR_IN4_PIN, LOW);
}

void stopAndPreserveData(bool interruptedCycle) {
  if (interruptedCycle) {
    // The microphone may have moved without a completed result, so the coordinate mapping
    // can no longer be trusted for a resumed scan.
    positionReady = 0;
  }
  phase = 0;
  revnum = lastResultRev;
  stepIndex = 0;
  measureIndex = 0;
  releaseMotor();
}

void resetAllState() {
  runRequested = 0;
  phase = 0;
  revnum = 0;
  stepIndex = 0;
  measureIndex = 0;
  dataSeq = 0;
  lastResultRev = 0;
  lastAvgMilliVolts = 0;
  lastStdMilliVolts = 0;
  lastClipped = 0;
  positionReady = 0;  // Resetting data never claims that the physical microphone is at x=0.
  jogDirection = 0;
  jogLastCommandMs = 0;

  settleStartMs = 0;
  windowStartMs = 0;
  signalMax = 0;
  signalMin = 1023;
  windowHasSample = false;
  measurementSetClipped = false;

  for (int i = 0; i < measurementCount; ++i) volts[i] = 0.0f;
  for (int i = 0; i <= MAX_RESULT_CAPACITY; ++i) {
    resultAvgMilliVolts[i] = 0;
    resultStdMilliVolts[i] = 0;
    resultClipped[i] = 0;
  }
  releaseMotor();
}

void beginMeasurement() {
  measureIndex = 0;
  signalMax = 0;
  signalMin = 1023;
  windowHasSample = false;
  measurementSetClipped = false;
  windowStartMs = millis();
  phase = 3;
}

void beginNextMeasurementWindow() {
  signalMax = 0;
  signalMin = 1023;
  windowHasSample = false;
  windowStartMs = millis();
}

void finishMeasurementSet() {
  int maxIndex = 0;
  int minIndex = 0;
  for (int i = 1; i < measurementCount; ++i) {
    if (volts[i] > volts[maxIndex]) maxIndex = i;
    if (volts[i] < volts[minIndex]) minIndex = i;
  }

  float sum = 0.0f;
  int kept = 0;
  for (int i = 0; i < measurementCount; ++i) {
    if (i == maxIndex || i == minIndex) continue;
    sum += volts[i];
    kept += 1;
  }
  float avgVolts = kept > 0 ? sum / kept : 0.0f;

  float sumSq = 0.0f;
  for (int i = 0; i < measurementCount; ++i) {
    if (i == maxIndex || i == minIndex) continue;
    float d = volts[i] - avgVolts;
    sumSq += d * d;
  }
  float stdVolts = kept > 1 ? sqrtf(sumSq / (kept - 1)) : 0.0f;

  lastAvgMilliVolts = (int)lroundf(avgVolts * 1000.0f);
  lastStdMilliVolts = (int)lroundf(stdVolts * 1000.0f);
  lastClipped = measurementSetClipped ? 1 : 0;

  resultAvgMilliVolts[revnum] = lastAvgMilliVolts;
  resultStdMilliVolts[revnum] = lastStdMilliVolts;
  resultClipped[revnum] = lastClipped ? 1 : 0;
  lastResultRev = revnum;
  dataSeq += 1;

  phase = 0;
  stepIndex = 0;
  measureIndex = 0;

  if (lastResultRev >= maxCycles) {
    runRequested = 0;
    revnum = lastResultRev;
    releaseMotor();
  }
}

void collectMeasurement() {
  unsigned int sample = (unsigned int)analogRead(A0);

  if (sample <= clipLowCount || sample >= clipHighCount) {
    measurementSetClipped = true;
  }
  if (sample > signalMax) signalMax = sample;
  if (sample < signalMin) signalMin = sample;
  windowHasSample = true;

  if (millis() - windowStartMs < sampleWindowMs) return;

  unsigned int peakToPeak = 0;
  if (windowHasSample && signalMax >= signalMin) {
    peakToPeak = signalMax - signalMin;
  }

  volts[measureIndex] = (peakToPeak * adcFullScaleVolts) / adcCounts;
  measureIndex += 1;

  if (measureIndex >= measurementCount) finishMeasurementSet();
  else beginNextMeasurementWindow();
}

void setup() {
  analogReadResolution(10);

  // A lightweight non-cryptographic boot/session identifier. It only needs to change
  // with high probability when the MCU restarts; it is not used for security.
  randomSeed(((unsigned long)micros() << 16) ^ (unsigned long)analogRead(A1) ^ ((unsigned long)analogRead(A2) << 8));
  bootId = (int)random(1, 2147483646L);
  myStepper.setSpeed(motorSpeedRpm);
  releaseMotor();

  Bridge.begin();
  Bridge.provide("set_run", set_run);
  Bridge.provide("set_jog_direction", set_jog_direction);
  Bridge.provide("get_jog_direction", get_jog_direction);
  Bridge.provide("confirm_start_position", confirm_start_position);
  Bridge.provide("reset_all", reset_all);
  Bridge.provide("get_running", get_running);
  Bridge.provide("get_phase", get_phase);
  Bridge.provide("get_revnum", get_revnum);
  Bridge.provide("get_step_index", get_step_index);
  Bridge.provide("get_measure_index", get_measure_index);
  Bridge.provide("get_data_seq", get_data_seq);
  Bridge.provide("get_last_result_rev", get_last_result_rev);
  Bridge.provide("get_last_avg_mv", get_last_avg_mv);
  Bridge.provide("get_last_std_mv", get_last_std_mv);
  Bridge.provide("get_last_clipped", get_last_clipped);
  Bridge.provide("set_max_cycles", set_max_cycles);
  Bridge.provide("get_max_cycles", get_max_cycles);
  Bridge.provide("set_first_measure_at_start", set_first_measure_at_start);
  Bridge.provide("get_first_measure_at_start", get_first_measure_at_start);
  Bridge.provide("get_max_result_capacity", get_max_result_capacity);
  Bridge.provide("get_position_ready", get_position_ready);
  Bridge.provide("get_boot_id", get_boot_id);
  Bridge.provide("get_result_avg_mv", get_result_avg_mv);
  Bridge.provide("get_result_std_mv", get_result_std_mv);
  Bridge.provide("get_result_clipped", get_result_clipped);
}

void loop() {
  if (resetRequested) {
    resetAllState();
    resetRequested = 0;
    delay(1);
    return;
  }

  if (jogDirection != 0) {
    // The browser refreshes this command every 500 ms. If communication is lost,
    // automatically stop and release the coils instead of running indefinitely.
    if (millis() - jogLastCommandMs > jogWatchdogMs) {
      jogDirection = 0;
      myStepper.setSpeed(motorSpeedRpm);
      releaseMotor();
      delay(1);
      return;
    }

    // Keep the requested JOG direction explicit and sticky. The watchdog above may
    // change JOG only to STOP (0); it never changes reverse (-1) into forward (+1).
    if (jogDirection == 1) {
      myStepper.step(1);
    } else if (jogDirection == -1) {
      myStepper.step(-1);
    }
    delay(1);
    return;
  }

  if (!runRequested) {
    // phase != 0 means STOP occurred after the next cycle had already started.
    bool interruptedCycle = (phase != 0);
    stopAndPreserveData(interruptedCycle);
    delay(1);
    return;
  }

  if (!positionReady) {
    runRequested = 0;
    stopAndPreserveData(false);
    delay(1);
    return;
  }

  if (lastResultRev >= maxCycles) {
    runRequested = 0;
    stopAndPreserveData(false);
    delay(1);
    return;
  }

  switch (phase) {
    case 0:
      myStepper.setSpeed(motorSpeedRpm);
      revnum = lastResultRev + 1;
      stepIndex = 0;

      // Default mode: the first data point is taken at the confirmed start position,
      // so rev 1 corresponds to x=0 cm. Later cycles move once before measuring.
      if (firstMeasureAtStart && lastResultRev == 0) {
        settleStartMs = millis();
        phase = 2;
      } else {
        phase = 1;
      }
      break;

    case 1:
      // One step per loop keeps STOP responsive and releases the coils immediately afterward.
      myStepper.step(1);
      stepIndex += 1;
      if (stepIndex >= stepsPerRevolution) {
        stepIndex = stepsPerRevolution;
        settleStartMs = millis();
        phase = 2;
      }
      break;

    case 2:
      if (millis() - settleStartMs >= settleDelayMs) beginMeasurement();
      break;

    case 3:
      collectMeasurement();
      break;

    default:
      phase = 0;
      break;
  }
}
