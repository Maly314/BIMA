/*
  BIMA — Teensy 4.1 + four SparkFun LSM6DSV16X IMUs + SSD1306 OLED

  This is the LSM6DSV16X hardware (handoff 2026-08-03) carrying the BIMA boot
  animation and STANDBY/RUNNING display from the earlier ICM-20948 build
  (handoff 2026-07-20).

  Wiring — do not change without a hardware redesign (see the handoff docs):
    IMU 1: Wire,  SDA 18, SCL 19, address 0x6B
    IMU 2: Wire,  SDA 18, SCL 19, address 0x6A
    IMU 3: Wire1, SDA 17, SCL 16, address 0x6B
    IMU 4: Wire1, SDA 17, SCL 16, address 0x6A
    OLED:  Wire2, SDA 25, SCL 24, address 0x3C (0x3D fallback)

  Serial commands: r = RUNNING, s = STANDBY, i = rescan buses, d = diagnostics

  TWO serial formats are emitted, because two consumers disagree:
    * Labeled tab-separated `I1_AX:...` — the Windows viewer
      (Teensy_IMU_Viewer) parses this and the 2026-08-03 handoff requires it
      to be preserved. Emitted at a display-rate cadence only; there is no
      benefit to feeding a chart faster than it can be looked at, and the
      spare cycles go to the recorded stream instead.
    * `IMUn A: ... m/s^2 | G: ... rad/s` blocks terminated by `T:`/`N:` and a
      blank line — the BIMA web app parses this (app/sensor-calibration.ts)
      and uses the blank line as its per-sample record boundary. Emitted every
      sample, because this is the stream that becomes study data.
  Neither parser matches the other's lines, so they coexist safely. The USB
  CDC link is not really running at SERIAL_BAUD — it is native USB — so the
  combined volume is not the constraint. The I2C read cycle is.

  TIMING CONTRACT
  Every record carries the device's own clock, not just the host's arrival
  time:
    T: <device_us>   monotonic microseconds since boot, 64-bit, never wraps
    N: <seq>         sample counter, +1 per record, gaps mean dropped packets
  The host can then fit host_ms = a*device_us + b and recover an alignment
  residual it can actually report, instead of silently inheriting USB
  scheduling jitter. Do not remove these without updating the web app.
*/

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "SparkFun_LSM6DSV16X.h"
#include "bima_boot_frames.h"

constexpr uint32_t SERIAL_BAUD = 921600;

// 400 kHz. The 2026-08-03 handoff set 100 kHz as a deliberate safety margin
// for the hand-crimped harnesses, and warned not to raise it without testing
// signal integrity on the installed cables. That test is now built in rather
// than assumed: every getAccel/getGyro return code is checked and any failure
// is counted per IMU and reported on the STATS line. A clean run at 400 kHz is
// evidence the harnesses tolerate it; a non-zero error count is the signal to
// drop back to 100000 here.
constexpr uint32_t I2C_CLOCK_HZ = 400000;

// The four-IMU read cycle is the bottleneck, not the ODR and not the USB link.
// SAMPLE_PERIOD_US is set from the measured cycle time so that one read yields
// exactly one fresh sample: sampling faster than the ODR duplicates samples,
// and sampling slower decimates a signal whose anti-alias filter is set
// relative to the ODR, which aliases. Keep IMU_ODR and SAMPLE_PERIOD_US in
// step — that pairing is the whole point.
constexpr lsm6dsv16x_data_rate_t IMU_ODR = LSM6DSV16X_ODR_AT_480Hz;
constexpr uint32_t SAMPLE_PERIOD_US = 2083;   // 480 Hz
constexpr uint32_t LABELED_PERIOD_US = 10000; // 100 Hz is plenty for a chart
constexpr uint32_t STATS_PERIOD_US = 1000000;

// Worst-case bytes in one app record (4 IMU lines + T/N + blank). Serial.write
// blocks once the USB TX buffer fills, and a blocked write stalls the sampling
// loop itself — a slow host would silently corrupt the device's own timebase.
// Records are therefore skipped, and counted, rather than allowed to back up.
constexpr int APP_RECORD_MAX_BYTES = 360;
constexpr int LABELED_RECORD_MAX_BYTES = 340;
constexpr uint32_t DISPLAY_ANIMATION_PERIOD_MS = 700;

constexpr float MG_TO_G = 0.001f;
constexpr float MDPS_TO_DPS = 0.001f;
constexpr float G_TO_MS2 = 9.80665f;
constexpr float DPS_TO_RAD = 0.01745329252f;

constexpr uint8_t OLED_WIDTH = 128;
constexpr uint8_t OLED_HEIGHT = 64;
constexpr int8_t OLED_RESET = -1;
// Start at the supplied-logo reveal. Earlier flower/target frames are skipped.
constexpr uint8_t BIMA_BOOT_START_FRAME = 124;

Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire2, OLED_RESET);

SparkFun_LSM6DSV16X imu1;
SparkFun_LSM6DSV16X imu2;
SparkFun_LSM6DSV16X imu3;
SparkFun_LSM6DSV16X imu4;

SparkFun_LSM6DSV16X* const imus[] = {&imu1, &imu2, &imu3, &imu4};
TwoWire* const imuBuses[] = {&Wire, &Wire, &Wire1, &Wire1};
const uint8_t imuAddresses[] = {0x6B, 0x6A, 0x6B, 0x6A};
const char* const imuPinNames[] = {"18/19", "18/19", "17/16", "17/16"};

struct ImuReading {
  float ax, ay, az;   // g
  float gx, gy, gz;   // degrees/second
  // Microseconds from the start of this read cycle to the instant THIS IMU was
  // read. The four are on two buses and are read in sequence, so they are not
  // simultaneous — the last trails the first by most of a cycle. Recording the
  // offset makes each sensor's true sample time recoverable instead of
  // pretending all four share one instant.
  uint16_t offsetUs;
  bool valid;
};

// Monotonic device clock. micros() is 32-bit and wraps every ~71.6 minutes;
// accumulating deltas into 64 bits means the host never has to unwrap it, and
// a long session cannot produce a timestamp that goes backwards.
uint64_t deviceMicros = 0;
uint32_t lastMicrosRaw = 0;

struct BootScreenPoint {
  int16_t x;
  int16_t y;
  bool visible;
};

ImuReading readings[4] = {};
bool imuHealthy[4] = {};
bool displayHealthy = false;
uint8_t oledAddress = 0;
uint32_t lastSampleMicros = 0;
uint32_t lastLabeledMicros = 0;
uint32_t lastStatsMicros = 0;
uint32_t lastDisplayAnimationMillis = 0;
uint8_t displayDotCount = 1;

// Per-record device clock and counter, sampled once per read cycle so every
// stream in a record refers to the same instant.
uint64_t sampleDeviceUs = 0;
uint32_t sampleSeq = 0;

// Signal-integrity instrumentation for the 400 kHz bus. i2cErrors counts
// failed register reads per IMU; cycleMinUs/cycleMaxUs bound the read cost so
// SAMPLE_PERIOD_US can be justified from measurement rather than guessed.
uint32_t i2cErrors[4] = {};
uint32_t cycleMinUs = 0xFFFFFFFF;
uint32_t cycleMaxUs = 0;
uint32_t cycleSumUs = 0;
uint32_t cycleCount = 0;
uint32_t recordsThisWindow = 0;
// Records skipped because the USB TX buffer was full. Non-zero means the host
// could not keep up — the data is genuinely gone, and the sequence counter will
// show it, which is exactly why it is counted rather than hidden.
uint32_t serialDrops = 0;
int minWriteHeadroom = 32767;

enum class SystemState { Starting, Standby, Running, SensorError, DisplayError };
SystemState systemState = SystemState::Starting;

// Defined after the type declarations above: the .ino preprocessor inserts its
// generated prototypes immediately before the first function definition in the
// file, so a function placed above these types would push those prototypes out
// of scope of SystemState and BootScreenPoint.
uint64_t deviceClockUs() {
  const uint32_t now = micros();
  deviceMicros += static_cast<uint32_t>(now - lastMicrosRaw);
  lastMicrosRaw = now;
  return deviceMicros;
}

const char* stateText(SystemState state) {
  switch (state) {
    case SystemState::Starting: return "STARTING";
    case SystemState::Standby: return "STANDBY";
    case SystemState::Running: return "RUNNING";
    case SystemState::SensorError: return "SENSOR ERROR";
    case SystemState::DisplayError: return "DISPLAY ERROR";
  }
  return "UNKNOWN";
}

const char* displayStateText(SystemState state) {
  switch (state) {
    case SystemState::Starting: return "STARTING...";
    case SystemState::Standby: return "STANDBY...";
    case SystemState::Running: return "RUNNING...";
    case SystemState::SensorError: return "SENSOR ERROR";
    case SystemState::DisplayError: return "DISPLAY ERROR";
  }
  return "UNKNOWN";
}

void drawImuConnection(uint8_t imuIndex, int16_t y) {
  const bool connected = imuHealthy[imuIndex];

  if (connected) {
    display.fillCircle(3, y + 3, 3, SSD1306_WHITE);
  } else {
    display.drawCircle(3, y + 3, 3, SSD1306_WHITE);
    display.drawLine(1, y + 1, 5, y + 5, SSD1306_WHITE);
    display.drawLine(5, y + 1, 1, y + 5, SSD1306_WHITE);
  }

  display.setCursor(10, y);
  display.print("IMU");
  display.print(imuIndex + 1);
  display.println(connected ? " CONNECTED" : " NOT FOUND");
}

void drawState(int failedImu = -1) {
  if (!displayHealthy) return;
  (void)failedImu;

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);

  const bool largeHeader =
    systemState == SystemState::Standby ||
    systemState == SystemState::Running;

  display.setTextSize(largeHeader ? 2 : 1);
  if (largeHeader) {
    display.print(stateText(systemState));
    if (systemState == SystemState::Running) {
      for (uint8_t i = 0; i < displayDotCount; ++i) {
        display.print('.');
      }
    }
    display.println();
  } else {
    display.println(displayStateText(systemState));
  }
  display.setTextSize(1);

  const int16_t firstRowY = largeHeader ? 20 : 16;
  const int16_t rowSpacing = largeHeader ? 11 : 12;
  for (uint8_t i = 0; i < 4; ++i) {
    drawImuConnection(i, firstRowY + i * rowSpacing);
  }

  display.display();
}

void setSystemState(SystemState newState, int failedImu = -1) {
  if (systemState == newState && failedImu < 0) return;
  systemState = newState;
  displayDotCount = 1;
  lastDisplayAnimationMillis = millis();
  Serial.print("System state: ");
  Serial.println(stateText(systemState));
  drawState(failedImu);
}

void updateDisplayAnimation() {
  if (!displayHealthy) return;
  if (systemState != SystemState::Running) return;

  const uint32_t now = millis();
  if (static_cast<uint32_t>(now - lastDisplayAnimationMillis) <
      DISPLAY_ANIMATION_PERIOD_MS) return;

  lastDisplayAnimationMillis = now;
  displayDotCount = displayDotCount % 3 + 1;
  drawState();
}

void drawBimaLogo(bool showName, bool showSubtitle) {
  // Biometric pulse mark.
  display.drawCircle(64, 13, 10, SSD1306_WHITE);
  display.drawLine(54, 13, 58, 13, SSD1306_WHITE);
  display.drawLine(58, 13, 61, 7, SSD1306_WHITE);
  display.drawLine(61, 7, 64, 19, SSD1306_WHITE);
  display.drawLine(64, 19, 68, 10, SSD1306_WHITE);
  display.drawLine(68, 10, 71, 13, SSD1306_WHITE);
  display.drawLine(71, 13, 74, 13, SSD1306_WHITE);

  if (showName) {
    display.setTextSize(2);
    display.setCursor(40, 25);
    display.print("BIMA");
  }

  if (showSubtitle) {
    display.setTextSize(1);
    display.setCursor(16, 44);
    display.print("BIOMETRIC INFANT");
    display.setCursor(7, 54);
    display.print("MOVEMENT ASSESSMENT");
  }
}

BootScreenPoint projectBootPoint(
  float x,
  float y,
  float z,
  float yaw,
  float pitch,
  float zoom
) {
  const float cosYaw = cosf(yaw);
  const float sinYaw = sinf(yaw);
  const float cosPitch = cosf(pitch);
  const float sinPitch = sinf(pitch);

  const float yawX = x * cosYaw - z * sinYaw;
  const float yawZ = x * sinYaw + z * cosYaw;
  const float pitchY = y * cosPitch - yawZ * sinPitch;
  const float pitchZ = y * sinPitch + yawZ * cosPitch;
  const float depth = pitchZ + 4.2f;

  if (depth <= 0.2f) return {0, 0, false};

  const float scale = 32.0f * zoom / depth;
  return {
    static_cast<int16_t>(64.0f + yawX * scale),
    static_cast<int16_t>(31.0f + pitchY * scale),
    true
  };
}

void drawPerspectiveGrid(uint8_t frame) {
  const int16_t horizonY = 25;
  const int16_t vanishingX = 64 + static_cast<int16_t>(sinf(frame * 0.05f) * 4.0f);

  display.drawFastHLine(0, horizonY, OLED_WIDTH, SSD1306_WHITE);
  for (int16_t bottomX = -16; bottomX <= 144; bottomX += 16) {
    display.drawLine(vanishingX, horizonY, bottomX, 63, SSD1306_WHITE);
  }

  for (uint8_t line = 0; line < 7; ++line) {
    const uint16_t phaseValue = (line * 17 + frame * 3) % 119;
    const float depth = phaseValue / 119.0f;
    const int16_t y = horizonY + static_cast<int16_t>(depth * depth * 39.0f);
    display.drawFastHLine(0, y, OLED_WIDTH, SSD1306_WHITE);
  }
}

void drawSensorFusionCube(float yaw, float pitch, float zoom, uint8_t frame) {
  static const float vertices[8][3] = {
    {-1, -1, -1}, { 1, -1, -1}, { 1,  1, -1}, {-1,  1, -1},
    {-1, -1,  1}, { 1, -1,  1}, { 1,  1,  1}, {-1,  1,  1}
  };
  static const uint8_t edges[12][2] = {
    {0,1}, {1,2}, {2,3}, {3,0},
    {4,5}, {5,6}, {6,7}, {7,4},
    {0,4}, {1,5}, {2,6}, {3,7}
  };

  BootScreenPoint projected[8];
  for (uint8_t i = 0; i < 8; ++i) {
    projected[i] = projectBootPoint(
      vertices[i][0], vertices[i][1], vertices[i][2], yaw, pitch, zoom
    );
  }

  for (uint8_t i = 0; i < 12; ++i) {
    const BootScreenPoint& a = projected[edges[i][0]];
    const BootScreenPoint& b = projected[edges[i][1]];
    if (a.visible && b.visible) {
      display.drawLine(a.x, a.y, b.x, b.y, SSD1306_WHITE);
    }
  }

  for (uint8_t i = 0; i < 8; ++i) {
    if (projected[i].visible) {
      display.fillCircle(projected[i].x, projected[i].y, 1, SSD1306_WHITE);
    }
  }

  const BootScreenPoint core = projectBootPoint(0, 0, 0, yaw, pitch, zoom);
  if (!core.visible) return;

  // Four IMU nodes orbit the fusion core in real 3D.
  for (uint8_t imu = 0; imu < 4; ++imu) {
    const float angle = yaw * 1.35f + imu * 1.5707963f;
    const float nodeX = cosf(angle) * 1.75f;
    const float nodeY = sinf(angle * 1.7f + imu) * 0.48f;
    const float nodeZ = sinf(angle) * 1.75f;
    const BootScreenPoint node = projectBootPoint(nodeX, nodeY, nodeZ, yaw * 0.2f, pitch, zoom);
    if (node.visible) {
      display.drawLine(core.x, core.y, node.x, node.y, SSD1306_WHITE);
      if ((frame + imu) % 5 < 3) {
        display.fillCircle(node.x, node.y, 2, SSD1306_WHITE);
      } else {
        display.drawCircle(node.x, node.y, 2, SSD1306_WHITE);
      }
    }
  }

  // A pulsing biometric core lives inside the cube.
  const uint8_t pulseRadius = 2 + ((frame / 3) % 3);
  display.drawCircle(core.x, core.y, pulseRadius, SSD1306_WHITE);
  display.fillCircle(core.x, core.y, 1, SSD1306_WHITE);
}

void playProceduralBootLogoAnimation() {
  if (!displayHealthy) return;
  display.setTextColor(SSD1306_WHITE);

  struct BootStar { int16_t x, y, z; };
  BootStar stars[24];
  randomSeed(0xB1A4);
  for (BootStar& star : stars) {
    star.x = random(-70, 71);
    star.y = random(-40, 41);
    star.z = random(18, 110);
  }

  // Scene 1: accelerate through a biometric data field.
  for (uint8_t frame = 0; frame < 30; ++frame) {
    display.clearDisplay();
    for (BootStar& star : stars) {
      star.z -= 4;
      if (star.z < 7) {
        star.x = random(-70, 71);
        star.y = random(-40, 41);
        star.z = 108;
      }

      const int16_t x = 64 + star.x * 58 / star.z;
      const int16_t y = 30 + star.y * 58 / star.z;
      const int16_t tailX = 64 + star.x * 58 / (star.z + 8);
      const int16_t tailY = 30 + star.y * 58 / (star.z + 8);
      display.drawLine(tailX, tailY, x, y, SSD1306_WHITE);
      if (star.z < 36) display.drawPixel(x + 1, y, SSD1306_WHITE);
    }

    const uint8_t reticleRadius = 4 + (frame % 10);
    display.drawCircle(64, 30, reticleRadius, SSD1306_WHITE);
    display.drawFastHLine(52, 30, 25, SSD1306_WHITE);
    display.drawFastVLine(64, 18, 25, SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(8, 54);
    display.print("BIMA / BOOT VECTOR");
    display.display();
    delay(24);
  }

  // Scene 2: enter the 3D sensor-fusion chamber.
  for (uint8_t frame = 0; frame < 50; ++frame) {
    display.clearDisplay();
    drawPerspectiveGrid(frame);
    drawSensorFusionCube(frame * 0.075f, -0.28f + sinf(frame * 0.05f) * 0.12f, 1.05f, frame);

    display.fillRect(0, 0, OLED_WIDTH, 9, SSD1306_BLACK);
    display.setTextSize(1);
    display.setCursor(20, 0);
    display.print("BIMA SENSOR FUSION");

    // Four telemetry lights represent the four IMU channels.
    for (uint8_t imu = 0; imu < 4; ++imu) {
      const int16_t x = 49 + imu * 10;
      if ((frame + imu * 2) % 12 < 8) display.fillCircle(x, 61, 2, SSD1306_WHITE);
      else display.drawCircle(x, 61, 2, SSD1306_WHITE);
    }

    display.display();
    delay(24);
  }

  // Scene 3: the camera dives through the fusion core.
  for (uint8_t frame = 0; frame < 18; ++frame) {
    display.clearDisplay();
    drawPerspectiveGrid(frame + 50);
    const float zoom = 1.1f + frame * frame * 0.012f;
    drawSensorFusionCube(3.75f + frame * 0.11f, -0.22f, zoom, frame + 50);
    display.drawCircle(64, 31, 3 + frame * 2, SSD1306_WHITE);
    display.display();
    delay(26);
  }

  display.invertDisplay(true);
  delay(70);
  display.invertDisplay(false);

  // Scene 4: the BIMA hologram materializes from the scan plane upward.
  for (uint8_t frame = 0; frame < 12; ++frame) {
    display.clearDisplay();
    drawBimaLogo(true, true);
    const int16_t scanY = 63 - frame * 6;
    if (scanY > 0) display.fillRect(0, 0, OLED_WIDTH, scanY, SSD1306_BLACK);
    if (scanY >= 0 && scanY < OLED_HEIGHT) {
      display.drawFastHLine(0, scanY, OLED_WIDTH, SSD1306_WHITE);
      if (scanY + 2 < OLED_HEIGHT) {
        for (int16_t x = frame % 4; x < OLED_WIDTH; x += 8) {
          display.drawPixel(x, scanY + 2, SSD1306_WHITE);
        }
      }
    }
    display.display();
    delay(42);
  }

  display.clearDisplay();
  drawBimaLogo(true, true);
  display.display();
  delay(950);
}

void playBootLogoAnimation() {
  if (!displayHealthy) return;

  // These frames are rendered at 4x resolution, antialiased, then reduced to
  // clean 1-bit artwork for the SSD1306. They live in Teensy program flash.
  for (uint8_t frame = BIMA_BOOT_START_FRAME;
       frame < BIMA_BOOT_FRAME_COUNT;
       ++frame) {
    display.clearDisplay();
    display.drawBitmap(
      0,
      0,
      BIMA_BOOT_FRAMES[frame],
      OLED_WIDTH,
      OLED_HEIGHT,
      SSD1306_WHITE
    );
    display.display();
    delay(18);
  }
}

void scanI2CBus(TwoWire& bus, const char* busName) {
  Serial.print("\n");
  Serial.print(busName);
  Serial.println(":");
  uint8_t found = 0;
  for (uint8_t address = 1; address < 127; ++address) {
    bus.beginTransmission(address);
    const uint8_t error = bus.endTransmission();
    if (error == 0) {
      Serial.print("  Device at 0x");
      if (address < 16) Serial.print('0');
      Serial.println(address, HEX);
      ++found;
    }
  }
  if (found == 0) Serial.println("  No devices found");
}

bool addressResponds(TwoWire& bus, uint8_t address) {
  bus.beginTransmission(address);
  return bus.endTransmission() == 0;
}

void scanAllBuses() {
  scanI2CBus(Wire, "Wire");
  scanI2CBus(Wire1, "Wire1");
  scanI2CBus(Wire2, "Wire2");
  Serial.println();
}

void initializeDisplay() {
  if (addressResponds(Wire2, 0x3C)) oledAddress = 0x3C;
  else if (addressResponds(Wire2, 0x3D)) oledAddress = 0x3D;

  if (oledAddress == 0 || !display.begin(SSD1306_SWITCHCAPVCC, oledAddress, true, false)) {
    Serial.println("ERROR: OLED not detected on Wire2 at 0x3C or 0x3D");
    displayHealthy = false;
    return;
  }
  displayHealthy = true;
  Serial.print("OLED found at 0x");
  Serial.println(oledAddress, HEX);
  Serial.println("Playing BIMA boot animation");
  playBootLogoAnimation();
}

bool initializeImus() {
  bool allHealthy = true;
  for (uint8_t i = 0; i < 4; ++i) {
    SparkFun_LSM6DSV16X* imu = imus[i];

    imuHealthy[i] = imu->begin(*imuBuses[i], imuAddresses[i]);
    if (imuHealthy[i]) {
      imu->deviceReset();
      const uint32_t resetStart = millis();
      while (!imu->getDeviceReset() && millis() - resetStart < 500) delay(1);

      // Registers hold their value until read, so a sample is never a mix of
      // pre- and post-update bytes.
      imu->enableBlockDataUpdate();

      imu->setAccelDataRate(IMU_ODR);
      imu->setAccelFullScale(LSM6DSV16X_16g);
      imu->setGyroDataRate(IMU_ODR);
      imu->setGyroFullScale(LSM6DSV16X_2000dps);

      imu->enableFilterSettling();
      imu->enableAccelLP2Filter();
      imu->setAccelLP2Bandwidth(LSM6DSV16X_XL_STRONG);
      imu->enableGyroLP1Filter();
      imu->setGyroLP1Bandwidth(LSM6DSV16X_GY_ULTRA_LIGHT);
    }

    Serial.print("IMU");
    Serial.print(i + 1);
    Serial.print(" (pins ");
    Serial.print(imuPinNames[i]);
    Serial.print(", address 0x");
    Serial.print(imuAddresses[i], HEX);
    Serial.println(imuHealthy[i] ? "): FOUND" : "): NOT FOUND");

    if (!imuHealthy[i]) allHealthy = false;
  }

  if (allHealthy) {
    Serial.println("All sensors initialized");
  } else {
    Serial.println("WARNING: One or more IMUs are not connected");
  }

  // Connection health is shown per IMU. Missing sensors do not replace the
  // primary STANDBY/RUNNING mode heading on the OLED.
  setSystemState(SystemState::Standby);
  return allHealthy;
}

void readAllImus() {
  const uint32_t cycleStart = micros();
  sampleDeviceUs = deviceClockUs();
  ++sampleSeq;

  sfe_lsm_data_t accel;
  sfe_lsm_data_t gyro;

  for (uint8_t i = 0; i < 4; ++i) {
    readings[i].valid = false;
    readings[i].offsetUs = static_cast<uint16_t>(micros() - cycleStart);
    if (!imuHealthy[i]) continue;

    // Return codes are checked, not discarded: at 400 kHz on hand-crimped
    // harnesses a marginal connection shows up as a failed read, and a failed
    // read must invalidate the sample rather than silently repeat the last one.
    const bool okAccel = imus[i]->getAccel(&accel);
    const bool okGyro = imus[i]->getGyro(&gyro);
    if (!okAccel || !okGyro) { ++i2cErrors[i]; continue; }

    // The library reports acceleration in mg and rotation in mdps.
    readings[i].ax = accel.xData * MG_TO_G;
    readings[i].ay = accel.yData * MG_TO_G;
    readings[i].az = accel.zData * MG_TO_G;
    readings[i].gx = gyro.xData * MDPS_TO_DPS;
    readings[i].gy = gyro.yData * MDPS_TO_DPS;
    readings[i].gz = gyro.zData * MDPS_TO_DPS;
    readings[i].valid = true;
  }

  const uint32_t elapsed = micros() - cycleStart;
  if (elapsed < cycleMinUs) cycleMinUs = elapsed;
  if (elapsed > cycleMaxUs) cycleMaxUs = elapsed;
  cycleSumUs += elapsed;
  ++cycleCount;
}

// Windows viewer format: 24 tab-separated label:value fields on one line.
void printLabeledStream() {
  for (uint8_t i = 0; i < 4; ++i) {
    const char index = static_cast<char>('1' + i);
    Serial.print('I'); Serial.print(index); Serial.print("_AX:"); Serial.print(readings[i].ax, 3); Serial.print('\t');
    Serial.print('I'); Serial.print(index); Serial.print("_AY:"); Serial.print(readings[i].ay, 3); Serial.print('\t');
    Serial.print('I'); Serial.print(index); Serial.print("_AZ:"); Serial.print(readings[i].az, 3); Serial.print('\t');
    Serial.print('I'); Serial.print(index); Serial.print("_GX:"); Serial.print(readings[i].gx, 2); Serial.print('\t');
    Serial.print('I'); Serial.print(index); Serial.print("_GY:"); Serial.print(readings[i].gy, 2); Serial.print('\t');
    Serial.print('I'); Serial.print(index); Serial.print("_GZ:"); Serial.print(readings[i].gz, 2);
    if (i < 3) Serial.print('\t');
  }
  Serial.println();
}

// BIMA web-app format. Units are m/s^2 and rad/s to match the calibration
// records captured against the earlier ICM-20948 firmware.
//
// The record ends with a `T:`/`N:` line carrying the device clock and sample
// counter, then a blank line. The blank line is the app's per-sample record
// boundary — do not remove it, and keep T/N immediately before it so the host
// stamps the row it actually belongs to.
void printAppStream() {
  for (uint8_t i = 0; i < 4; ++i) {
    Serial.print("IMU"); Serial.print(i + 1);
    if (!readings[i].valid) {
      Serial.println(" INVALID");
      continue;
    }
    Serial.print(" A: ");
    Serial.print(readings[i].ax * G_TO_MS2, 3); Serial.print(", ");
    Serial.print(readings[i].ay * G_TO_MS2, 3); Serial.print(", ");
    Serial.print(readings[i].az * G_TO_MS2, 3); Serial.print(" m/s^2 | G: ");
    Serial.print(readings[i].gx * DPS_TO_RAD, 3); Serial.print(", ");
    Serial.print(readings[i].gy * DPS_TO_RAD, 3); Serial.print(", ");
    Serial.print(readings[i].gz * DPS_TO_RAD, 3); Serial.println(" rad/s");
  }
  // T is the cycle start on the device clock; O gives each IMU's offset from it
  // in microseconds, so IMU n's true sample time is T + O[n].
  Serial.print("T: "); Serial.print(sampleDeviceUs);
  Serial.print(" N: "); Serial.print(sampleSeq);
  Serial.print(" O: ");
  for (uint8_t i = 0; i < 4; ++i) { Serial.print(readings[i].offsetUs); if (i < 3) Serial.print(','); }
  Serial.println();
  Serial.println();
  ++recordsThisWindow;
}

// Once a second, on its own line. Lets the host — and a person watching the
// serial monitor — see the achieved rate and, critically, whether the 400 kHz
// bus is producing any read failures.
void printStats(uint32_t windowUs) {
  const uint32_t avg = cycleCount ? cycleSumUs / cycleCount : 0;
  uint32_t totalErrors = 0;
  for (uint8_t i = 0; i < 4; ++i) totalErrors += i2cErrors[i];

  Serial.print("STATS hz:");
  Serial.print(windowUs ? (recordsThisWindow * 1000000.0f / windowUs) : 0.0f, 1);
  Serial.print(" cycle_us_min:"); Serial.print(cycleMinUs == 0xFFFFFFFF ? 0 : cycleMinUs);
  Serial.print(" cycle_us_avg:"); Serial.print(avg);
  Serial.print(" cycle_us_max:"); Serial.print(cycleMaxUs);
  Serial.print(" i2c_err:");
  for (uint8_t i = 0; i < 4; ++i) { Serial.print(i2cErrors[i]); if (i < 3) Serial.print(','); }
  Serial.print(" i2c_err_total:"); Serial.print(totalErrors);
  Serial.print(" i2c_hz:"); Serial.print(I2C_CLOCK_HZ);
  Serial.print(" serial_drops:"); Serial.print(serialDrops);
  Serial.print(" tx_headroom_min:"); Serial.print(minWriteHeadroom);
  Serial.print(" seq:"); Serial.println(sampleSeq);

  cycleMinUs = 0xFFFFFFFF; cycleMaxUs = 0; cycleSumUs = 0; cycleCount = 0;
  recordsThisWindow = 0;
  minWriteHeadroom = 32767;
}

void handleSerialCommands() {
  while (Serial.available()) {
    switch (Serial.read()) {
      case 'r': case 'R':
        setSystemState(SystemState::Running);
        break;
      case 's': case 'S':
        setSystemState(SystemState::Standby);
        break;
      case 'i': case 'I': scanAllBuses(); break;
      case 'd': case 'D': printStats(STATS_PERIOD_US); break;
    }
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  const uint32_t serialWaitStart = millis();
  while (!Serial && millis() - serialWaitStart < 3000) {}
  Serial.println();
  Serial.println("Four LSM6DSV16X IMU test");
  Serial.println("Starting sensors...");

  Wire.begin();
  Wire1.begin();
  Wire2.begin();
  Wire.setClock(I2C_CLOCK_HZ);
  Wire1.setClock(I2C_CLOCK_HZ);
  Wire2.setClock(I2C_CLOCK_HZ);

  initializeDisplay();
  initializeImus();

  // Reapply AFTER both init paths. SparkFun's begin() calls Wire.begin()
  // internally (sfe_bus QwI2C::init with bInit true), and Adafruit_SSD1306's
  // begin() does the same for Wire2 — both reset the bus to the 100 kHz
  // default and silently discard the setClock above. Setting it here is what
  // actually puts the buses at 400 kHz; without this the STATS line reports a
  // 400000 that the hardware never saw.
  Wire.setClock(I2C_CLOCK_HZ);
  Wire1.setClock(I2C_CLOCK_HZ);
  Wire2.setClock(I2C_CLOCK_HZ);

  lastMicrosRaw = micros();
  lastSampleMicros = lastMicrosRaw;
  lastLabeledMicros = lastMicrosRaw;
  lastStatsMicros = lastMicrosRaw;
}

void loop() {
  handleSerialCommands();
  updateDisplayAnimation();

  const uint32_t now = micros();

  // One read produces one record. The app stream is emitted every sample
  // because it is the one that becomes study data; the viewer stream is
  // decimated to a display rate so it cannot steal cycles from acquisition.
  if (static_cast<uint32_t>(now - lastSampleMicros) >= SAMPLE_PERIOD_US) {
    lastSampleMicros = now;
    readAllImus();

    // Emit only if the record fits in the free TX space. Sampling continues
    // regardless, so the device clock stays regular and a host stall costs
    // records — visibly, via the sequence counter — instead of corrupting the
    // timebase for everything that follows.
    const int headroom = Serial.availableForWrite();
    if (headroom < minWriteHeadroom) minWriteHeadroom = headroom;
    if (headroom >= APP_RECORD_MAX_BYTES) {
      printAppStream();
      if (static_cast<uint32_t>(now - lastLabeledMicros) >= LABELED_PERIOD_US
          && Serial.availableForWrite() >= LABELED_RECORD_MAX_BYTES) {
        lastLabeledMicros = now;
        printLabeledStream();
      }
    } else {
      ++serialDrops;
    }
  }

  if (static_cast<uint32_t>(now - lastStatsMicros) >= STATS_PERIOD_US) {
    printStats(static_cast<uint32_t>(now - lastStatsMicros));
    lastStatsMicros = now;
  }
}
