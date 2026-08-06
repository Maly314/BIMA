/*
  Teensy 4.1 + four Adafruit ICM-20948 IMUs + SSD1306 OLED

  Wiring (all devices use Teensy 3.3 V and GND):
    IMU 1: Wire,  SDA pin 18, SCL pin 19, address 0x69 (ADR open)
    IMU 2: Wire,  SDA pin 18, SCL pin 19, address 0x68 (ADR bridged)
    IMU 3: Wire1, SDA pin 17, SCL pin 16, address 0x69 (ADR open)
    IMU 4: Wire1, SDA pin 17, SCL pin 16, address 0x68 (ADR bridged)
    OLED:  Wire2, SDA pin 25, SCL pin 24, address 0x3C (0x3D fallback)

  Serial commands: r = RUNNING, s = STANDBY, i = rescan buses
*/

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_Sensor.h>
#include "bima_boot_frames.h"

constexpr uint32_t SERIAL_BAUD = 921600;
constexpr uint32_t I2C_CLOCK_HZ = 400000;
constexpr uint32_t SENSOR_PERIOD_US = 10000;   // 100 Hz
constexpr uint32_t SERIAL_PERIOD_US = 10000;  // 100 Hz; preserves fine tremor content
constexpr uint32_t DISPLAY_ANIMATION_PERIOD_MS = 700;

constexpr uint8_t OLED_WIDTH = 128;
constexpr uint8_t OLED_HEIGHT = 64;
constexpr int8_t OLED_RESET = -1;
// Start at the supplied-logo reveal. Earlier flower/target frames are skipped.
constexpr uint8_t BIMA_BOOT_START_FRAME = 124;

Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire2, OLED_RESET);
Adafruit_ICM20948 imu1;
Adafruit_ICM20948 imu2;
Adafruit_ICM20948 imu3;
Adafruit_ICM20948 imu4;

Adafruit_ICM20948* const imus[] = {&imu1, &imu2, &imu3, &imu4};
TwoWire* const imuBuses[] = {&Wire, &Wire, &Wire1, &Wire1};
const uint8_t imuAddresses[] = {0x69, 0x68, 0x69, 0x68};
const char* const imuBusNames[] = {"Wire", "Wire", "Wire1", "Wire1"};

struct ImuReading {
  uint32_t timestampMicros;
  float ax, ay, az;
  float gx, gy, gz;
  float temperatureC;
  bool valid;
};

struct BootScreenPoint {
  int16_t x;
  int16_t y;
  bool visible;
};

ImuReading readings[4] = {};
bool imuHealthy[4] = {};
bool displayHealthy = false;
uint8_t oledAddress = 0;
uint32_t lastSensorMicros = 0;
uint32_t lastSerialMicros = 0;
uint32_t lastDisplayAnimationMillis = 0;
uint8_t displayDotCount = 1;

enum class SystemState { Starting, Standby, Running, SensorError, DisplayError };
SystemState systemState = SystemState::Starting;

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
  Serial.print("OLED found at Wire2 address 0x");
  Serial.println(oledAddress, HEX);
  Serial.println("Playing BIMA boot animation");
  playBootLogoAnimation();
}

bool initializeImus() {
  bool allHealthy = true;
  for (uint8_t i = 0; i < 4; ++i) {
    imuHealthy[i] = imus[i]->begin_I2C(imuAddresses[i], imuBuses[i]);
    if (imuHealthy[i]) {
      imus[i]->setAccelRange(ICM20948_ACCEL_RANGE_16_G);
      imus[i]->setGyroRange(ICM20948_GYRO_RANGE_2000_DPS);
      Serial.print("IMU "); Serial.print(i + 1);
      Serial.print(" found on "); Serial.print(imuBusNames[i]);
      Serial.print(" at 0x"); Serial.println(imuAddresses[i], HEX);
    } else {
      allHealthy = false;
      Serial.print("ERROR: IMU "); Serial.print(i + 1);
      Serial.print(" not detected on "); Serial.print(imuBusNames[i]);
      Serial.print(" at 0x"); Serial.println(imuAddresses[i], HEX);
    }
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
  const uint32_t cycleTimestamp = micros();
  for (uint8_t i = 0; i < 4; ++i) {
    readings[i].timestampMicros = cycleTimestamp;
    readings[i].valid = imuHealthy[i];
    if (!imuHealthy[i]) continue;
    sensors_event_t accel, gyro, mag, temp;
    imus[i]->getEvent(&accel, &gyro, &temp, &mag);
    readings[i].ax = accel.acceleration.x;
    readings[i].ay = accel.acceleration.y;
    readings[i].az = accel.acceleration.z;
    readings[i].gx = gyro.gyro.x;
    readings[i].gy = gyro.gyro.y;
    readings[i].gz = gyro.gyro.z;
    readings[i].temperatureC = temp.temperature;
  }
}

void printReadings() {
  for (uint8_t i = 0; i < 4; ++i) {
    Serial.print("IMU"); Serial.print(i + 1);
    if (!readings[i].valid) {
      Serial.println(" INVALID");
      continue;
    }
    Serial.print(" A: ");
    Serial.print(readings[i].ax, 3); Serial.print(", ");
    Serial.print(readings[i].ay, 3); Serial.print(", ");
    Serial.print(readings[i].az, 3); Serial.print(" m/s^2 | G: ");
    Serial.print(readings[i].gx, 3); Serial.print(", ");
    Serial.print(readings[i].gy, 3); Serial.print(", ");
    Serial.print(readings[i].gz, 3); Serial.println(" rad/s");
  }
  Serial.println();
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
    }
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  const uint32_t serialWaitStart = millis();
  while (!Serial && millis() - serialWaitStart < 3000) {}
  Serial.println("Starting system...");

  Wire.begin();
  Wire1.begin();
  Wire2.begin();
  Wire.setClock(I2C_CLOCK_HZ);
  Wire1.setClock(I2C_CLOCK_HZ);
  Wire2.setClock(I2C_CLOCK_HZ);

  scanAllBuses();
  initializeDisplay();
  initializeImus();
  lastSensorMicros = micros();
  lastSerialMicros = micros();
}

void loop() {
  handleSerialCommands();
  updateDisplayAnimation();
  const uint32_t now = micros();
  if (static_cast<uint32_t>(now - lastSensorMicros) >= SENSOR_PERIOD_US) {
    lastSensorMicros += SENSOR_PERIOD_US;
    readAllImus();
  }
  if (static_cast<uint32_t>(now - lastSerialMicros) >= SERIAL_PERIOD_US) {
    lastSerialMicros += SERIAL_PERIOD_US;
    printReadings();
  }
}
