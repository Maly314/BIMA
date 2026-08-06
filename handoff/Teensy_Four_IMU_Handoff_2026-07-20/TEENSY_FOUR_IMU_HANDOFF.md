# Codex Project Handoff: Teensy 4.1 Four-IMU System

## Current status

The Teensy 4.1 is connected and has working diagnostic firmware installed.

- Arduino IDE 2.3.10 is installed.
- Teensy 4.1 board support is installed and selected.
- USB Type is set to `Serial`.
- The diagnostic sketch compiles and uploads successfully.
- Teensy Loader reported `Reboot OK` after programming.
- Windows assigned the Teensy to `COM8` during testing. The COM number may change after reconnecting it.
- Serial Monitor is configured for `115200` baud.
- No OLED display is currently connected.
- Two ICM-20948 boards work correctly.
- Two ICM-20948 boards appear to be defective or have failed I2C interfaces.

## Moving this project to another PC

This handoff is packaged with the complete Arduino sketch. On the destination PC:

1. Extract the ZIP file to a normal folder such as `Documents\Teensy_Four_IMU_Handoff`.
2. Install Arduino IDE 2.x from `https://www.arduino.cc/en/software`.
3. In Arduino IDE, open **File > Preferences** and add this Boards Manager URL:

   ```text
   https://www.pjrc.com/teensy/package_teensy_index.json
   ```

4. Open Boards Manager, search for `Teensy`, and install **Teensy by Paul Stoffregen**.
5. Install the Arduino libraries listed below using Library Manager.
6. Open the included folder `teensy_four_imu_test` and its file `teensy_four_imu_test.ino`.
7. Select **Teensy 4.1** and set **USB Type** to **Serial**.
8. Compile and upload. If Teensy Loader waits, press the Teensy Program button once briefly.
9. Select the COM port assigned on the new PC and open Serial Monitor at `115200` baud.

The COM port will probably not be `COM8` on the destination PC. Select whichever port appears as the Teensy after upload.

## Original-PC file locations

Diagnostic sketch:

```text
E:\Codex stuffs\teensy_four_imu_test\teensy_four_imu_test.ino
```

This handoff:

```text
E:\Codex stuffs\TEENSY_FOUR_IMU_HANDOFF.md
```

Arduino IDE executable:

```text
E:\Codeing stuff\Arduino IDE\Arduino IDE.exe
```

These absolute paths apply only to the original PC. Use the files included in the portable ZIP on the destination PC.

## Installed/required Arduino libraries

- Adafruit ICM20X
- Adafruit Unified Sensor
- Adafruit BusIO
- Adafruit GFX Library
- Adafruit SSD1306

The OLED libraries are required to compile the current sketch even though no OLED is connected.

## Confirmed Teensy wiring

All devices use Teensy `3.3V` and `GND`.

| IMU | SDA | SCL | Expected address |
|---|---:|---:|---:|
| IMU 1 | Teensy pin 18 | Teensy pin 19 | `0x69`, ADR jumper open |
| IMU 2 | Teensy pin 18 | Teensy pin 19 | `0x68`, ADR jumper bridged |
| IMU 3 | Teensy pin 17 | Teensy pin 16 | `0x69`, ADR jumper open |
| IMU 4 | Teensy pin 17 | Teensy pin 16 | `0x68`, ADR jumper bridged |

Power for every IMU:

```text
IMU VIN -> Teensy 3.3V
IMU GND -> Teensy GND
```

Typical STEMMA QT colors:

```text
Red    = VIN / 3.3V
Black  = GND
Blue   = SDA
Yellow = SCL
```

Trust the printed board labels over wire colors if a cable differs.

The two STEMMA QT connectors on an Adafruit ICM-20948 board are electrically equivalent. Either connector may be used as the input, and the other may daisy-chain to another sensor.

## Confirmed test results

The scanner produced:

```text
Wire:
  No devices found

Wire1:
  Device at 0x68
  Device at 0x69

Wire2:
  No devices found
```

Serial readings showed:

```text
IMU1 INVALID
IMU2 INVALID
IMU3 A: live values ... | G: live values ...
IMU4 A: live values ... | G: live values ...
```

The user then tested boards individually. The two working boards operated on the other Teensy pins/buses, including pins 18/19. This confirms:

- Teensy pins 18/19 work.
- Teensy pins 17/16 work.
- The firmware and I2C scanners work.
- The known-good cables/connections work.
- The failure follows two specific IMU boards rather than a Teensy bus.

The user reported that two boards work everywhere tested and two boards do not work.

## Final verification before replacing boards

For each suspected faulty board:

1. Unplug Teensy USB power.
2. Connect only that IMU to the known-good pins 18/19 with the known-good cable:
   - `VIN -> 3.3V`
   - `GND -> GND`
   - `SDA -> pin 18`
   - `SCL -> pin 19`
3. Reconnect USB.
4. Open Serial Monitor at `115200` baud.
5. Enter `i` and press Enter.
6. Try both STEMMA QT connectors on the board, powering off before moving the cable.
7. Look for either address `0x68` or `0x69` under `Wire`.

If a known-good IMU appears immediately using the same cable and pins, but the suspected board reports `No devices found` through both connectors, treat that board as faulty and replace it.

## Diagnostic sketch behavior

The current sketch:

- Initializes `Wire`, `Wire1`, and `Wire2` at 400 kHz.
- Scans all three I2C buses during startup.
- Treats the absent OLED as a nonfatal error.
- Initializes four separate ICM-20948 objects.
- Samples sensors at 100 Hz.
- Prints readings at 100 Hz over 921600-baud serial so fine tremor content is not discarded.
- Prints `INVALID` for sensors that failed initialization.
- Uses nonblocking scheduling in the main loop.

Serial commands:

```text
i = scan all I2C buses
r = change state to RUNNING
s = change state to STANDBY
```

## Restart procedure

To restart the Teensy safely:

1. Close Serial Monitor.
2. Unplug the USB cable.
3. Wait about five seconds.
4. Reconnect USB.
5. Select the Teensy COM port again in Arduino IDE if necessary.
6. Open Serial Monitor at `115200` baud.

Do not hold the white Teensy Program button. A long hold can start the Teensy factory-restore process. A normal upload may require pressing it once briefly when Teensy Loader is waiting.

## Recommended next steps

1. Complete the final individual-board verification described above.
2. Replace the two ICM-20948 boards if they remain undetectable.
3. Connect all four working boards using two addresses per bus:
   - `Wire`: `0x68` and `0x69`
   - `Wire1`: `0x68` and `0x69`
4. Run the `i` scan and confirm all four addresses.
5. Confirm live readings from IMU1 through IMU4.
6. Add the SSD1306 OLED later on `Wire2`, pins 25/24, at `0x3C` or `0x3D`.
7. After all hardware works, refactor the single-file sketch into separate IMU, display, and system-state modules.

## Expected fully working scan

```text
Wire:
  Device at 0x68
  Device at 0x69

Wire1:
  Device at 0x68
  Device at 0x69

Wire2:
  No devices found
```

The empty `Wire2` result is expected until the OLED is installed.
