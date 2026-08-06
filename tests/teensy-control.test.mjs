import assert from "node:assert/strict";
import test from "node:test";

import { parseTeensyDisplayState, writeTeensyDisplayState } from "../app/teensy-control.ts";

test("parses the state acknowledgements emitted by the Teensy firmware", () => {
  assert.equal(parseTeensyDisplayState("System state: RUNNING"), "running");
  assert.equal(parseTeensyDisplayState("System state: STANDBY\r"), "standby");
  assert.equal(parseTeensyDisplayState("IMU1 A: 0, 0, 9.8"), "unknown");
});

test("writes the firmware's single-byte running and standby commands", async () => {
  const writes = [];
  let releases = 0;
  const port = {
    writable: {
      getWriter() {
        return {
          async write(data) { writes.push(new TextDecoder().decode(data)); },
          releaseLock() { releases += 1; },
        };
      },
    },
  };

  await writeTeensyDisplayState(port, "running");
  await writeTeensyDisplayState(port, "standby");

  assert.deepEqual(writes, ["r", "s"]);
  assert.equal(releases, 2);
});

test("rejects commands when the serial port is not writable", async () => {
  await assert.rejects(() => writeTeensyDisplayState(null, "running"), /not writable/);
});
