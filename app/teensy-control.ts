export type TeensyDisplayState = "standby" | "running" | "unknown";
export type TeensyRequestedState = Exclude<TeensyDisplayState, "unknown">;

type SerialWriter = {
  write(data: Uint8Array): Promise<void>;
  releaseLock(): void;
};

export type WritableSerialPort = {
  writable?: {
    getWriter(): SerialWriter;
  } | null;
};

const COMMANDS: Record<TeensyRequestedState, string> = {
  running: "r",
  standby: "s",
};

export function parseTeensyDisplayState(line: string): TeensyDisplayState {
  const match = /^System state:\s*(RUNNING|STANDBY)\s*$/i.exec(line.trim());
  if (!match) return "unknown";
  return match[1].toLowerCase() as TeensyRequestedState;
}

export async function writeTeensyDisplayState(
  port: WritableSerialPort | null,
  state: TeensyRequestedState,
): Promise<void> {
  if (!port?.writable) throw new Error("The Teensy serial connection is not writable");

  const writer = port.writable.getWriter();
  try {
    // The flashed firmware consumes one byte at a time: r = RUNNING, s = STANDBY.
    await writer.write(new TextEncoder().encode(COMMANDS[state]));
  } finally {
    writer.releaseLock();
  }
}
