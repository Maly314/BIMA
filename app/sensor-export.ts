export type SensorRow = Record<string, string | number>;

const EXCEL_BOM = "\uFEFF";
const CRLF = "\r\n";

export const SENSOR_BASE_COLUMNS = [
  "session_id", "packet_index", "session_time_ms", "epoch_ms", "t",
  "device_us", "seq", "dropped_packets", "patient_number", "study_id",
  "study_date", "age_basis", "age_days", "corrected_age_days",
  "chronological_age_days", "gestational_age_birth_days",
  "postmenstrual_age_days", "weight_kg",
];

export const SENSOR_CHANNEL_SUFFIXES = [
  "placement", "imu", "offset_us", "device_us", "ax", "ay", "az", "gx", "gy", "gz", "mov",
];

export const SENSOR_LONG_COLUMNS = [
  ...SENSOR_BASE_COLUMNS,
  "sensor_id", "placement", "sample_offset_us", "sample_device_us", "sample_valid",
  "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
  "gyro_x_rads", "gyro_y_rads", "gyro_z_rads", "movement_mps2",
];

export function csvCell(value: unknown): string {
  const raw = String(value ?? "");
  // Excel evaluates cells beginning with these characters as formulas even
  // when the CSV field is quoted. Protect operator-entered IDs, placements,
  // and notes without changing real negative measurements (which are numbers).
  const text = typeof value === "string" && /^[=+\-@]/.test(raw) ? `'${raw}` : raw;
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function csvParts(columns: string[], rows: SensorRow[], batchSize = 1000): string[] {
  const parts = [EXCEL_BOM + columns.map(csvCell).join(",")];
  for (let offset = 0; offset < rows.length; offset += batchSize) {
    const lines = rows.slice(offset, offset + batchSize).map((row) => columns.map((column) => csvCell(row[column])).join(","));
    if (lines.length) parts.push(CRLF + lines.join(CRLF));
  }
  return parts;
}

function csv(columns: string[], rows: SensorRow[]): string {
  return csvParts(columns, rows).join("");
}

export function sensorWideColumns(rows: SensorRow[], sensorCount = 4): string[] {
  const ordered = [
    ...SENSOR_BASE_COLUMNS,
    ...Array.from({ length: sensorCount }, (_, index) => SENSOR_CHANNEL_SUFFIXES.map((suffix) => `s${index + 1}_${suffix}`)).flat(),
  ];
  const present = new Set(rows.flatMap((row) => Object.keys(row)));
  const extras = [...present].filter((column) => !ordered.includes(column)).sort();
  return [...ordered.filter((column) => present.has(column) || SENSOR_BASE_COLUMNS.includes(column)), ...extras];
}

export function buildSensorWideCsv(rows: SensorRow[], sensorCount = 4): string {
  return csv(sensorWideColumns(rows, sensorCount), rows);
}

export function buildSensorWideCsvParts(rows: SensorRow[], sensorCount = 4): string[] {
  return csvParts(sensorWideColumns(rows, sensorCount), rows);
}

export function sensorLongRows(rows: SensorRow[], sensorCount = 4): SensorRow[] {
  return rows.flatMap((row) => Array.from({ length: sensorCount }, (_, index) => {
    const prefix = `s${index + 1}_`;
    const output: SensorRow = {};
    for (const column of SENSOR_BASE_COLUMNS) output[column] = row[column] ?? "";
    const hasSample = ["ax", "ay", "az", "gx", "gy", "gz"].every((suffix) => row[`${prefix}${suffix}`] !== undefined);
    Object.assign(output, {
      sensor_id: row[`${prefix}imu`] ?? index + 1,
      placement: row[`${prefix}placement`] ?? "unspecified",
      sample_offset_us: row[`${prefix}offset_us`] ?? "",
      sample_device_us: row[`${prefix}device_us`] ?? "",
      sample_valid: hasSample ? 1 : 0,
      accel_x_mps2: row[`${prefix}ax`] ?? "",
      accel_y_mps2: row[`${prefix}ay`] ?? "",
      accel_z_mps2: row[`${prefix}az`] ?? "",
      gyro_x_rads: row[`${prefix}gx`] ?? "",
      gyro_y_rads: row[`${prefix}gy`] ?? "",
      gyro_z_rads: row[`${prefix}gz`] ?? "",
      movement_mps2: row[`${prefix}mov`] ?? "",
    });
    return output;
  }));
}

export function buildSensorLongCsv(rows: SensorRow[], sensorCount = 4): string {
  return csv(SENSOR_LONG_COLUMNS, sensorLongRows(rows, sensorCount));
}

export function buildSensorLongCsvParts(rows: SensorRow[], sensorCount = 4, batchSize = 500): string[] {
  const parts = [EXCEL_BOM + SENSOR_LONG_COLUMNS.map(csvCell).join(",")];
  for (let offset = 0; offset < rows.length; offset += batchSize) {
    const lines: string[] = [];
    for (const row of rows.slice(offset, offset + batchSize)) {
      for (const longRow of sensorLongRows([row], sensorCount)) {
        lines.push(SENSOR_LONG_COLUMNS.map((column) => csvCell(longRow[column])).join(","));
      }
    }
    if (lines.length) parts.push(CRLF + lines.join(CRLF));
  }
  return parts;
}

const SENSOR_DICTIONARY: Array<[string, string, string]> = [
  ["session_id", "Stable synchronized capture identifier", "identifier"],
  ["packet_index", "Zero-based host acquisition-cycle index", "count"],
  ["session_time_ms", "Host monotonic time from shared capture start", "ms"],
  ["epoch_ms", "Estimated UTC Unix time for the acquisition cycle", "ms since 1970-01-01"],
  ["device_us", "Teensy monotonic clock at acquisition-cycle start", "microseconds"],
  ["seq", "Teensy packet sequence number used to identify gaps", "count"],
  ["dropped_packets", "Cumulative missing packet count", "count"],
  ["sensor_id", "Fixed physical IMU number; columns never shift after dropout", "1-4"],
  ["placement", "Operator-selected body placement", "text"],
  ["sample_offset_us", "Per-IMU read offset from acquisition-cycle start", "microseconds"],
  ["sample_device_us", "Teensy clock plus the per-IMU read offset", "microseconds"],
  ["sample_valid", "1 when all six raw IMU axes were present; otherwise 0", "boolean"],
  ["accel_x_mps2", "Raw X-axis acceleration", "m/s^2"],
  ["accel_y_mps2", "Raw Y-axis acceleration", "m/s^2"],
  ["accel_z_mps2", "Raw Z-axis acceleration", "m/s^2"],
  ["gyro_x_rads", "Bias-corrected X-axis angular velocity", "rad/s"],
  ["gyro_y_rads", "Bias-corrected Y-axis angular velocity", "rad/s"],
  ["gyro_z_rads", "Bias-corrected Z-axis angular velocity", "rad/s"],
  ["movement_mps2", "Gravity- and calibration-noise-adjusted acceleration magnitude", "m/s^2"],
];

export function buildSensorDataDictionaryCsv(): string {
  return EXCEL_BOM + ["column,definition,unit", ...SENSOR_DICTIONARY.map((row) => row.map(csvCell).join(","))].join(CRLF);
}
