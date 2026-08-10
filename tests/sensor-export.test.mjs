import assert from 'node:assert/strict';
import test from 'node:test';
import { buildSensorDataDictionaryCsv, buildSensorLongCsv, buildSensorLongCsvParts, buildSensorWideCsv, buildSensorWideCsvParts, sensorLongRows } from '../app/sensor-export.ts';

const rows = [{
  session_id: 'session-1', packet_index: 0, session_time_ms: 12, epoch_ms: 100012,
  t: 0.012, device_us: 8000, seq: 4, dropped_packets: 0, patient_number: 7,
  study_id: '=NICU, trial', study_date: '2026-08-10', age_basis: 'corrected', age_days: 42,
  corrected_age_days: 42, chronological_age_days: 63, gestational_age_birth_days: 238,
  postmenstrual_age_days: 301, weight_kg: 2.4,
  s1_placement: 'Left wrist', s1_imu: 1, s1_offset_us: 10, s1_device_us: 8010,
  s1_ax: 0.1, s1_ay: 0.2, s1_az: 9.8, s1_gx: 0.01, s1_gy: 0.02, s1_gz: 0.03, s1_mov: 0.04,
  s2_placement: 'Right wrist', s2_imu: 2,
}];

test('sensor exports are Excel-compatible, stable, and analysis-ready', () => {
  const wide = buildSensorWideCsv(rows);
  assert.ok(wide.startsWith('\uFEFFsession_id,packet_index,session_time_ms'));
  assert.match(wide, /\r\n/);
  assert.match(wide, /"'=NICU, trial"/);

  const longRows = sensorLongRows(rows);
  assert.equal(longRows.length, 4);
  assert.equal(longRows[0].sensor_id, 1);
  assert.equal(longRows[0].sample_valid, 1);
  assert.equal(longRows[1].sensor_id, 2);
  assert.equal(longRows[1].sample_valid, 0);
  const long = buildSensorLongCsv(rows);
  assert.match(long, /sensor_id,placement,sample_offset_us,sample_device_us,sample_valid/);
  assert.match(long, /accel_x_mps2/);
  assert.equal(buildSensorWideCsvParts(rows).join(''), wide);
  assert.equal(buildSensorLongCsvParts(rows).join(''), long);
});

test('sensor data dictionary documents units and quality fields', () => {
  const dictionary = buildSensorDataDictionaryCsv();
  assert.match(dictionary, /sample_valid/);
  assert.match(dictionary, /m\/s\^2/);
  assert.match(dictionary, /rad\/s/);
  assert.match(dictionary, /dropped_packets/);
});
