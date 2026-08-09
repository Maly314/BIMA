const assert = require('node:assert/strict');
const test = require('node:test');

const { GPU_SWITCHES, applyCaptureRuntimeSwitches, parseListeningPids } = require('../desktop/capture-runtime.cjs');
const { idMatches, isAllowedPermission, selectSerialPort } = require('../desktop/device-policy.cjs');

test('capture runtime keeps every measured GPU and anti-throttling switch', () => {
  const applied = [];
  applyCaptureRuntimeSwitches({ appendSwitch: (...args) => applied.push(args) });
  assert.deepEqual(applied, GPU_SWITCHES);
  assert.ok(applied.some(([name, value]) => name === 'use-angle' && value === 'd3d11'));
  assert.ok(applied.some(([name, value]) => name === 'disable-features' && value === 'CalculateNativeWinOcclusion'));
  assert.ok(applied.some(([name]) => name === 'disable-renderer-backgrounding'));
});

test('listener parsing returns only unique positive PIDs on the requested port', () => {
  const netstat = [
    '  TCP    127.0.0.1:4820     0.0.0.0:0       LISTENING       111',
    '  TCP    [::1]:4820         [::]:0          LISTENING       111',
    '  TCP    127.0.0.1:4831     0.0.0.0:0       LISTENING       222',
    '  TCP    127.0.0.1:4820     0.0.0.0:0       LISTENING       333',
  ].join('\n');
  assert.deepEqual(parseListeningPids(netstat, 4820), ['111', '333']);
  assert.deepEqual(parseListeningPids(netstat, 4831), ['222']);
});

test('Teensy IDs match Electron decimal strings and USB hexadecimal strings', () => {
  assert.equal(idMatches('5824', '16C0'), true);
  assert.equal(idMatches('16c0', '16C0'), true);
  assert.equal(idMatches('0x0483', '0483'), true);
  assert.equal(idMatches('', '16C0'), false);
});

test('serial selection prefers Teensy, then USB, and never chooses legacy COM1', () => {
  const acpi = { portId: 'COM1', vendorId: '', productId: '', displayName: 'Communications Port' };
  const usb = { portId: 'COM7', vendorId: '1234', productId: '9', displayName: 'USB Serial' };
  const teensy = { portId: 'COM9', vendorId: '5824', productId: '1155', displayName: 'Teensy' };
  assert.equal(selectSerialPort([acpi, usb, teensy]), teensy);
  assert.equal(selectSerialPort([acpi, usb]), usb);
  assert.equal(selectSerialPort([acpi]), undefined);
});

test('camera and serial permissions are restricted to the BIMA application origin', () => {
  const appUrl = 'http://127.0.0.1:4820';
  assert.equal(isAllowedPermission('camera', `${appUrl}/`, appUrl), true);
  assert.equal(isAllowedPermission('serial', `${appUrl}/capture`, appUrl), true);
  assert.equal(isAllowedPermission('geolocation', `${appUrl}/`, appUrl), false);
  assert.equal(isAllowedPermission('camera', 'https://example.com/', appUrl), false);
  assert.equal(isAllowedPermission('serial', 'http://127.0.0.1:48200/hostile', appUrl), false);
  assert.equal(isAllowedPermission('camera', 'http://127.0.0.1:4820.evil.example/', appUrl), false);
});
