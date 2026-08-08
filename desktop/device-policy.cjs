const ALLOWED_PERMISSIONS = ['media', 'serial', 'camera', 'microphone'];

function isAppUrl(url, appUrl) {
  return typeof url === 'string' && url.startsWith(appUrl);
}

function isAllowedPermission(permission, url, appUrl) {
  return isAppUrl(url, appUrl) && ALLOWED_PERMISSIONS.includes(permission);
}

function idMatches(value, hex) {
  const raw = String(value ?? '').trim().toLowerCase().replace(/^0x/, '');
  if (!raw) return false;
  const wanted = parseInt(hex, 16);
  return parseInt(raw, 10) === wanted || parseInt(raw, 16) === wanted;
}

function selectSerialPort(ports) {
  const teensy = ports.find(
    (port) => idMatches(port.vendorId, '16C0') && idMatches(port.productId, '0483'),
  );
  return teensy || ports.find((port) => String(port.vendorId || '').trim() !== '');
}

module.exports = { ALLOWED_PERMISSIONS, idMatches, isAllowedPermission, isAppUrl, selectSerialPort };
