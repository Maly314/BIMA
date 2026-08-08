const GPU_SWITCHES = [
  ['ignore-gpu-blocklist'],
  ['enable-gpu-rasterization'],
  ['enable-gpu'],
  ['enable-gpu-compositing'],
  ['enable-zero-copy'],
  ['enable-accelerated-2d-canvas'],
  ['enable-accelerated-video-decode'],
  ['enable-webgl'],
  ['enable-webgl2'],
  ['use-angle', 'd3d11'],
  ['use-gl', 'angle'],
  ['disable-features', 'CalculateNativeWinOcclusion'],
  ['disable-renderer-backgrounding'],
];

function applyCaptureRuntimeSwitches(commandLine) {
  for (const [name, value] of GPU_SWITCHES) {
    if (value === undefined) commandLine.appendSwitch(name);
    else commandLine.appendSwitch(name, value);
  }
}

function parseListeningPids(netstatOutput, port) {
  const pids = new Set();
  for (const line of String(netstatOutput || '').split('\n')) {
    const match = line.match(/TCP\s+\S+:(\d+)\s+\S+\s+LISTENING\s+(\d+)/);
    if (match && Number(match[1]) === port && Number(match[2]) > 0) pids.add(match[2]);
  }
  return [...pids];
}

module.exports = { GPU_SWITCHES, applyCaptureRuntimeSwitches, parseListeningPids };
