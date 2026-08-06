export type TremorFrame = [number[], number[]];

type BandResult = { ratio: number; rms: number };

function bandResult(values: number[], sampleRate: number): BandResult {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  let bandPower = 0;
  let totalPower = 0;
  for (let k = 1; k <= Math.floor(values.length / 2); k++) {
    const frequency = k * sampleRate / values.length;
    if (frequency > 20) break;
    if (frequency < 1) continue;
    let real = 0, imaginary = 0;
    for (let n = 0; n < values.length; n++) {
      const angle = 2 * Math.PI * k * n / values.length;
      const centered = values[n] - mean;
      real += centered * Math.cos(angle);
      imaginary -= centered * Math.sin(angle);
    }
    const power = real * real + imaginary * imaginary;
    totalPower += power;
    if (frequency >= 6 && frequency <= 10) bandPower += power;
  }
  return {
    ratio: bandPower / Math.max(totalPower, 1e-12),
    rms: Math.sqrt(2 * bandPower) / values.length,
  };
}

export function hasFineTremor(frames: TremorFrame[], sampleRate: number): boolean {
  if (sampleRate < 25 || frames.length < 45) return false;
  return [0, 1].some((sensor) => {
    const channels = [0, 1, 2, 3, 4, 5].map((axis) => frames.map((frame) => frame[sensor][axis]));
    return channels.some((values, axis) => {
      const result = bandResult(values, sampleRate);
      const minimumRms = axis < 3 ? .05 : .015;
      return result.ratio > .55 && result.rms > minimumRms;
    });
  });
}
