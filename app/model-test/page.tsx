"use client";

import { useEffect, useRef, useState } from "react";
import { getCalibration } from "../recordings";
import { correctedGyro, parseImuLine, type ImuSample } from "../sensor-calibration";
import styles from "./ModelTest.module.css";
import { hasFineTremor, type TremorFrame } from "./tremor-detector";

type LiveSample = ImuSample & { correctedGyro: [number, number, number] };
type TimedFrame = { at: number; frame: TremorFrame };

export default function ModelTest() {
  const [connected, setConnected] = useState(false);
  const [samples, setSamples] = useState(0);
  const [sampleRateHz, setSampleRateHz] = useState(0);
  const [result, setResult] = useState<"shaky" | "not-shaky">("not-shaky");
  const [message, setMessage] = useState("Connect both sensors to begin");
  const portRef = useRef<any>(null);
  const readerRef = useRef<any>(null);
  const readTaskRef = useRef<Promise<void> | null>(null);
  const readingRef = useRef(false);
  const latestRef = useRef<Record<number, LiveSample>>({});
  const sensorIdsRef = useRef<number[]>([]);
  const rowsRef = useRef<TimedFrame[]>([]);
  const votesRef = useRef<boolean[]>([]);
  const lastAnalysisRef = useRef(0);

  const consumeLine = (raw: string) => {
    const line = raw.trim();
    if (line) {
      const parsed = parseImuLine(line);
      if (!parsed) return;
      const calibration = getCalibration()[parsed.imu];
      const gyro = correctedGyro(parsed.sample, calibration);
      latestRef.current[parsed.imu] = {
        ...parsed.sample,
        correctedGyro: gyro,
      };
      if (!sensorIdsRef.current.includes(parsed.imu)) sensorIdsRef.current = [...sensorIdsRef.current, parsed.imu].sort((a, b) => a - b).slice(0, 2);
      return;
    }

    const ids = sensorIdsRef.current;
    if (ids.length < 2 || !latestRef.current[ids[0]] || !latestRef.current[ids[1]]) {
      setMessage("Waiting for both sensors");
      return;
    }
    const first = latestRef.current[ids[0]], second = latestRef.current[ids[1]], now = performance.now() / 1000;
    rowsRef.current.push({ at: now, frame: [[first.ax, first.ay, first.az, ...first.correctedGyro], [second.ax, second.ay, second.az, ...second.correctedGyro]] });
    while (rowsRef.current.length && now - rowsRef.current[0].at > 2) rowsRef.current.shift();
    setSamples(rowsRef.current.length);
    const duration = rowsRef.current.length > 1 ? now - rowsRef.current[0].at : 0;
    const sampleRate = duration > 0 ? (rowsRef.current.length - 1) / duration : 0;
    setSampleRateHz(sampleRate);
    if (duration < 1.8) { setMessage(`Starting live detector · ${sampleRate.toFixed(0)} Hz`); return; }
    if (now - lastAnalysisRef.current < .1) return;
    lastAnalysisRef.current = now;
    if (sampleRate < 25) { setResult("not-shaky"); setMessage(`${sampleRate.toFixed(0)} Hz received · high-rate firmware required`); return; }
    const frames = rowsRef.current.map((row) => row.frame);
    votesRef.current.push(hasFineTremor(frames, sampleRate));
    if (votesRef.current.length > 5) votesRef.current.shift();
    const votes = votesRef.current.filter(Boolean).length;
    if (votes >= 3) setResult("shaky");
    else if (votes <= 1) setResult("not-shaky");
    setMessage(`Live · ${sampleRate.toFixed(0)} Hz · frequency detector`);
  };

  const readLoop = async (port: any) => {
    const reader = port.readable.getReader();
    readerRef.current = reader;
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (readingRef.current) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() ?? "";
        lines.forEach(consumeLine);
      }
    } catch (error) {
      if (readingRef.current) setMessage(error instanceof Error ? error.message : "Sensor connection ended");
    } finally {
      try { reader.releaseLock(); } catch { /* already released */ }
      readerRef.current = null;
    }
  };

  const connect = async () => {
    try {
      const serial = (navigator as any).serial;
      const authorized = await serial.getPorts();
      const port = authorized[0] ?? await serial.requestPort();
      await port.open({ baudRate: 921600 });
      portRef.current = port;
      readingRef.current = true;
      rowsRef.current = [];
      votesRef.current = [];
      sensorIdsRef.current = [];
      setSamples(0); setSampleRateHz(0); setResult("not-shaky"); setConnected(true); setMessage("Waiting for both sensors");
      readTaskRef.current = readLoop(port);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not connect sensors");
    }
  };

  const disconnect = async () => {
    readingRef.current = false;
    try { await readerRef.current?.cancel(); } catch { /* noop */ }
    try { await readTaskRef.current; } catch { /* noop */ }
    try { await portRef.current?.close(); } catch { /* noop */ }
    portRef.current = null; readTaskRef.current = null;
    setConnected(false); setSamples(0); setSampleRateHz(0); setResult("not-shaky"); setMessage("Connect both sensors to begin");
  };

  useEffect(() => () => {
    readingRef.current = false;
    void (async () => {
      try { await readerRef.current?.cancel(); } catch { /* noop */ }
      try { await readTaskRef.current; } catch { /* noop */ }
      try { await portRef.current?.close(); } catch { /* noop */ }
    })();
  }, []);

  const state = !connected ? "idle" : result;
  const label = state === "shaky" ? "SHAKY" : state === "not-shaky" ? "NOT SHAKY" : "NOT CONNECTED";

  return (
    <main className={styles.page}>
      <section className={styles.card}>
        <div className={styles.top}><span>Movement model test</span><i className={connected ? styles.on : ""} /></div>
        <div className={`${styles.result} ${styles[state]}`} aria-live="polite">{label}</div>
        <p>{message}</p>
        <button type="button" onClick={connected ? disconnect : connect}>{connected ? "Disconnect" : "Connect sensors"}</button>
        <small>{!connected ? "Connect both sensors to run continuously." : sampleRateHz > 0 && sampleRateHz < 25 ? "High-rate firmware is required; no fallback classification is used." : samples < 45 ? "Measuring the incoming sample rate…" : "Checks sustained 6–10 Hz motion from the 100 Hz stream."}</small>
      </section>
    </main>
  );
}
