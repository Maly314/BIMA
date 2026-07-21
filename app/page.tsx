"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { HandLandmarker, HandLandmarkerResult } from "@mediapipe/tasks-vision";

const sensors = ["Sensor 1", "Sensor 2"];
const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_PATH = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

function EmptyPlot() {
  return (
    <div className="plot" aria-label="Waiting for sensor data">
      <div className="plot-title">Movement</div>
      <div className="amplitude">— <span>mm</span></div>
      <div className="axis-label axis-top">+0.50</div>
      <div className="axis-label axis-mid">0</div>
      <div className="axis-label axis-bottom">-0.50</div>
      <div className="axis-unit">(mm)</div>
      <div className="plot-area">
        <div className="grid-line top" /><div className="grid-line zero" /><div className="grid-line bottom" />
        <div className="waiting-label"><span className="spinner" />Waiting for data</div>
      </div>
    </div>
  );
}

function SensorView() {
  const [placements, setPlacements] = useState(["", ""]);
  const updatePlacement = (index: number, value: string) => setPlacements((current) => current.map((item, i) => i === index ? value : item));

  return (
    <>
      <div className="sensors">
        {sensors.map((sensor, index) => (
          <section className="sensor-row" key={sensor} aria-labelledby={`sensor-${index}`}>
            <div className="sensor-meta">
              <h2 id={`sensor-${index}`}>{sensor}</h2>
              <label><span>Placement</span>
                <select value={placements[index]} onChange={(event) => updatePlacement(index, event.target.value)}>
                  <option value="" disabled>Select placement</option><option>Left ankle</option><option>Right ankle</option>
                  <option>Left wrist</option><option>Right wrist</option><option>Chest</option><option>Other</option>
                </select>
              </label>
              <div className="connection-status"><span className="status-dot" />Waiting for sensor</div>
            </div>
            <EmptyPlot />
          </section>
        ))}
      </div>
      <div className="time-scale" aria-hidden="true"><div className="ticks" /><span>-60 s</span><span>-45 s</span><span>-30 s</span><span>-15 s</span><span>0 s</span></div>
      <footer className="record-bar"><div className="timer">00:00</div><div className="ready-state"><span className="status-dot" />Waiting for sensor connection</div><button type="button" disabled>Record</button></footer>
    </>
  );
}

function VideoView() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const frameRef = useRef<number | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startTimeRef = useRef(0);
  const [cameraState, setCameraState] = useState<"off" | "starting" | "ready" | "error">("off");
  const [status, setStatus] = useState("Camera is off");
  const [hands, setHands] = useState(0);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState("00:00");
  const [downloadUrl, setDownloadUrl] = useState("");

  const drawResult = useCallback((result: HandLandmarkerResult) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.save(); ctx.translate(canvas.width, 0); ctx.scale(-1, 1); ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.lineWidth = Math.max(2, canvas.width / 420); ctx.strokeStyle = "#24d2c1"; ctx.fillStyle = "#ffffff";
    for (const landmarks of result.landmarks) {
      const links = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
      ctx.beginPath();
      links.forEach(([a,b]) => { const p=landmarks[a], q=landmarks[b]; ctx.moveTo(p.x*canvas.width,p.y*canvas.height); ctx.lineTo(q.x*canvas.width,q.y*canvas.height); });
      ctx.stroke();
      landmarks.forEach((point, index) => { ctx.beginPath(); ctx.arc(point.x*canvas.width, point.y*canvas.height, index===0?6:4, 0, Math.PI*2); ctx.fill(); ctx.stroke(); });
    }
    ctx.restore();
  }, []);

  const runTracking = useCallback(() => {
    const video = videoRef.current;
    const detector = landmarkerRef.current;
    if (video && detector && video.readyState >= 2) {
      const result = detector.detectForVideo(video, performance.now());
      drawResult(result); setHands(result.landmarks.length);
    }
    frameRef.current = requestAnimationFrame(runTracking);
  }, [drawResult]);

  const enableCamera = async () => {
    setCameraState("starting"); setStatus("Starting camera and hand tracking…");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width:{ideal:1280}, height:{ideal:720}, facingMode:"user" }, audio:false });
      streamRef.current = stream;
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
      const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision");
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
      landmarkerRef.current = await HandLandmarker.createFromOptions(vision, { baseOptions:{modelAssetPath:MODEL_PATH,delegate:"GPU"}, runningMode:"VIDEO", numHands:2, minHandDetectionConfidence:.5, minHandPresenceConfidence:.5, minTrackingConfidence:.5 });
      setCameraState("ready"); setStatus("Camera connected · hand tracking active"); runTracking();
    } catch (error) {
      console.error(error); setCameraState("error"); setStatus("Camera unavailable or permission denied");
      streamRef.current?.getTracks().forEach((track) => track.stop()); streamRef.current=null;
    }
  };

  const toggleRecording = () => {
    if (!canvasRef.current || cameraState !== "ready") return;
    if (!recording) {
      if (downloadUrl) URL.revokeObjectURL(downloadUrl);
      setDownloadUrl(""); chunksRef.current=[];
      const stream = canvasRef.current.captureStream(30);
      const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
      const recorder = new MediaRecorder(stream,{mimeType}); recorderRef.current=recorder;
      recorder.ondataavailable=(event)=>{ if(event.data.size) chunksRef.current.push(event.data); };
      recorder.onstop=()=>{ const blob=new Blob(chunksRef.current,{type:mimeType}); setDownloadUrl(URL.createObjectURL(blob)); stream.getTracks().forEach((track)=>track.stop()); };
      startTimeRef.current=Date.now(); recorder.start(); setRecording(true);
    } else { recorderRef.current?.stop(); setRecording(false); }
  };

  useEffect(() => {
    if (!recording) return;
    const timer = window.setInterval(() => { const total=Math.floor((Date.now()-startTimeRef.current)/1000); setElapsed(`${String(Math.floor(total/60)).padStart(2,"0")}:${String(total%60).padStart(2,"0")}`); },250);
    return () => window.clearInterval(timer);
  }, [recording]);

  useEffect(() => () => {
    if (frameRef.current) cancelAnimationFrame(frameRef.current);
    recorderRef.current?.state === "recording" && recorderRef.current.stop();
    streamRef.current?.getTracks().forEach((track) => track.stop()); landmarkerRef.current?.close();
  }, []);

  return (
    <section className="video-view" aria-label="Video hand tracking">
      <div className="video-stage">
        <video ref={videoRef} muted playsInline aria-hidden="true" />
        <canvas ref={canvasRef} />
        {cameraState !== "ready" && <div className="camera-empty"><div className="camera-outline" /><h2>Webcam preview</h2><p>Camera access starts only when you choose Enable camera.</p><button className="enable-camera" type="button" onClick={enableCamera} disabled={cameraState==="starting"}>{cameraState==="starting" ? "Starting…" : "Enable camera"}</button></div>}
        {cameraState === "ready" && <div className="tracking-badge"><span className="live-dot" />{hands ? `${hands} hand${hands===1?"":"s"} tracked` : "Looking for hands"}</div>}
      </div>
      <aside className="video-panel">
        <div><span className="eyebrow">Hand tracking</span><h2>Finger landmarks</h2><p>Tracks up to two hands and overlays 21 landmarks per hand on the recorded video.</p></div>
        <dl><div><dt>Camera</dt><dd>{cameraState === "ready" ? "Connected" : "Not connected"}</dd></div><div><dt>Hands detected</dt><dd>{cameraState === "ready" ? hands : "—"}</dd></div><div><dt>Overlay</dt><dd>{cameraState === "ready" ? "Active" : "Waiting"}</dd></div></dl>
        <p className="privacy-note">Video and landmark processing stay in this browser session.</p>
      </aside>
      <footer className="record-bar video-record-bar"><div className="timer">{elapsed}</div><div className="ready-state"><span className={`status-dot ${cameraState==="ready"?"connected":""}`} />{status}</div>{downloadUrl && <a className="download-link" href={downloadUrl} download={`movement-video-${Date.now()}.webm`}>Download recording</a>}<button className={recording?"recording":""} type="button" disabled={cameraState!=="ready"} onClick={toggleRecording}>{recording ? "Stop recording" : "Record video"}</button></footer>
    </section>
  );
}

export default function Home() {
  const [view, setView] = useState<"sensor" | "video">("sensor");
  return (
    <main className="page-shell"><section className="app-window" aria-labelledby="page-title">
      <div className="window-bar" aria-hidden="true"><i className="traffic red" /><i className="traffic amber" /><i className="traffic green" /></div>
      <header className="session-header"><div className="title-line"><h1 id="page-title">Movement capture</h1><nav className="view-switch" aria-label="Capture view"><button className={view==="sensor"?"active":""} onClick={()=>setView("sensor")}>Sensor</button><button className={view==="video"?"active":""} onClick={()=>setView("video")}>Video</button></nav></div>
        <div className="field-row"><label><span>Study ID</span><input type="text" placeholder="Enter study ID" autoComplete="off" /></label><label><span>Age</span><input type="text" inputMode="numeric" placeholder="Enter age" autoComplete="off" /></label><label><span>Note</span><input type="text" placeholder="Add a note" autoComplete="off" /></label></div>
      </header>
      {view === "sensor" ? <SensorView /> : <VideoView />}
    </section></main>
  );
}
