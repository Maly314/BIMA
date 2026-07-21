"use client";

import { useState } from "react";

const sensors = ["Sensor 1", "Sensor 2"];

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
        <div className="grid-line top" />
        <div className="grid-line zero" />
        <div className="grid-line bottom" />
        <div className="waiting-label"><span className="spinner" />Waiting for data</div>
      </div>
    </div>
  );
}

export default function Home() {
  const [placements, setPlacements] = useState(["", ""]);

  function updatePlacement(index: number, value: string) {
    setPlacements((current) => current.map((item, i) => i === index ? value : item));
  }

  return (
    <main className="page-shell">
      <section className="app-window" aria-labelledby="page-title">
        <div className="window-bar" aria-hidden="true">
          <i className="traffic red" /><i className="traffic amber" /><i className="traffic green" />
        </div>

        <header className="session-header">
          <h1 id="page-title">Movement capture</h1>
          <div className="field-row">
            <label>
              <span>Study ID</span>
              <input type="text" placeholder="Enter study ID" autoComplete="off" />
            </label>
            <label>
              <span>Age</span>
              <input type="text" inputMode="numeric" placeholder="Enter age" autoComplete="off" />
            </label>
            <label>
              <span>Note</span>
              <input type="text" placeholder="Add a note" autoComplete="off" />
            </label>
          </div>
        </header>

        <div className="sensors">
          {sensors.map((sensor, index) => (
            <section className="sensor-row" key={sensor} aria-labelledby={`sensor-${index}`}>
              <div className="sensor-meta">
                <h2 id={`sensor-${index}`}>{sensor}</h2>
                <label>
                  <span>Placement</span>
                  <select value={placements[index]} onChange={(event) => updatePlacement(index, event.target.value)}>
                    <option value="" disabled>Select placement</option>
                    <option>Left ankle</option>
                    <option>Right ankle</option>
                    <option>Left wrist</option>
                    <option>Right wrist</option>
                    <option>Chest</option>
                    <option>Other</option>
                  </select>
                </label>
                <div className="connection-status"><span className="status-dot" />Waiting for sensor</div>
              </div>
              <EmptyPlot />
            </section>
          ))}
        </div>

        <div className="time-scale" aria-hidden="true">
          <div className="ticks" />
          <span>-60 s</span><span>-45 s</span><span>-30 s</span><span>-15 s</span><span>0 s</span>
        </div>

        <footer className="record-bar">
          <div className="timer">00:00</div>
          <div className="ready-state"><span className="status-dot" />Waiting for sensor connection</div>
          <button type="button" disabled>Record</button>
        </footer>
      </section>
    </main>
  );
}
