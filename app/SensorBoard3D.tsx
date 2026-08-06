"use client";

import type { RefObject } from "react";
import styles from "./SensorBoard3D.module.css";

type SensorBoard3DProps = {
  boardRef: RefObject<HTMLDivElement | null>;
  // Written imperatively from the render loop: textContent is the tilt readout
  // and data-level ("on" | "near" | "off") drives the styling. Kept out of React
  // state so it can update every frame without re-rendering the board.
  levelRef: RefObject<HTMLSpanElement | null>;
  live: boolean;
  sensorLabel: string;
};

const pins = Array.from({ length: 7 });

export default function SensorBoard3D({ boardRef, levelRef, live, sensorLabel }: SensorBoard3DProps) {
  return (
    <div className={`${styles.scene} ${live ? styles.live : styles.idle}`} aria-label={`${sensorLabel} LSM6DSV16X orientation model`}>
      <div className={styles.world}>
        <div className={styles.floor} aria-hidden="true" />
        <div className={styles.shadow} aria-hidden="true" />

        <div className={styles.board} ref={boardRef}>
          <div className={`${styles.pcbFace} ${styles.topFace}`}>
            <span className={styles.boardName}>LSM6DSV16X</span>
            <span className={styles.axisMark}>X↗&nbsp; Y↖</span>

            <span className={`${styles.qtSocket} ${styles.qtLeft}`}><i /></span>
            <span className={`${styles.qtSocket} ${styles.qtRight}`}><i /></span>
            <span className={styles.whitePlug}>
              {Array.from({ length: 4 }, (_, index) => <i key={index} />)}
            </span>
            <span className={styles.cableBundle}>
              {Array.from({ length: 4 }, (_, index) => <i key={index} />)}
            </span>

            <span className={`${styles.ic} ${styles.imuChip}`} aria-hidden="true">
              <b>ST</b><small>6DSV16X</small>
              <span className={styles.pinsTop}>{pins.map((_, index) => <i key={index} />)}</span>
              <span className={styles.pinsBottom}>{pins.map((_, index) => <i key={index} />)}</span>
            </span>
            <span className={`${styles.ic} ${styles.regulator}`}>1V8</span>
            <span className={`${styles.ic} ${styles.levelChip}`}>472</span>
            <span className={`${styles.ic} ${styles.smallChipA}`} />
            <span className={`${styles.ic} ${styles.smallChipB}`} />

            <span className={`${styles.passiveBank} ${styles.bankA}`}>
              {Array.from({ length: 6 }, (_, index) => <i key={index} />)}
            </span>
            <span className={`${styles.passiveBank} ${styles.bankB}`}>
              {Array.from({ length: 5 }, (_, index) => <i key={index} />)}
            </span>
            <span className={`${styles.passiveBank} ${styles.bankC}`}>
              {Array.from({ length: 4 }, (_, index) => <i key={index} />)}
            </span>

            <span className={styles.powerLed} aria-label={live ? "Red power LED on" : "Red power LED off"} />
            <span className={styles.ledMark}>PWR</span>
          </div>

          <div className={`${styles.pcbFace} ${styles.bottomFace}`}>
            <span className={styles.bottomTitle}>LSM6DSV16X</span>
            <span className={styles.bottomSubtitle}>6-DOF IMU</span>
            <span className={styles.bottomSpecs}>Accel: ±2–16 g<br />Gyro: ±125–2000 dps<br />ODR: 240 Hz</span>
            <span className={styles.bottomLogo}>6DOF</span>
            <span className={styles.solderDots}>{Array.from({ length: 12 }, (_, index) => <i key={index} />)}</span>
          </div>

          <div className={`${styles.edge} ${styles.edgeFront}`} />
          <div className={`${styles.edge} ${styles.edgeBack}`} />
          <div className={`${styles.edge} ${styles.edgeLeft}`} />
          <div className={`${styles.edge} ${styles.edgeRight}`} />
        </div>
      </div>
      <span className={styles.level} ref={levelRef} data-level="off" role="status">—</span>
      <span className={styles.caption}>LSM6DSV16X · accel + gyro</span>
    </div>
  );
}
