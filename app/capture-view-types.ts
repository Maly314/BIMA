import type { RefObject } from "react";
import type { CaptureRun } from "./capture-sync";
import type { Session } from "./session-domain";

export type CaptureChildProps = {
  session: Session;
  captureRun: CaptureRun | null;
  onReadyChange: (ready: boolean) => void;
  onSaved: (kind: "sensor" | "pose", ok: boolean) => void;
};

export type SensorViewProps = CaptureChildProps & {
  posePreviewRef: RefObject<HTMLCanvasElement | null>;
  posePreviewActive: boolean;
  active: boolean;
};

export type VideoViewProps = CaptureChildProps & {
  posePreviewRef: RefObject<HTMLCanvasElement | null>;
  registerCameraControl: (enable: () => Promise<boolean>) => void;
};
