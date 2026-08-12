import test from "node:test";
import assert from "node:assert/strict";
import { alignmentFilename, buildFrameSensorAlignmentCsv } from "../app/alignment-export.ts";

test("combined export aligns each video frame to the nearest sensor packet",async()=>{
  const sensor={filename:"patient-1-run-sensors.csv",captureSessionId:"run-1",blob:new Blob(["\uFEFFsession_id,packet_index,session_time_ms,s1_ax\r\nrun-1,0,0,1\r\nrun-1,1,10,2\r\nrun-1,2,20,3"]),sync:{startedAtEpochMs:1000}};
  const video={captureSessionId:"run-1",sync:{startedAtEpochMs:1000,streamStartOffsetMs:5},sidecarBlob:new Blob([JSON.stringify({synchronization:{recorderStartedOffsetMs:5},frames:[{frameIndex:0,sourceVideoTimeMs:0,hands:[{}]},{frameIndex:1,sourceVideoTimeMs:11,hands:[{},{}]}]})])};
  const lines=(await(await buildFrameSensorAlignmentCsv(sensor,video)).text()).replace(/^\uFEFF/,"").split("\r\n");
  assert.match(lines[0],/^capture_session_id,frame_index,video_time_ms,frame_session_time_ms/); assert.equal(lines[1],"run-1,0,0,5,1005,0,-5,1,run-1,0,0,1"); assert.equal(lines[2],"run-1,1,11,16,1016,20,4,2,run-1,2,20,3"); assert.equal(alignmentFilename(sensor),"patient-1-run-frame-sensor-alignment.csv");
});
