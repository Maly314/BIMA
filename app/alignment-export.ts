import { csvCell } from "./sensor-export.ts";
import type { Recording } from "./recording-types.ts";

type TrackingFrame = { frameIndex?:number; sessionTimeMs?:number; sourceVideoTimeMs?:number; hands?:unknown[]; displayHands?:unknown[]; segments?:unknown[] };

function parseCsv(text:string):string[][] {
  const rows:string[][]=[]; let row:string[]=[], cell="", quoted=false; const source=text.replace(/^\uFEFF/,"");
  for(let i=0;i<source.length;i+=1){const char=source[i]; if(quoted){if(char==='"'&&source[i+1]==='"'){cell+='"';i+=1;}else if(char==='"')quoted=false;else cell+=char;}else if(char==='"')quoted=true;else if(char===","){row.push(cell);cell="";}else if(char==="\n"){row.push(cell.replace(/\r$/, ""));rows.push(row);row=[];cell="";}else cell+=char;}
  if(cell||row.length){row.push(cell.replace(/\r$/, ""));rows.push(row);} return rows;
}

function nearestIndex(times:number[],target:number):number { let low=0,high=times.length; while(low<high){const mid=(low+high)>>>1;if(times[mid]<target)low=mid+1;else high=mid;} if(low===0)return 0;if(low===times.length)return times.length-1;return Math.abs(times[low]-target)<Math.abs(times[low-1]-target)?low:low-1; }

export async function buildFrameSensorAlignmentCsv(sensor:Recording,video:Recording):Promise<Blob>{
  if(!video.sidecarBlob)throw new Error("The video recording has no frame-level tracking data");
  const [headers,...values]=parseCsv(await sensor.blob.text()); const timeIndex=headers?.indexOf("session_time_ms")??-1;
  if(!headers?.length||timeIndex<0||!values.length)throw new Error("The sensor recording has no synchronized samples");
  const sensorTimes=values.map(row=>Number(row[timeIndex])); if(sensorTimes.some(time=>!Number.isFinite(time)))throw new Error("The sensor timeline is invalid");
  const sidecar=JSON.parse(await video.sidecarBlob.text()) as {synchronization?:{recorderStartedOffsetMs?:number};frames?:TrackingFrame[]};
  const offset=Number(sidecar.synchronization?.recorderStartedOffsetMs??video.sync?.streamStartOffsetMs??0); const frames=sidecar.frames??[];
  if(!frames.length)throw new Error("The video recording has no frame timestamps");
  const columns=["capture_session_id","frame_index","video_time_ms","frame_session_time_ms","frame_epoch_ms","nearest_sensor_time_ms","sensor_time_delta_ms","tracking_count",...headers];
  const lines=["\uFEFF"+columns.map(csvCell).join(",")];
  frames.forEach((frame,index)=>{const videoTime=Number(frame.sourceVideoTimeMs??0);const sessionTime=Number(frame.sessionTimeMs??offset+videoTime);const sampleIndex=nearestIndex(sensorTimes,sessionTime);const sensorTime=sensorTimes[sampleIndex];const count=(frame.displayHands??frame.hands??frame.segments??[]).length;lines.push([sensor.captureSessionId??video.captureSessionId??"",frame.frameIndex??index,Number(videoTime.toFixed(3)),Number(sessionTime.toFixed(3)),(video.sync?.startedAtEpochMs??sensor.sync?.startedAtEpochMs??0)+sessionTime,Number(sensorTime.toFixed(3)),Number((sensorTime-sessionTime).toFixed(3)),count,...values[sampleIndex]].map(csvCell).join(","));});
  return new Blob([lines.join("\r\n")],{type:"text/csv;charset=utf-8"});
}

export const alignmentFilename=(sensor:Recording)=>sensor.filename.replace(/-sensors\.csv$/i,"-frame-sensor-alignment.csv");
