"""Evaluate the frequency-aware live rule against 100 Hz synthetic fixtures."""
import csv, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"synthetic-data"; FS=100; WINDOW=200; STRIDE=10

def classify(window):
    frequency=np.fft.rfftfreq(len(window),1/FS); band=(frequency>=6)&(frequency<=10); broad=(frequency>=1)&(frequency<=20)
    for sensor in range(2):
        values=window[:,sensor*6:(sensor+1)*6]; values-=values.mean(axis=0); power=np.abs(np.fft.rfft(values,axis=0))**2
        ratio=power[band].sum(axis=0)/np.maximum(power[broad].sum(axis=0),1e-12)
        rms=np.sqrt(2*power[band].sum(axis=0))/len(window)
        if np.any((ratio[:3]>.55)&(rms[:3]>.05)) or np.any((ratio[3:]>.55)&(rms[3:]>.015)): return True
    return False

def read(path):
    rows=list(csv.DictReader(path.open(encoding="utf-8")))
    t=np.array([float(r["t"]) for r in rows])
    values=np.array([[float(r[f"s{s}_{axis}"]) for s in (1,2) for axis in ("ax","ay","az","gx","gy","gz")] for r in rows])
    return t,values

manifest=json.loads((DATA/"manifest.json").read_text()); report={"input_rate_hz":FS,"window_s":2,"warning":"Synthetic-fixture performance is not clinical performance.","scenarios":[]}; confusion={"tp":0,"tn":0,"fp":0,"fn":0}
for scenario in manifest["scenarios"]:
    t,values=read(DATA/"100hz"/f"{scenario['name']}.csv"); fine=[e for e in scenario["events"] if e["type"]=="fine_tremor"]; flagged=truth=correct=count=0
    for end in range(WINDOW-1,len(values),STRIDE):
        prediction=classify(values[end-WINDOW+1:end+1].copy()); center=t[end-WINDOW//2]; expected=any(e["start_s"]<=center<=e["end_s"] for e in fine)
        flagged+=prediction; truth+=expected; correct+=(prediction==expected); count+=1; confusion["tp" if prediction and expected else "fp" if prediction else "fn" if expected else "tn"]+=1
    report["scenarios"].append({"name":scenario["name"],"windows":count,"expected_tremor_windows":truth,"flagged_windows":flagged,"flagged_fraction":round(flagged/count,4),"accuracy_against_event_labels":round(correct/count,4)})
report["confusion"]=confusion; report["synthetic_tremor_recall"]=round(confusion["tp"]/(confusion["tp"]+confusion["fn"]),4); report["synthetic_artifact_still_specificity"]=round(confusion["tn"]/(confusion["tn"]+confusion["fp"]),4)
(DATA/"detector-evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
