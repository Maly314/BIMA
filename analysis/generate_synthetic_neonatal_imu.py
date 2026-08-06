"""Research-constrained synthetic two-IMU movement generator.

These traces are engineering test fixtures, not patient data and not clinical
ground truth. Generate at 100 Hz, then create a separate 10 Hz legacy view to
show what the current serial stream preserves (and aliases).
"""

from __future__ import annotations

import argparse, csv, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

G = 9.80665
FS = 100
DURATION_S = 30
ACCEL_NOISE_DENSITY_G = 230e-6       # ICM-20948 typical, g/sqrt(Hz)
GYRO_NOISE_DENSITY_DPS = 0.015       # ICM-20948 typical, deg/s/sqrt(Hz)
ACCEL_RMS = ACCEL_NOISE_DENSITY_G * G * np.sqrt(FS / 2)
GYRO_RMS = np.deg2rad(GYRO_NOISE_DENSITY_DPS) * np.sqrt(FS / 2)


@dataclass(frozen=True)
class Scenario:
    name: str
    expected: str
    description: str


SCENARIOS = [
    Scenario("still_threshold", "not_shaky", "Quiet sensor noise, small bias drift, and sub-threshold incubator sway."),
    Scenario("fine_tremor_low", "shaky", "Low-amplitude, narrow-band 6.5-8.5 Hz tremor with slow amplitude/frequency drift."),
    Scenario("fine_tremor_intermittent", "shaky_during_events", "Three short fine-tremor bouts separated by genuine stillness."),
    Scenario("spontaneous_movement", "not_shaky", "Irregular, asymmetric smooth limb-movement bouts without sustained periodicity."),
    Scenario("caregiver_handling", "artifact", "Large low-frequency, strongly bilateral common-mode handling/repositioning."),
    Scenario("bed_impact", "artifact", "Brief shared impulses with damped mechanical ring-down."),
    Scenario("sensor_slip", "artifact", "One-sensor orientation step with a short attachment transient."),
    Scenario("cable_vibration", "artifact", "Short high-frequency vibration isolated to one sensor."),
    Scenario("mixed_artifacts", "artifact", "Stillness plus handling, two impacts, sensor slip, and cable vibration."),
]


def smooth_gate(t: np.ndarray, start: float, stop: float, edge: float = .2) -> np.ndarray:
    return .5 * (np.tanh((t - start) / edge) - np.tanh((t - stop) / edge))


def oscillator(t: np.ndarray, rng: np.random.Generator, low_hz: float, high_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = rng.uniform(low_hz, high_hz)
    drift = .25 * np.sin(2*np.pi*rng.uniform(.035,.08)*t + rng.uniform(0,2*np.pi))
    frequency = center + drift
    phase = 2*np.pi*np.cumsum(frequency) / FS + rng.uniform(0,2*np.pi)
    return np.sin(phase), np.cos(phase), frequency


def add_fine_tremor(t: np.ndarray, a: np.ndarray, g: np.ndarray, sensor: int, rng: np.random.Generator, gate: np.ndarray) -> None:
    wave, quadrature, freq = oscillator(t, rng, 6.5, 8.5)
    envelope = gate * (.68 + .22*np.sin(2*np.pi*.18*t + rng.uniform(0,2*np.pi)))
    displacement_m = rng.uniform(.00025, .00085)  # explicit engineering assumption; configurable
    angular_rad = np.deg2rad(rng.uniform(.25, .8))
    acceleration = displacement_m * (2*np.pi*freq)**2 * wave * envelope
    angular_velocity = angular_rad * 2*np.pi*freq * quadrature * envelope
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    a[:, sensor, :] += acceleration[:, None] * axis
    g[:, sensor, :] += angular_velocity[:, None] * np.roll(axis, 1)


def add_smooth_bout(t: np.ndarray, a: np.ndarray, g: np.ndarray, sensor: int, rng: np.random.Generator, start: float, duration: float) -> None:
    gate = smooth_gate(t, start, start+duration, .12)
    f = rng.uniform(.45,1.5); phase=rng.uniform(0,2*np.pi)
    axis=rng.normal(size=3); axis/=np.linalg.norm(axis)
    a[:,sensor,:] += (rng.uniform(.35,1.6)*np.sin(2*np.pi*f*t+phase)*gate)[:,None]*axis
    g[:,sensor,:] += (rng.uniform(.15,.8)*np.cos(2*np.pi*f*t+phase)*gate)[:,None]*np.roll(axis,1)


def add_impact(t: np.ndarray, a: np.ndarray, g: np.ndarray, rng: np.random.Generator, at: float, strength: float=1.) -> None:
    elapsed=np.maximum(0,t-at); ring=np.exp(-elapsed/0.18)*np.sin(2*np.pi*17*elapsed)*(t>=at)
    axis=rng.normal(size=3); axis/=np.linalg.norm(axis)
    for sensor, scale in enumerate((1,.82)):
        a[:,sensor,:] += (strength*5.5*scale*ring)[:,None]*axis
        g[:,sensor,:] += (strength*1.3*scale*ring)[:,None]*np.roll(axis,1)


def simulate(scenario: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rng=np.random.default_rng(seed); t=np.arange(0,DURATION_S,1/FS)
    a=np.zeros((len(t),2,3)); g=np.zeros_like(a); events=[]
    # Slightly different resting orientations and slowly varying bias per unit.
    a[:,0,:]=[.06,-.12,G]; a[:,1,:]=[-.09,.08,G]
    a += rng.normal(0,ACCEL_RMS,size=a.shape)
    g += rng.normal(0,GYRO_RMS,size=g.shape)
    drift=np.cumsum(rng.normal(0,1.5e-5,size=(len(t),2,3)),axis=0)
    a += drift
    sway=.012*np.sin(2*np.pi*.35*t)
    a[:,:,0] += sway[:,None]

    if scenario == "fine_tremor_low":
        gate=smooth_gate(t,2,DURATION_S-2,.35)
        for s in range(2): add_fine_tremor(t,a,g,s,rng,gate)
        events=[{"start_s":2,"end_s":28,"type":"fine_tremor"}]
    elif scenario == "fine_tremor_intermittent":
        for start,stop in [(3,7),(12,17),(22,27)]:
            gate=smooth_gate(t,start,stop,.22)
            for s in range(2): add_fine_tremor(t,a,g,s,rng,gate)
            events.append({"start_s":start,"end_s":stop,"type":"fine_tremor"})
    elif scenario == "spontaneous_movement":
        for s in range(2):
            for start in np.sort(rng.uniform(2,27,6)): add_smooth_bout(t,a,g,s,rng,float(start),rng.uniform(.35,1.2))
        events=[{"start_s":2,"end_s":28,"type":"irregular_spontaneous_movement"}]
    elif scenario == "caregiver_handling":
        gate=smooth_gate(t,7,18,.45); common=1.8*np.sin(2*np.pi*.55*t)*gate
        a[:,:,0] += common[:,None]; a[:,:,2] += (.7*np.sin(2*np.pi*.31*t+1)*gate)[:,None]
        g[:,:,1] += (1.1*np.cos(2*np.pi*.55*t)*gate)[:,None]
        events=[{"start_s":7,"end_s":18,"type":"caregiver_handling"}]
    elif scenario == "bed_impact":
        for at,strength in [(8,1),(18,.7),(24,1.15)]: add_impact(t,a,g,rng,at,strength); events.append({"start_s":at,"end_s":at+.8,"type":"bed_impact"})
    elif scenario == "sensor_slip":
        gate=smooth_gate(t,12,DURATION_S,.05); a[:,0,0]+=1.3*gate; a[:,0,2]-=.11*gate
        add_impact(t,a,g,rng,12,.35); events=[{"start_s":12,"end_s":12.8,"type":"sensor_slip_transient"},{"start_s":12,"end_s":30,"type":"orientation_offset"}]
    elif scenario == "cable_vibration":
        gate=smooth_gate(t,10,13,.04); wave=np.sin(2*np.pi*24*t)*gate
        a[:,0,1]+=1.4*wave; g[:,0,2]+=.65*wave
        events=[{"start_s":10,"end_s":13,"type":"cable_vibration"}]
    elif scenario == "mixed_artifacts":
        gate=smooth_gate(t,4,10,.35); common=1.2*np.sin(2*np.pi*.48*t)*gate
        a[:,:,0]+=common[:,None]; g[:,:,1]+=(.7*np.cos(2*np.pi*.48*t)*gate)[:,None]
        add_impact(t,a,g,rng,14,.8); add_impact(t,a,g,rng,22,1.1)
        slip=smooth_gate(t,17,DURATION_S,.06); a[:,1,1]+=.9*slip; a[:,1,2]-=.05*slip
        vib=smooth_gate(t,25,27,.04)*np.sin(2*np.pi*21*t); a[:,0,1]+=.9*vib; g[:,0,2]+=.5*vib
        events=[{"start_s":4,"end_s":10,"type":"handling"},{"start_s":14,"end_s":14.8,"type":"impact"},{"start_s":17,"end_s":30,"type":"sensor_slip"},{"start_s":22,"end_s":22.8,"type":"impact"},{"start_s":25,"end_s":27,"type":"cable_vibration"}]
    return t,a,g,events


def movement(a: np.ndarray) -> np.ndarray:
    return np.maximum(0,np.abs(np.linalg.norm(a,axis=2)-G)-.0625)


def write_csv(path: Path, t: np.ndarray, a: np.ndarray, g: np.ndarray, indices: np.ndarray) -> None:
    mov=movement(a); path.parent.mkdir(parents=True,exist_ok=True)
    fields=["t"]+[f"s{s}_{axis}" for s in (1,2) for axis in ("ax","ay","az","gx","gy","gz","mov")]
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for i in indices:
            row={"t":f"{t[i]:.3f}"}
            for s in range(2):
                for j,axis in enumerate(("ax","ay","az")): row[f"s{s+1}_{axis}"]=f"{a[i,s,j]:.6f}"
                for j,axis in enumerate(("gx","gy","gz")): row[f"s{s+1}_{axis}"]=f"{g[i,s,j]:.6f}"
                row[f"s{s+1}_mov"]=f"{mov[i,s]:.6f}"
            writer.writerow(row)


def summarize(t: np.ndarray,a: np.ndarray,g: np.ndarray) -> dict:
    mov=movement(a); gyro=np.linalg.norm(g,axis=2)
    dynamic=a[:,0,:]-a[:,0,:].mean(axis=0); axis=int(np.argmax(dynamic.std(axis=0))); centered=dynamic[:,axis]
    freq=np.fft.rfftfreq(len(t),1/FS); power=np.abs(np.fft.rfft(centered))**2
    band=(freq>=1)&(freq<=20); peak=float(freq[band][np.argmax(power[band])])
    return {"rows_100hz":len(t),"movement_p50":round(float(np.median(mov)),4),"movement_p99":round(float(np.quantile(mov,.99)),4),"gyro_p99_rad_s":round(float(np.quantile(gyro,.99)),4),"dominant_raw_accel_1_20hz_sensor1":round(peak,3)}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=Path("synthetic-data")); parser.add_argument("--seed",type=int,default=20948); args=parser.parse_args()
    manifest={"synthetic":True,"clinical_ground_truth":False,"sample_rates_hz":{"ground_truth":100,"legacy_app_view":10},"duration_s":DURATION_S,"seed":args.seed,"assumptions":{"fine_tremor_frequency_hz":[6.5,8.5],"fine_tremor_displacement_m":[.00025,.00085],"still_movement_target_m_s2":"99th percentile below 0.12","warning":"Amplitude range is an engineering stress-test assumption because published neonatal IMU amplitude distributions were not found."},"scenarios":[]}
    for n,s in enumerate(SCENARIOS):
        t,a,g,events=simulate(s.name,args.seed+n)
        write_csv(args.output/"100hz"/f"{s.name}.csv",t,a,g,np.arange(len(t)))
        write_csv(args.output/"10hz_legacy"/f"{s.name}.csv",t,a,g,np.arange(0,len(t),10))
        manifest["scenarios"].append({"name":s.name,"expected":s.expected,"description":s.description,"events":events,"metrics_100hz":summarize(t,a,g)})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__": main()
