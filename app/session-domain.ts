import { calculateCorrectedAge, formatAgeDays, formatPma } from "./corrected-age.ts";

export type Session = {
  patientNumber: number;
  suspected: boolean;
  ageYears: number;
  ageMonths: number;
  ageDays: number;
  studyDate: string;
  weightKg: number;
  dateOfBirth: string;
  gestationalAgeWeeks: number;
  gestationalAgeDays: number;
  gestationalAgeAtBirthDays: number;
  chronologicalAgeDays: number;
  prematurityCorrectionDays: number;
  correctedAgeDays: number;
  postmenstrualAgeDays: number;
  expectedDueDate: string;
  preterm: boolean;
  useCorrectedAge: boolean;
};

export type AgeLike = {
  ageYears: number;
  ageMonths: number;
  ageDays: number;
  correctedAgeDays?: number;
  chronologicalAgeDays?: number;
  postmenstrualAgeDays?: number;
  preterm?: boolean;
  useCorrectedAge?: boolean;
};

export type PatientSessionInput = {
  patientNumber: string;
  nextPatient: number;
  suspected: boolean;
  dateOfBirth: string;
  studyDate: string;
  gestationalWeeks: string;
  gestationalDays: string;
  weight: string;
  weightUnit: "kg" | "lb";
};

export const ageTotalDays = (age: AgeLike) => age.correctedAgeDays ?? (age.ageYears * 365 + age.ageMonths * 30 + age.ageDays);
export const ageWeeks = (age: AgeLike) => Math.floor(ageTotalDays(age) / 7);
export const ageRangeLabel = (age: AgeLike) => age.correctedAgeDays != null
  ? `${ageWeeks(age)}–${ageWeeks(age) + 1} corrected weeks`
  : `${ageWeeks(age)}–${ageWeeks(age) + 1} weeks`;

export const formatAge = (age: AgeLike) => {
  if (age.correctedAgeDays != null && age.chronologicalAgeDays != null) {
    const days = age.useCorrectedAge ? age.correctedAgeDays : age.chronologicalAgeDays;
    return `${formatAgeDays(days)} ${age.useCorrectedAge ? "corrected" : "chronological"}`;
  }
  const parts: string[] = [];
  if (age.ageYears) parts.push(`${age.ageYears}y`);
  if (age.ageMonths) parts.push(`${age.ageMonths}mo`);
  if (age.ageDays) parts.push(`${age.ageDays}d`);
  return parts.length ? parts.join(" ") : "0d";
};

export const detailedAgeLabel = (age: AgeLike) => age.preterm && age.postmenstrualAgeDays != null
  ? `${formatAge(age)} · ${formatPma(age.postmenstrualAgeDays)}`
  : formatAge(age);

export const clinicalAgeMetadata = (session: Session) => ({
  dateOfBirth: session.dateOfBirth,
  gestationalAgeWeeks: session.gestationalAgeWeeks,
  gestationalAgeDays: session.gestationalAgeDays,
  gestationalAgeAtBirthDays: session.gestationalAgeAtBirthDays,
  chronologicalAgeDays: session.chronologicalAgeDays,
  prematurityCorrectionDays: session.prematurityCorrectionDays,
  correctedAgeDays: session.correctedAgeDays,
  postmenstrualAgeDays: session.postmenstrualAgeDays,
  expectedDueDate: session.expectedDueDate,
  preterm: session.preterm,
  useCorrectedAge: session.useCorrectedAge,
});

export function createPatientSession(input: PatientSessionInput): Session {
  const correctedAge = calculateCorrectedAge(
    input.dateOfBirth,
    input.studyDate,
    Number(input.gestationalWeeks),
    Number(input.gestationalDays),
  );
  const parsedPatientNumber = parseInt(input.patientNumber, 10);
  const enteredWeight = Math.max(0, parseFloat(input.weight) || 0);
  const weightKg = input.weightUnit === "lb"
    ? Number((enteredWeight * 0.45359237).toFixed(3))
    : enteredWeight;
  const ageYears = Math.floor(correctedAge.chronologicalAgeDays / 365);
  const afterYears = correctedAge.chronologicalAgeDays - ageYears * 365;
  const ageMonths = Math.floor(afterYears / 30);

  return {
    patientNumber: Number.isFinite(parsedPatientNumber) ? parsedPatientNumber : input.nextPatient,
    suspected: input.suspected,
    ageYears,
    ageMonths,
    ageDays: afterYears - ageMonths * 30,
    studyDate: input.studyDate,
    weightKg,
    dateOfBirth: input.dateOfBirth,
    gestationalAgeWeeks: Number(input.gestationalWeeks),
    gestationalAgeDays: Number(input.gestationalDays),
    ...correctedAge,
  };
}
