const DAY_MS = 86_400_000;
export const TERM_REFERENCE_DAYS = 40 * 7;
export const PRETERM_THRESHOLD_DAYS = 37 * 7;
export const CORRECTED_AGE_USE_LIMIT_DAYS = 2 * 365;

export type CorrectedAgeResult = {
  chronologicalAgeDays: number;
  gestationalAgeAtBirthDays: number;
  prematurityCorrectionDays: number;
  correctedAgeDays: number;
  postmenstrualAgeDays: number;
  expectedDueDate: string;
  preterm: boolean;
  useCorrectedAge: boolean;
};

function utcDay(date: string): number {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(date);
  if (!match) throw new Error("Enter a valid date");
  const value = Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
  if (!Number.isFinite(value)) throw new Error("Enter a valid date");
  return value;
}

function isoDateFromUtcDay(value: number): string {
  return new Date(value).toISOString().slice(0, 10);
}

export function calculateCorrectedAge(
  dateOfBirth: string,
  studyDate: string,
  gestationalWeeks: number,
  gestationalDays: number,
): CorrectedAgeResult {
  if (!Number.isInteger(gestationalWeeks) || gestationalWeeks < 20 || gestationalWeeks > 45) {
    throw new Error("Gestational age must be between 20 and 45 weeks");
  }
  if (!Number.isInteger(gestationalDays) || gestationalDays < 0 || gestationalDays > 6) {
    throw new Error("Gestational days must be between 0 and 6");
  }
  const birthDay = utcDay(dateOfBirth);
  const studyDay = utcDay(studyDate);
  const chronologicalAgeDays = Math.round((studyDay - birthDay) / DAY_MS);
  if (chronologicalAgeDays < 0) throw new Error("Study date cannot be before date of birth");

  const gestationalAgeAtBirthDays = gestationalWeeks * 7 + gestationalDays;
  const preterm = gestationalAgeAtBirthDays < PRETERM_THRESHOLD_DAYS;
  // AAP corrected age: chronological age minus the time born before 40 weeks.
  // Correction applies to preterm infants; term infants retain chronological age.
  const prematurityCorrectionDays = preterm
    ? Math.max(0, TERM_REFERENCE_DAYS - gestationalAgeAtBirthDays)
    : 0;
  const correctedAgeDays = chronologicalAgeDays - prematurityCorrectionDays;
  const postmenstrualAgeDays = gestationalAgeAtBirthDays + chronologicalAgeDays;

  return {
    chronologicalAgeDays,
    gestationalAgeAtBirthDays,
    prematurityCorrectionDays,
    correctedAgeDays,
    postmenstrualAgeDays,
    expectedDueDate: isoDateFromUtcDay(birthDay + Math.max(0, TERM_REFERENCE_DAYS - gestationalAgeAtBirthDays) * DAY_MS),
    preterm,
    useCorrectedAge: preterm && chronologicalAgeDays < CORRECTED_AGE_USE_LIMIT_DAYS,
  };
}

export function formatAgeDays(days: number): string {
  const sign = days < 0 ? "−" : "";
  const absolute = Math.abs(days);
  const weeks = Math.floor(absolute / 7);
  const remainder = absolute % 7;
  if (!weeks) return `${sign}${remainder}d`;
  return `${sign}${weeks}w ${remainder}d`;
}

export function formatPma(days: number): string {
  return `${Math.floor(days / 7)}w ${days % 7}d PMA`;
}

