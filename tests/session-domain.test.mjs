import assert from "node:assert/strict";
import test from "node:test";

import {
  ageRangeLabel,
  clinicalAgeMetadata,
  createPatientSession,
  detailedAgeLabel,
} from "../app/session-domain.ts";

test("patient form values produce corrected-age metadata and normalized pounds", () => {
  const session = createPatientSession({
    patientNumber: "",
    nextPatient: 12,
    suspected: true,
    dateOfBirth: "2026-01-01",
    studyDate: "2026-04-23",
    gestationalWeeks: "32",
    gestationalDays: "0",
    weight: "8",
    weightUnit: "lb",
  });

  assert.equal(session.patientNumber, 12);
  assert.equal(session.weightKg, 3.629);
  assert.equal(session.chronologicalAgeDays, 112);
  assert.equal(session.correctedAgeDays, 56);
  assert.equal(session.useCorrectedAge, true);
  assert.match(detailedAgeLabel(session), /corrected/);
  assert.equal(ageRangeLabel(session), "8–9 corrected weeks");
  assert.deepEqual(clinicalAgeMetadata(session), {
    dateOfBirth: session.dateOfBirth,
    gestationalAgeWeeks: 32,
    gestationalAgeDays: 0,
    gestationalAgeAtBirthDays: 224,
    chronologicalAgeDays: 112,
    prematurityCorrectionDays: 56,
    correctedAgeDays: 56,
    postmenstrualAgeDays: 336,
    expectedDueDate: "2026-02-26",
    preterm: true,
    useCorrectedAge: true,
  });
});

test("term patient form values keep chronological age and kilograms", () => {
  const session = createPatientSession({
    patientNumber: "4",
    nextPatient: 12,
    suspected: false,
    dateOfBirth: "2026-05-01",
    studyDate: "2026-05-11",
    gestationalWeeks: "40",
    gestationalDays: "0",
    weight: "3.45",
    weightUnit: "kg",
  });

  assert.equal(session.patientNumber, 4);
  assert.equal(session.weightKg, 3.45);
  assert.equal(session.correctedAgeDays, 10);
  assert.equal(session.useCorrectedAge, false);
  assert.equal(detailedAgeLabel(session), "1w 3d chronological");
});
