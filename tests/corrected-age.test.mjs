import assert from "node:assert/strict";
import test from "node:test";

import { calculateCorrectedAge, formatAgeDays, formatPma } from "../app/corrected-age.ts";

test("AAP example: a 32-week infant at 16 chronological weeks is 8 weeks corrected", () => {
  const age = calculateCorrectedAge("2026-01-01", "2026-04-23", 32, 0);
  assert.equal(age.chronologicalAgeDays, 112);
  assert.equal(age.prematurityCorrectionDays, 56);
  assert.equal(age.correctedAgeDays, 56);
  assert.equal(age.postmenstrualAgeDays, 32 * 7 + 112);
  assert.equal(age.useCorrectedAge, true);
});

test("a NICU recording before the due date preserves negative corrected age and PMA", () => {
  const age = calculateCorrectedAge("2026-01-01", "2026-01-08", 32, 0);
  assert.equal(age.correctedAgeDays, -49);
  assert.equal(formatAgeDays(age.correctedAgeDays), "−7w 0d");
  assert.equal(formatPma(age.postmenstrualAgeDays), "33w 0d PMA");
});

test("term infants are not corrected and corrected-age display ends at two years", () => {
  const term = calculateCorrectedAge("2026-01-01", "2026-02-01", 39, 0);
  assert.equal(term.correctedAgeDays, term.chronologicalAgeDays);
  assert.equal(term.useCorrectedAge, false);

  const olderPreterm = calculateCorrectedAge("2024-01-01", "2026-01-01", 32, 0);
  assert.equal(olderPreterm.useCorrectedAge, false);
  assert.equal(olderPreterm.correctedAgeDays, olderPreterm.chronologicalAgeDays - 56);
});

test("study date cannot precede birth", () => {
  assert.throws(() => calculateCorrectedAge("2026-02-01", "2026-01-01", 32, 0), /before date of birth/);
});

