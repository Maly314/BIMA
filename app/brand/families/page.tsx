import type { Metadata } from "next";
import ContentPage from "../ContentPage";

export const metadata: Metadata = { title: "For Families | BIMA" };

export default function FamiliesPage() {
  return <ContentPage
    eyebrow="For families"
    title="A calm recording with a clear purpose."
    intro="Families should understand what a study records, what participation feels like, and what the technology can and cannot say about their child."
    image="/brand/families-hero-v2.png"
    imageAlt="Caregivers enjoying relaxed tummy time with their infant"
    sections={[
      { eyebrow:"What a session may involve", title:"Short, gentle, and centered on natural movement.", copy:"A study team may place lightweight motion sensors at the wrists and ankles and position a camera to view the infant. The infant is allowed to move naturally while the system records. The exact procedure, duration, and eligibility depend on the approved research protocol." },
      { eyebrow:"What is collected", title:"Movement data, video, and essential study context.", copy:"A session may include accelerometer and gyroscope readings, video, sensor placement, session timing, corrected-age information, and research notes. Consent materials should identify every data type, who can access it, how long it is retained, and whether it may be reused." },
      { eyebrow:"What BIMA does not do", title:"A recording is not a diagnosis.", copy:"BIMA does not independently diagnose cerebral palsy or another condition. Questions about an infant’s development should be discussed with the child’s pediatrician, neonatology follow-up team, neurologist, or other qualified clinician." },
      { eyebrow:"Family rights", title:"Participation should always be informed and voluntary.", copy:"Research participation should follow ethics-board approval and a clear consent process. Families should know whom to contact, whether they may stop a session, how withdrawal is handled, and whether declining affects clinical care." },
      { eyebrow:"Support", title:"Concerned about movement or development?", copy:"Contact the infant’s healthcare team rather than waiting for a research result. Early conversations can help families understand available developmental surveillance, standardized assessment, and early-intervention resources." },
    ]}
  />;
}
