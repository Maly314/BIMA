import type { Metadata } from "next";
import ContentPage from "../ContentPage";

export const metadata: Metadata = { title: "Partners | BIMA" };

export default function PartnersPage() {
  return <ContentPage
    eyebrow="Partners"
    title="Build the evidence together."
    intro="BIMA is designed for collaboration across neonatology, developmental pediatrics, rehabilitation, movement science, engineering, data science, and research operations."
    image="/brand/partners-hero.png"
    imageAlt="Clinical and engineering research team collaborating"
    sections={[
      { eyebrow:"Clinical collaborators", title:"Design around real workflows.", copy:"Clinical partners can help define meaningful populations, recording conditions, outcomes, exclusion criteria, artifact labels, and follow-up schedules. Their input is also essential for usability, safety, and interpretation." },
      { eyebrow:"Research collaborators", title:"Create datasets that can answer durable questions.", copy:"Strong studies need preregistered hypotheses, participant-level splits, standardized sensor placement, consistent age definitions, quality-control procedures, and sufficiently diverse cohorts. Shared data dictionaries and versioned exports make later analysis more dependable." },
      { eyebrow:"Engineering collaborators", title:"Treat the acquisition system as a measurement instrument.", copy:"Hardware and software work should document sampling rate, clock drift, dropped samples, calibration, coordinate systems, firmware version, camera properties, and post-processing provenance. Improvements must be tested without changing the meaning of previously collected data." },
      { eyebrow:"Data stewardship", title:"Plan for privacy before the first recording.", copy:"Video and movement data can be sensitive. A healthcare-ready study requires governance for access, encryption, retention, deletion, audit trails, de-identification, incident response, and approved data-sharing agreements. A folder of downloaded files is not enough." },
      { eyebrow:"Connect", title:"Start with the research question.", copy:"The best partnership begins with a specific population, age window, clinical or developmental outcome, and study design. From there, the team can determine whether BIMA’s synchronized capture approach is appropriate and what validation is required." },
    ]}
  />;
}
