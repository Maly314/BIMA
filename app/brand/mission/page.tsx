import type { Metadata } from "next";
import ContentPage from "../ContentPage";

export const metadata: Metadata = { title: "Our Mission | BIMA" };

export default function MissionPage() {
  return <ContentPage
    eyebrow="Our mission"
    title="Make early movement easier to understand."
    intro="BIMA brings engineering, infant-development research, and careful clinical collaboration together around one goal: collecting clearer movement data without losing the human context around it."
    image="/brand/mission-hero.png"
    imageAlt="Researcher speaking with a caregiver holding an infant"
    sections={[
      { eyebrow:"Why movement", title:"The earliest patterns deserve careful attention.", copy:"Spontaneous infant movement carries rich information about developing motor control. Today, expert observation remains essential. BIMA is exploring how synchronized video and small wearable motion sensors can create an objective record that researchers can review alongside established clinical methods.", points:["Observe natural movement","Preserve timing and context","Support reproducible research","Keep clinicians in the loop"] },
      { eyebrow:"Our approach", title:"Measure thoughtfully, not intrusively.", copy:"The system is designed around short, calm recordings. Sensors capture acceleration and rotation while video preserves the visible movement context. Both streams share a session clock so researchers can compare what the camera saw with what the sensors measured." },
      { eyebrow:"Long-term direction", title:"Build evidence before making claims.", copy:"BIMA’s near-term purpose is high-quality data collection and research. Any future clinical use would require representative datasets, predefined outcomes, external validation, human-factors testing, privacy safeguards, regulatory review, and prospective clinical evaluation." },
      { eyebrow:"How we work", title:"Families, clinicians, and researchers belong in the design process.", copy:"Useful technology begins with the people who will encounter it. We aim to make sessions understandable for families, efficient for study teams, and transparent about what is collected, why it is collected, and how it may be used." },
    ]}
  />;
}
