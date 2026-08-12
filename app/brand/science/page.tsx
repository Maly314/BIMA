import type { Metadata } from "next";
import ContentPage from "../ContentPage";

export const metadata: Metadata = { title: "The Science | BIMA" };

export default function SciencePage() {
  return <ContentPage
    eyebrow="The science"
    title="Movement in context, frame by frame."
    intro="BIMA is built around synchronized observation: motion sensors and video recorded on one timeline, with age and study context kept alongside the data for later analysis."
    image="/brand/science-hero-v2.png"
    imagePosition="right center"
    imageAlt="Infant moving naturally during a calm research observation"
    note="A multimodal research record can preserve both measurable motion and the visual context needed to interpret it."
    sources={[
      { label:"AACPDM Early Detection of Cerebral Palsy Care Pathway", href:"https://www.aacpdm.org/publications/care-pathways/early-detection-of-cerebral-palsy" },
      { label:"WHO recommendations for care of the preterm or low-birth-weight infant", href:"https://www.who.int/publications/i/item/9789240058262" },
      { label:"Markerless Measurement and Evaluation of General Movements in Infants", href:"https://pmc.ncbi.nlm.nih.gov/articles/PMC6989465/" },
      { label:"Deep learning empowered sensor fusion boosts infant movement classification", href:"https://pmc.ncbi.nlm.nih.gov/articles/PMC11733215/" },
    ]}
    sections={[
      { eyebrow:"Spontaneous movement", title:"Observe the infant, not a prompted performance.", copy:"General movements are spontaneous whole-body movement patterns seen in early infancy. Established assessment approaches rely on trained observers and appropriate behavioral state. BIMA does not automate or replace those assessments; it creates aligned raw data that may support future research." },
      { eyebrow:"Synchronized capture", title:"Four sensors and video share the same clock.", copy:"Each sensor records three-axis acceleration and gyroscope signals. Video frames, sensor samples, session time, device sequence information, placement labels, calibration state, and corrected-age fields are kept together so later analyses can be reproduced and audited.", points:["Left and right wrists","Left and right ankles","Frame-aligned timestamps","Raw and derived files kept separate"] },
      { eyebrow:"Age matters", title:"Corrected age keeps preterm development in context.", copy:"For infants born preterm, chronological age alone may not reflect the developmental time that has elapsed since the expected due date. BIMA stores gestational age at birth, chronological age, corrected age, and postmenstrual age so researchers can choose the appropriate measure for a protocol." },
      { eyebrow:"Evidence boundary", title:"No single signal should stand alone.", copy:"Published early-detection pathways emphasize combining standardized neurological examination, movement assessment, imaging, medical history, and clinical judgment. Sensor or video models must be validated on independent infants and across sites before they can support clinical decisions." },
      { eyebrow:"Responsible analysis", title:"Train, test, and report without data leakage.", copy:"Future modeling should separate participants rather than random windows between training and test sets, preserve longer movement sequences, quantify missing data and artifacts, report calibration and subgroup performance, and compare against meaningful clinical baselines." },
    ]}
  />;
}
