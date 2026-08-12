import type { Metadata } from "next";
import BrandHeader from "./BrandHeader";
import styles from "./brand.module.css";

export const metadata: Metadata = {
  title: "BIMA | Every movement tells a story",
  description: "BIMA advances infant motor research through thoughtful measurement and collaboration.",
};

const destinations = [
  {
    href: "/brand/mission",
    eyebrow: "Our Mission",
    title: "Begin with the movement that is already there.",
    copy: "Why careful observation, thoughtful measurement, and clinical partnership guide the work.",
    image: "/brand/mission-hero.png",
    alt: "A calm infant research setting",
    tone: "missionCard",
  },
  {
    href: "/brand/science",
    eyebrow: "The Science",
    title: "See movement in context.",
    copy: "How synchronized video, motion sensors, timing, and corrected age create a richer research record.",
    image: "/brand/science-hero-v2.png",
    alt: "Movement observation supported by video and sensors",
    tone: "scienceCard",
  },
  {
    href: "/brand/families",
    eyebrow: "For Families",
    title: "A gentle session, clearly explained.",
    copy: "What participation may involve, what is collected, and the limits of what a recording can say.",
    image: "/brand/families-hero-v2.png",
    alt: "A welcoming space prepared for a family",
    tone: "familiesCard",
  },
  {
    href: "/brand/partners",
    eyebrow: "For Partners",
    title: "Build stronger studies together.",
    copy: "A place for clinicians, researchers, engineers, and institutions to explore collaboration.",
    image: "/brand/partners-hero.png",
    alt: "Researchers collaborating around movement data",
    tone: "partnersCard",
  },
];

export default function BrandPage() {
  return (
    <main className={styles.page}>
      <section className={styles.homeHero}>
        <img
          className={styles.homeHeroPhoto}
          src="/brand/hero-caregiver-infant.png"
          alt="Caregiver smiling while observing an infant"
        />
        <div className={styles.homeHeroVeil} aria-hidden="true" />
        <BrandHeader />
        <div className={styles.homeHeroCopy}>
          <span className={styles.eyebrow}>Biometric Infant Motor Assessment</span>
          <h1>Every movement<br />tells a story.</h1>
          <p>
            BIMA is exploring how synchronized video and wearable motion sensors can help researchers study early movement with greater context.
          </p>
          <div className={styles.heroActions}>
            <a className={styles.primaryButton} href="/brand/mission">Discover BIMA</a>
            <a className={styles.quietLink} href="/brand/families">For families <span>→</span></a>
          </div>
        </div>
        <div className={styles.homeWaves} aria-hidden="true"><i /><i /><i /></div>
      </section>

      <section className={styles.homeIntro}>
        <span className={styles.eyebrow}>Explore BIMA</span>
        <h2>One purpose.<br />Four places to begin.</h2>
        <p>
          The details belong on their own pages. Choose the perspective that matters to you and explore it without wading through the entire project at once.
        </p>
      </section>

      <section className={styles.destinationGrid} aria-label="Explore BIMA">
        {destinations.map((destination) => (
          <a className={`${styles.destinationCard} ${styles[destination.tone]}`} href={destination.href} key={destination.href}>
            <img src={destination.image} alt={destination.alt} />
            <span className={styles.destinationShade} aria-hidden="true" />
            <div className={styles.destinationCopy}>
              <small>{destination.eyebrow}</small>
              <h3>{destination.title}</h3>
              <p>{destination.copy}</p>
              <b>Explore <span>→</span></b>
            </div>
          </a>
        ))}
      </section>

      <section className={styles.homeBoundary}>
        <div>
          <span className={styles.eyebrow}>Our research boundary</span>
          <h2>Build evidence before making claims.</h2>
        </div>
        <p>
          BIMA is a research data-collection platform. It does not diagnose cerebral palsy or another condition and does not replace qualified clinical judgment.
        </p>
        <a href="/brand/science">Read about the science <span>→</span></a>
      </section>

      <footer className={styles.homeFooter}>
        <a className={styles.footerBrand} href="/brand" aria-label="BIMA home">
          <img src="/bima-icon-192.png" alt="" />
          <span><strong>BIMA</strong><small>Biometric Infant<br />Motor Assessment</small></span>
        </a>
        <p>Advancing infant motor research through thoughtful measurement and collaboration.</p>
        <nav><a href="/brand/mission">Mission</a><a href="/brand/science">Science</a><a href="/brand/families">Families</a><a href="/brand/partners">Partners</a></nav>
      </footer>
    </main>
  );
}
