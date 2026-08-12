import BrandHeader from "./BrandHeader";
import styles from "./content.module.css";

export type ContentSection = {
  eyebrow: string;
  title: string;
  copy: string;
  points?: string[];
};

export default function ContentPage({
  eyebrow,
  title,
  intro,
  image,
  imagePosition,
  imageAlt,
  sections,
  note,
  sources,
}: {
  eyebrow: string;
  title: string;
  intro: string;
  image: string;
  imagePosition?: string;
  imageAlt: string;
  sections: ContentSection[];
  note?: string;
  sources?: { label: string; href: string }[];
}) {
  return (
    <main className={styles.page}>
      <section className={styles.hero}>
        <img src={image} alt={imageAlt} style={imagePosition ? { objectPosition: imagePosition } : undefined} />
        <BrandHeader />
        <div className={styles.heroCopy}>
          <span>{eyebrow}</span>
          <h1>{title}</h1>
          <p>{intro}</p>
        </div>
        <div className={styles.ribbon} aria-hidden="true" />
      </section>

      <section className={styles.introBand}>
        <p>{note ?? "Careful observation. Responsible research. Better questions about early movement."}</p>
      </section>

      <section className={styles.sections}>
        {sections.map((section, index) => (
          <article className={styles.section} key={section.title}>
            <div className={styles.sectionAccent} aria-hidden="true">
              <span>{String(index + 1).padStart(2, "0")}</span>
            </div>
            <div className={styles.sectionBody}>
              <div className={styles.sectionHeading}>
                <span className={styles.eyebrow}>{section.eyebrow}</span>
                <h2>{section.title}</h2>
              </div>
              <div className={styles.sectionDetail}>
                <p>{section.copy}</p>
              {section.points && (
                <ul>{section.points.map((point) => <li key={point}>{point}</li>)}</ul>
              )}
              </div>
            </div>
          </article>
        ))}
      </section>

      {sources && sources.length > 0 && (
        <section className={styles.sources}>
          <span className={styles.eyebrow}>Selected references</span>
          <h2>Evidence behind the context</h2>
          <div>{sources.map((source) => <a key={source.href} href={source.href} target="_blank" rel="noreferrer">{source.label}<b>↗</b></a>)}</div>
        </section>
      )}

      <footer className={styles.footer}>
        <a className={styles.footerBrand} href="/brand" aria-label="BIMA home">
          <img src="/bima-icon-192.png" alt="" />
          <span><strong>BIMA</strong><small>Biometric Infant<br />Motor Assessment</small></span>
        </a>
        <p>Advancing infant motor research through thoughtful measurement and collaboration.</p>
        <div><a href="/brand/mission">Mission</a><a href="/brand/science">Science</a><a href="/brand/families">Families</a><a href="/brand/partners">Partners</a></div>
      </footer>
    </main>
  );
}
