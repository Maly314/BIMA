import styles from "./brand.module.css";

export default function BrandHeader() {
  return (
    <header className={styles.header}>
      <a className={styles.logo} href="/brand" aria-label="BIMA home">
        <img src="/bima-logo.png" alt="BIMA, Biometric Infant Motor Assessment" />
      </a>
      <div className={styles.navIsland}>
        <nav className={styles.nav} aria-label="Main navigation">
          <a href="/brand/mission">Our Mission</a>
          <a href="/brand/science">The Science</a>
          <a href="/brand/families">Families</a>
          <a href="/brand/partners">Partners</a>
        </nav>
        <a className={styles.primaryButton} href="/brand/partners#connect">Collaborate With Us</a>
      </div>
    </header>
  );
}
