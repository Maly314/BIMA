export default function BrandLockup({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`brand-lockup${compact ? " compact" : ""}`} aria-label="BIMA">
      <img className="brand-mark" src="/bima-icon-192.png" width={compact ? 32 : 44} height={compact ? 32 : 44} alt="" />
      <span className="brand-copy">
        <strong>BIMA</strong>
      </span>
    </div>
  );
}
