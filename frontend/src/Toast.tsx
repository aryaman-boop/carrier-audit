import { useEffect } from "react";

export interface ToastMsg {
  id: number;
  tone: "warn" | "bad";
  gate: string;
  title: string;
  body?: string;
  cost?: number;
}

function One({ t, onClose }: { t: ToastMsg; onClose: (id: number) => void }) {
  useEffect(() => {
    // Long enough to read two sentences, short enough not to sit on the screen
    // during a demo.
    const h = setTimeout(() => onClose(t.id), 7000);
    return () => clearTimeout(h);
  }, [t.id, onClose]);

  return (
    <div className={`toast ${t.tone}`} role="status" aria-live="polite">
      <button className="t-close" onClick={() => onClose(t.id)} aria-label="Dismiss">
        ×
      </button>
      <div className="t-head">
        <span className="t-gate">{t.gate}</span>
        <span>{t.title}</span>
      </div>
      {t.body && <div className="t-body">{t.body}</div>}
      {t.cost !== undefined && (
        <div className="t-cost">
          {t.cost === 0 ? "$0.00 — no model was called" : `$${t.cost.toFixed(4)}`}
        </div>
      )}
    </div>
  );
}

export default function Toasts({
  items,
  onClose,
}: {
  items: ToastMsg[];
  onClose: (id: number) => void;
}) {
  if (items.length === 0) return null;
  return (
    <div className="toasts">
      {items.map((t) => (
        <One key={t.id} t={t} onClose={onClose} />
      ))}
    </div>
  );
}
