import { useEffect, useState } from "react";

/** Date, time, and whether the backend is actually answering.
 *
 *  A clock alone only proves the browser tab is running — it keeps ticking
 *  happily after the server dies. So the dot is driven by a health poll, not by
 *  the clock: green means the API answered within the last few seconds, red
 *  means it did not. During a demo that distinction is the whole point of
 *  putting it on screen. */
export default function LiveClock() {
  const [now, setNow] = useState(() => new Date());
  const [alive, setAlive] = useState<boolean | null>(null);

  useEffect(() => {
    const tick = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(tick);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const ping = async () => {
      try {
        const r = await fetch("/api/health", { cache: "no-store" });
        if (!cancelled) setAlive(r.ok);
      } catch {
        if (!cancelled) setAlive(false);
      }
    };
    void ping();
    const h = setInterval(() => void ping(), 5000);
    return () => {
      cancelled = true;
      clearInterval(h);
    };
  }, []);

  const date = now.toLocaleDateString("en-CA", {
    weekday: "short",
    day: "numeric",
    month: "short",
  });
  const time = now.toLocaleTimeString("en-CA", { hour12: false });

  return (
    <div
      className="clock"
      title={
        alive === false
          ? "The API is not responding — the server may have stopped."
          : "The API answered within the last 5 seconds."
      }
    >
      <span className={`dot ${alive === false ? "down" : alive ? "up" : ""}`} />
      <span className="c-date">{date}</span>
      <span className="c-time">{time}</span>
    </div>
  );
}
