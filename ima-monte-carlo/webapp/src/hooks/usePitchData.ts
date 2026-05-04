import { useEffect, useState } from "react";
import type { PitchData } from "@/data/types";

export function usePitchData(id: string | undefined) {
  const [data, setData] = useState<PitchData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    fetch(`${import.meta.env.BASE_URL}pitches/${id}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(`Pitch ${id} not found`)))
      .then((d) => {
        if (!cancelled) setData(d as PitchData);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  return { data, loading, error };
}
