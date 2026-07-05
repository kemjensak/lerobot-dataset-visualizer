"use client";

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useMemo,
  useEffect,
} from "react";
import type { TrimRange } from "@/utils/trimDetection";

const STORAGE_KEY = "trim-ranges";

function saveToStorage(m: Map<number, TrimRange>) {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify([...m.entries()]));
  } catch {
    /* ignore */
  }
}

type TrimContextType = {
  trims: Map<number, TrimRange>;
  count: number;
  get: (id: number) => TrimRange | undefined;
  set: (id: number, range: TrimRange) => void;
  setMany: (entries: [number, TrimRange][]) => void;
  remove: (id: number) => void;
  clear: () => void;
};

const TrimContext = createContext<TrimContextType | undefined>(undefined);

export function useTrims() {
  const ctx = useContext(TrimContext);
  if (!ctx) throw new Error("useTrims must be used within TrimProvider");
  return ctx;
}

export const TrimProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [trims, setTrims] = useState<Map<number, TrimRange>>(new Map());
  const [hydrated, setHydrated] = useState(false);

  // Hydrate from sessionStorage after mount (avoids SSR/client mismatch)
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (raw) setTrims(new Map(JSON.parse(raw) as [number, TrimRange][]));
    } catch {
      /* ignore */
    }
    setHydrated(true);
  }, []);

  // Only persist after hydration so the initial empty map doesn't
  // overwrite stored trims when the component remounts.
  useEffect(() => {
    if (!hydrated) return;
    saveToStorage(trims);
  }, [trims, hydrated]);

  const set = useCallback((id: number, range: TrimRange) => {
    setTrims((prev) => {
      const next = new Map(prev);
      next.set(id, range);
      return next;
    });
  }, []);

  const setMany = useCallback((entries: [number, TrimRange][]) => {
    setTrims((prev) => {
      const next = new Map(prev);
      for (const [id, range] of entries) next.set(id, range);
      return next;
    });
  }, []);

  const remove = useCallback((id: number) => {
    setTrims((prev) => {
      if (!prev.has(id)) return prev;
      const next = new Map(prev);
      next.delete(id);
      return next;
    });
  }, []);

  const clear = useCallback(() => setTrims(new Map()), []);

  const get = useCallback((id: number) => trims.get(id), [trims]);

  const value = useMemo(
    () => ({ trims, count: trims.size, get, set, setMany, remove, clear }),
    [trims, get, set, setMany, remove, clear],
  );

  return <TrimContext.Provider value={value}>{children}</TrimContext.Provider>;
};
