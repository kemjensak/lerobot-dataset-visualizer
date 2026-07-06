"use client";

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useMemo,
  useEffect,
} from "react";
import type { NumericEdit } from "@/utils/numericEdit";

const STORAGE_KEY = "numeric-edits";

function saveToStorage(edits: NumericEdit[]) {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(edits));
  } catch {
    /* ignore */
  }
}

type EditContextType = {
  edits: NumericEdit[];
  count: number;
  add: (edit: Omit<NumericEdit, "id">) => void;
  remove: (id: string) => void;
  clear: () => void;
};

const EditContext = createContext<EditContextType | undefined>(undefined);

export function useNumericEdits() {
  const ctx = useContext(EditContext);
  if (!ctx) throw new Error("useNumericEdits must be used within EditProvider");
  return ctx;
}

export const EditProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [edits, setEdits] = useState<NumericEdit[]>([]);
  const [hydrated, setHydrated] = useState(false);

  // Hydrate from sessionStorage after mount (avoids SSR/client mismatch)
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (raw) setEdits(JSON.parse(raw) as NumericEdit[]);
    } catch {
      /* ignore */
    }
    setHydrated(true);
  }, []);

  // Only persist after hydration so the initial empty list doesn't
  // overwrite stored edits when the component remounts.
  useEffect(() => {
    if (!hydrated) return;
    saveToStorage(edits);
  }, [edits, hydrated]);

  const add = useCallback((edit: Omit<NumericEdit, "id">) => {
    setEdits((prev) => [
      ...prev,
      { ...edit, id: `${Date.now()}-${prev.length}` },
    ]);
  }, []);

  const remove = useCallback((id: string) => {
    setEdits((prev) => prev.filter((e) => e.id !== id));
  }, []);

  const clear = useCallback(() => setEdits([]), []);

  const value = useMemo(
    () => ({ edits, count: edits.length, add, remove, clear }),
    [edits, add, remove, clear],
  );

  return <EditContext.Provider value={value}>{children}</EditContext.Provider>;
};
