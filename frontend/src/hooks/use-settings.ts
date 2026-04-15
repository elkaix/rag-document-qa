import { useCallback, useSyncExternalStore } from "react";

interface Settings {
  model: string;
  topK: number;
}

const STORAGE_KEY = "rag-settings";
const DEFAULTS: Settings = { model: "glm-5.1", topK: 5 };

// BUG FIX: useSyncExternalStore compares snapshots with Object.is().
// BEFORE: getSnapshot() returned a new object on every call, causing
//         infinite re-renders because Object.is({}, {}) is always false.
// AFTER:  Cache the last parsed value and only create a new object when
//         the raw JSON string actually changes.
let _cachedRaw: string | null = null;
let _cachedSettings: Settings = DEFAULTS;

function getSnapshot(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw !== _cachedRaw) {
      _cachedRaw = raw;
      _cachedSettings = raw ? { ...DEFAULTS, ...JSON.parse(raw) } : DEFAULTS;
    }
    return _cachedSettings;
  } catch {
    return DEFAULTS;
  }
}

let listeners: Array<() => void> = [];
function subscribe(cb: () => void) {
  listeners.push(cb);
  return () => {
    listeners = listeners.filter((l) => l !== cb);
  };
}
function emitChange() {
  listeners.forEach((l) => l());
}

export function useSettings() {
  const settings = useSyncExternalStore(subscribe, getSnapshot, () => DEFAULTS);

  const update = useCallback((patch: Partial<Settings>) => {
    const next = { ...getSnapshot(), ...patch };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    emitChange();
  }, []);

  return { settings, update };
}

export const MODEL_OPTIONS = [
  { label: "GLM 5.1", value: "glm-5.1" },
  { label: "GPT-4", value: "gpt-4" },
  { label: "GPT-3.5 Turbo", value: "gpt-3.5-turbo" },
  { label: "Llama 3 (Local)", value: "llama3" },
] as const;
