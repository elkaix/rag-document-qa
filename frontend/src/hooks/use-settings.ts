import { useCallback, useSyncExternalStore } from "react";

interface Settings {
  model: string;
  topK: number;
}

const STORAGE_KEY = "rag-settings";
// WHY gpt-5-mini as default: best speed/quality/cost balance on OpenAI's
// current lineup (April 2026) — ~250 tok/s streaming, $0.25/$2.00 per 1M,
// 400K context. Small enough to feel instant in chat, capable enough for
// RAG answers grounded in retrieved context.
const DEFAULTS: Settings = { model: "gpt-5-mini", topK: 5 };

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

// WHY this lineup (April 2026): OpenAI retired GPT-4 / GPT-3.5 Turbo for new
// work; the current fast tier is the GPT-5 family. Ordered by recommended
// default first (balanced) -> ultra-fast -> premium -> long-context -> other.
//
// PATTERN: label = short canonical name (fits one line in the sidebar pill);
//          hint  = one-word tier descriptor rendered dimmer beside the label.
//          Splitting them keeps menu items from wrapping and keeps the
//          radio-item checkmark aligned to a single row.
export const MODEL_OPTIONS = [
  { label: "GPT-5 Mini",   hint: "fast default", value: "gpt-5-mini" },
  { label: "GPT-5.4 Nano", hint: "fastest",      value: "gpt-5.4-nano" },
  { label: "GPT-5.4",      hint: "premium",      value: "gpt-5.4" },
  { label: "GPT-4.1 Mini", hint: "1M ctx",       value: "gpt-4.1-mini" },
  { label: "GLM 5.1",      hint: "",             value: "glm-5.1" },
  { label: "Llama 3",      hint: "local",        value: "llama3" },
] as const;
