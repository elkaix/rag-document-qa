/**
 * useDebounce — delays propagating a value until it has been stable for
 * `delay` ms.
 *
 * WHY a separate hook:
 *   Debouncing is used in multiple components (search inputs, live filters).
 *   Extracting to a shared hook avoids duplicate useEffect+setTimeout logic
 *   and keeps each component file under the 250-line guideline.
 *
 * PATTERN: useEffect + setTimeout is the canonical lightweight debounce.
 *   For heavy computation debouncing, a library hook (e.g., use-debounce) is
 *   worth the dependency — for simple UI filters this is sufficient.
 */

import { useEffect, useRef, useState } from "react";

export function useDebounce<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (timerRef.current !== null) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setDebounced(value), delay);
    return () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current);
    };
  }, [value, delay]);

  return debounced;
}
