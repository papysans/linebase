// Single source of truth for "what is the user currently working on".
// Mirrors to localStorage so a page refresh or new tab keeps the context.
//
// The nav uses `useSession()` so deep-link buttons like "审查" know which jobId
// to append to `/review/`. Page mounts call `setSession()` when they pick up a
// fresh id from `useParams()` (covers the shared-URL / deep-link case).
import React from "react";

const KEY = "linebase.session.v1";

export interface Session {
  uploadId?: string;
  jobId?: string;
}

function readLS(): Session {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === "object") return parsed as Session;
    return {};
  } catch {
    return {};
  }
}

let mem: Session = readLS();
const listeners = new Set<(s: Session) => void>();

export function getSession(): Session {
  return mem;
}

export function setSession(patch: Partial<Session>): void {
  // Treat `undefined` values as "clear this key", not "leave it alone".
  const next: Session = { ...mem };
  for (const k of Object.keys(patch) as (keyof Session)[]) {
    const v = patch[k];
    if (v === undefined) delete next[k];
    else next[k] = v;
  }
  mem = next;
  try {
    localStorage.setItem(KEY, JSON.stringify(mem));
  } catch {
    /* ignore quota / privacy mode */
  }
  for (const l of listeners) l(mem);
}

export function subscribe(l: (s: Session) => void): () => void {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
}

export function useSession(): Session {
  const [s, setS] = React.useState<Session>(mem);
  React.useEffect(() => subscribe(setS), []);
  return s;
}
