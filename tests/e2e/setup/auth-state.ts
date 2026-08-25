import path from 'node:path';

/**
 * Paths to the `storageState` files produced by `auth.setup.ts`. Kept in a
 * plain (non-test) module because Playwright forbids importing a `.spec`/
 * setup test file's contents from another test file.
 */
export const AUTH_DIR = path.join(__dirname, '..', '.auth');
export const DEMO_STATE = path.join(AUTH_DIR, 'demo.json');
/** Shared by logging.spec.ts and keyboard-and-mobile.spec.ts. */
export const SHARED_STATE = path.join(AUTH_DIR, 'shared.json');
/** Dedicated to import.spec.ts. */
export const IMPORT_STATE = path.join(AUTH_DIR, 'import.json');
/** Dedicated to assistant.spec.ts's "not enough data" case. */
export const ASSISTANT_FRESH_STATE = path.join(AUTH_DIR, 'assistant-fresh.json');
