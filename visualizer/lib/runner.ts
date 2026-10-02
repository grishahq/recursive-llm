import {
  validateComparison,
  type Comparison,
  type ReferenceValue,
} from './comparison.ts';

export type Connection = { address: string; token: string };
export type DocumentInfo = {
  document_id: string;
  filename: string;
  format: string;
  bytes: number;
  context_chars: number;
  sha256: string;
  preview: string;
  warnings: string[];
  source_url?: string;
  suggested_question?: string;
};
export type Run = {
  run_id: string;
  state: 'running' | 'completed' | 'failed' | 'cancelling' | 'cancelled';
  phase: string;
  comparison: Comparison | null;
  error?: string;
};
export function localAddress(input: string): string {
  const u = new URL(input);
  if (
    u.protocol !== 'http:' ||
    u.hostname !== '127.0.0.1' ||
    !u.port ||
    u.username ||
    u.password ||
    u.pathname !== '/' ||
    u.search ||
    u.hash
  )
    throw new Error('Use a local address such as http://127.0.0.1:8766.');
  return u.origin;
}
export function parseReferences(
  input: string,
): Record<string, ReferenceValue> | undefined {
  if (!input.trim()) return undefined;
  let v: unknown;
  try {
    v = JSON.parse(input);
  } catch {
    throw new Error(
      'Reference answers must be a JSON object, for example {"count": 12}.',
    );
  }
  if (
    !v ||
    Array.isArray(v) ||
    typeof v !== 'object' ||
    Object.keys(v).length === 0 ||
    Object.keys(v).length > 20 ||
    Object.entries(v).some(
      ([k, value]) =>
        !/^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(k) ||
        !(
          (typeof value === 'string' && value.length <= 1000) ||
          typeof value === 'boolean' ||
          (typeof value === 'number' && Number.isFinite(value))
        ),
    )
  )
    throw new Error(
      'Use field names with letters, digits or underscores, and string, number or boolean values.',
    );
  return v as Record<string, ReferenceValue>;
}
export async function runnerRequest<T>(
  connection: Connection,
  path: string,
  body?: unknown,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${localAddress(connection.address)}${path}`, {
      method: body === undefined ? 'GET' : 'POST',
      headers: {
        Authorization: `Bearer ${connection.token}`,
        ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: 'omit',
      cache: 'no-store',
      redirect: 'error',
      signal: AbortSignal.timeout(60_000),
    });
  } catch {
    throw new Error(
      'Cannot reach local Codex. Keep the companion running, check its address, and allow local network access in your browser.',
    );
  }
  const data = (await response.json()) as Record<string, unknown>;
  if (!response.ok)
    throw new Error(
      typeof data.error === 'string'
        ? data.error
        : `Local runner returned ${response.status}.`,
    );
  return data as T;
}
export function validateRun(raw: Run): Run {
  if (
    !raw ||
    typeof raw.run_id !== 'string' ||
    !/^[A-Za-z0-9_-]+$/.test(raw.run_id) ||
    !['running', 'completed', 'failed', 'cancelling', 'cancelled'].includes(
      raw.state,
    ) ||
    typeof raw.phase !== 'string'
  )
    throw new Error('Unexpected response from the local runner.');
  return {
    ...raw,
    comparison: raw.comparison ? validateComparison(raw.comparison) : null,
  };
}
