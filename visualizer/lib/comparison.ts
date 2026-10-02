export type Step = {
  at: number;
  title: string;
  detail: string;
  code?: string;
  output?: string;
  node_id?: string;
  parent_id?: string;
  kind?: string;
};
export type ReferenceValue = string | number | boolean;
export type Quality = {
  status: 'graded' | 'manual_review' | 'unavailable';
  matched: number | null;
  total: number | null;
  fields: {
    label: string;
    expected: ReferenceValue;
    observed: ReferenceValue | null;
    passed: boolean;
    accepted_aliases?: string[];
  }[];
};
export type Lane = {
  tokens: number | null;
  cached_tokens?: number | null;
  cost: number | null;
  seconds: number;
  calls: number;
  result: string;
  passed: string | null;
  steps: Step[];
  error?: string | null;
  quality?: Quality;
};
export type Comparison = {
  schema_version: 1;
  kind: 'recorded';
  title: string;
  model: string;
  reasoning_effort?: string;
  max_depth?: number;
  query: string;
  original_query?: string;
  context_chars: number;
  context_preview: string;
  sample_note: string;
  source_url?: string;
  evaluation_label?: string;
  rlm: Lane;
  direct: Lane;
};
export function validateComparison(raw: unknown): Comparison {
  const v = raw as Comparison;
  const finite = (n: unknown) =>
    typeof n === 'number' && Number.isFinite(n) && n >= 0;
  if (
    !v ||
    v.schema_version !== 1 ||
    v.kind !== 'recorded' ||
    !['title', 'model', 'query', 'context_preview', 'sample_note'].every(
      (k) => typeof (v as unknown as Record<string, unknown>)[k] === 'string',
    ) ||
    !finite(v.context_chars)
  )
    throw new Error(
      'Choose a JSON file exported by the RLM comparison runner.',
    );
  for (const l of [v.rlm, v.direct])
    if (
      !l ||
      !finite(l.seconds) ||
      !finite(l.calls) ||
      !(l.tokens === null || finite(l.tokens)) ||
      !(l.cost === null || finite(l.cost)) ||
      !(
        l.cached_tokens === undefined ||
        l.cached_tokens === null ||
        finite(l.cached_tokens)
      ) ||
      !(
        l.error === undefined ||
        l.error === null ||
        typeof l.error === 'string'
      ) ||
      typeof l.result !== 'string' ||
      !(l.passed === null || typeof l.passed === 'string') ||
      !Array.isArray(l.steps) ||
      l.steps.length > 5000 ||
      l.steps.some(
        (s) =>
          !s ||
          !finite(s.at) ||
          typeof s.title !== 'string' ||
          typeof s.detail !== 'string' ||
          ['code', 'output', 'node_id', 'parent_id', 'kind'].some(
            (k) =>
              (s as Record<string, unknown>)[k] !== undefined &&
              typeof (s as Record<string, unknown>)[k] !== 'string',
          ),
      )
    )
      throw new Error('The comparison contains missing or invalid run data.');
  const primitive = (x: unknown) =>
    typeof x === 'string' ||
    typeof x === 'boolean' ||
    (typeof x === 'number' && Number.isFinite(x));
  for (const l of [v.rlm, v.direct]) {
    const q = l.quality;
    if (
      q !== undefined &&
      (!q ||
        !['graded', 'manual_review', 'unavailable'].includes(q.status) ||
        !(
          q.matched === null ||
          (Number.isInteger(q.matched) && q.matched >= 0)
        ) ||
        !(q.total === null || (Number.isInteger(q.total) && q.total >= 0)) ||
        (q.matched !== null && q.total !== null && q.total < q.matched) ||
        !Array.isArray(q.fields) ||
        q.fields.length > 100 ||
        q.fields.some(
          (f) =>
            !f ||
            typeof f.label !== 'string' ||
            !primitive(f.expected) ||
            !(f.observed === null || primitive(f.observed)) ||
            typeof f.passed !== 'boolean' ||
            (f.accepted_aliases !== undefined &&
              (!Array.isArray(f.accepted_aliases) ||
                f.accepted_aliases.length > 20 ||
                f.accepted_aliases.some((a) => typeof a !== 'string'))),
        ) ||
        (q.status === 'graded' &&
          (q.matched === null ||
            q.total === null ||
            q.total !== q.fields.length ||
            q.matched !== q.fields.filter((f) => f.passed).length)))
    )
      throw new Error('The comparison contains invalid quality results.');
  }
  if (
    v.evaluation_label !== undefined &&
    typeof v.evaluation_label !== 'string'
  )
    throw new Error('Invalid evaluation note.');
  if (v.original_query !== undefined && typeof v.original_query !== 'string')
    throw new Error('Invalid original question.');
  if (
    (v.reasoning_effort !== undefined &&
      typeof v.reasoning_effort !== 'string') ||
    (v.max_depth !== undefined &&
      (!Number.isInteger(v.max_depth) || v.max_depth < 0))
  )
    throw new Error('Invalid model configuration.');
  return {
    ...v,
    source_url:
      typeof v.source_url === 'string' && v.source_url.startsWith('https://')
        ? v.source_url
        : undefined,
    rlm: { ...v.rlm, steps: [...v.rlm.steps].sort((a, b) => a.at - b.at) },
    direct: {
      ...v.direct,
      steps: [...v.direct.steps].sort((a, b) => a.at - b.at),
    },
  };
}
