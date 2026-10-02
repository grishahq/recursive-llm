import test from 'node:test';
import assert from 'node:assert/strict';
import { validateComparison } from '../lib/comparison.ts';
const fixture = () => ({
  schema_version: 1,
  kind: 'recorded',
  title: 'Test',
  model: 'gpt-5.6-luna',
  query: 'Query',
  context_chars: 30,
  context_preview: 'source',
  sample_note: 'single run',
  rlm: {
    tokens: 40,
    cost: null,
    calls: 2,
    seconds: 3,
    result: 'answer',
    passed: '4 / 4',
    steps: [
      { at: 2, title: 'Second', detail: 'done' },
      { at: 0, title: 'First', detail: 'start' },
    ],
  },
  direct: {
    tokens: 60,
    cost: null,
    calls: 1,
    seconds: 2,
    result: 'answer',
    passed: '4 / 4',
    steps: [],
  },
});
void test('sorts event replay without mutating imported data', () => {
  const original = fixture();
  const result = validateComparison(original);
  assert.equal(result.rlm.steps[0].at, 0);
  assert.equal(original.rlm.steps[0].at, 2);
});
void test('keeps unknown token use unknown', () => {
  const v = fixture();
  (v.rlm as { tokens: number | null }).tokens = null;
  assert.equal(validateComparison(v).rlm.tokens, null);
});
void test('preserves configured model and depth while accepting historical recordings', () => {
  const historical = validateComparison(fixture());
  assert.equal(historical.reasoning_effort, undefined);
  assert.equal(historical.max_depth, undefined);
  const current = validateComparison({
    ...fixture(),
    model: 'gpt-6.1-sol',
    reasoning_effort: 'low',
    max_depth: 0,
  });
  assert.equal(current.model, 'gpt-6.1-sol');
  assert.equal(current.reasoning_effort, 'low');
  assert.equal(current.max_depth, 0);
  for (const configuration of [
    { reasoning_effort: 3 },
    { max_depth: -1 },
    { max_depth: 1.5 },
    { max_depth: Infinity },
  ])
    assert.throws(() => validateComparison({ ...fixture(), ...configuration }));
});
void test('rejects malformed, nonfinite, negative, and unsupported records', () => {
  for (const v of [
    null,
    {},
    { ...fixture(), schema_version: 2 },
    { ...fixture(), context_chars: -1 },
    { ...fixture(), rlm: { ...fixture().rlm, seconds: Infinity } },
    {
      ...fixture(),
      direct: {
        ...fixture().direct,
        steps: [{ at: 0, title: [], detail: 'bad' }],
      },
    },
  ])
    assert.throws(() => validateComparison(v));
});
void test('removes executable URLs without interpreting model content', () => {
  const raw = { ...fixture(), source_url: 'javascript:alert(1)' };
  raw.rlm.result = '<script>alert(1)</script>';
  const result = validateComparison(raw);
  assert.equal(result.source_url, undefined);
  assert.equal(result.rlm.result, raw.rlm.result);
});

void test('rejects fabricated replays and malformed optional fields before rendering', () => {
  for (const v of [
    { ...fixture(), kind: 'illustration' },
    { ...fixture(), original_query: { invalid: true } },
    { ...fixture(), rlm: { ...fixture().rlm, error: { message: 'bad' } } },
    { ...fixture(), direct: { ...fixture().direct, cached_tokens: {} } },
  ])
    assert.throws(() => validateComparison(v));
});
