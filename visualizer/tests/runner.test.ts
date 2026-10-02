import test from 'node:test';
import assert from 'node:assert/strict';
import { localAddress, parseReferences, validateRun } from '../lib/runner.ts';
import { validateComparison } from '../lib/comparison.ts';
import { readFileSync } from 'node:fs';

void test('pairing credentials can only be sent to explicit loopback', () => {
  assert.equal(localAddress('http://127.0.0.1:8766/'), 'http://127.0.0.1:8766');
  for (const address of [
    'https://example.com',
    'http://127.0.0.1.evil.test:8766',
    'http://0.0.0.0:8766',
    'http://127.0.0.1:8766/path',
    'http://user:secret@127.0.0.1:8766',
    'http://127.0.0.1:8766/?token=secret',
  ])
    assert.throws(() => localAddress(address));
});
void test('reference input preserves exact JSON values and rejects unsupported answer keys', () => {
  assert.equal(parseReferences(''), undefined);
  assert.deepEqual(
    parseReferences('{"count": 12, "enabled": false, "terminator": "\\r\\n"}'),
    { count: 12, enabled: false, terminator: '\r\n' },
  );
  for (const value of [
    'null',
    '[]',
    '{}',
    '{"a":[]}',
    '{"bad key":1}',
    '{"number":1e999}',
    JSON.stringify({ text: 'a'.repeat(1001) }),
  ])
    assert.throws(() => parseReferences(value));
});
void test('real recordings retain field-level evidence, while incomplete snapshots remain ungraded', () => {
  const source = JSON.parse(
    readFileSync(
      new URL('../public/recordings/luna-100k.json', import.meta.url),
      'utf8',
    ),
  );
  const v = validateComparison(source);
  assert.equal(v.rlm.quality?.matched, 4);
  assert.equal(v.direct.quality?.fields[1].expected, 27404392);
  source.rlm.quality = {
    status: 'unavailable',
    matched: null,
    total: null,
    fields: [],
  };
  assert.equal(
    validateRun({
      run_id: 'abc123',
      state: 'running',
      phase: 'rlm',
      comparison: source,
    }).comparison?.rlm.quality?.matched,
    null,
  );
  source.rlm.quality = { status: 'graded', matched: 4, total: 4, fields: [] };
  assert.throws(() => validateComparison(source));
  assert.throws(() =>
    validateRun({
      run_id: '../health',
      state: 'running',
      phase: 'rlm',
      comparison: null,
    }),
  );
});
