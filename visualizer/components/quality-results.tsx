import { Check, Minus, X } from 'lucide-react';
import type { Comparison, Quality, ReferenceValue } from '@/lib/comparison';
const display = (v: ReferenceValue | null) =>
  v === null ? 'No answer' : JSON.stringify(v);
function ResultCell({ quality, label }: { quality?: Quality; label: string }) {
  const f = quality?.fields.find((f) => f.label === label);
  if (!f || quality?.status !== 'graded')
    return (
      <td className="ungraded">
        <Minus size={14} />
        Not graded
      </td>
    );
  return (
    <td className={f.passed ? 'field-pass' : 'field-fail'}>
      {f.passed ? <Check size={14} /> : <X size={14} />}
      <code>{display(f.observed)}</code>
    </td>
  );
}
export function QualityResults({
  comparison,
  live,
}: {
  comparison: Comparison;
  live: boolean;
}) {
  const fields = comparison.rlm.quality?.fields.length
    ? comparison.rlm.quality.fields
    : comparison.direct.quality?.fields || [];
  const score = (q?: Quality) =>
    q?.status === 'graded'
      ? `${q.matched} / ${q.total} correct`
      : q?.status === 'unavailable'
        ? 'Unavailable'
        : 'Needs review';
  return (
    <section className="quality-results">
      <div className="quality-heading">
        <div>
          <span className="eyebrow">CORRECTNESS BEFORE EFFICIENCY</span>
          <h2>
            {live
              ? 'Answers are still coming in.'
              : 'Did they answer the question?'}
          </h2>
        </div>
        <span className="quality-badge">
          {fields.length ? 'Reference check' : 'Manual review'}
        </span>
      </div>
      <p>
        {comparison.evaluation_label ||
          (fields.length
            ? 'Each requested field is checked against an independently computed answer. This measures these facts, not overall intelligence or prose quality.'
            : 'This experiment has no field-level reference. Review the full answers below. Token use and speed do not measure correctness.')}
      </p>
      {!!fields.length && (
        <div className="quality-table-scroll">
          <table className="quality-table">
            <thead>
              <tr>
                <th>Requested field</th>
                <th>Reference answer</th>
                <th>
                  With RLM <span>{score(comparison.rlm.quality)}</span>
                </th>
                <th>
                  Direct LLM <span>{score(comparison.direct.quality)}</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {fields.map((f) => (
                <tr key={f.label}>
                  <th>{f.label.replaceAll('_', ' ')}</th>
                  <td>
                    <code>{display(f.expected)}</code>
                    {!!f.accepted_aliases?.length && (
                      <small className="accepted-aliases">
                        Also accepts: {f.accepted_aliases.join(', ')}
                      </small>
                    )}
                  </td>
                  <ResultCell
                    quality={comparison.rlm.quality}
                    label={f.label}
                  />
                  <ResultCell
                    quality={comparison.direct.quality}
                    label={f.label}
                  />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="answer-pair">
        {(['rlm', 'direct'] as const).map((side) => (
          <details key={side} open={!fields.length && !live}>
            <summary>
              {side === 'rlm' ? 'RLM' : 'Direct LLM'} · full answer
            </summary>
            {comparison[side].error && (
              <p className="answer-error">{comparison[side].error}</p>
            )}
            <pre>
              {comparison[side].result ||
                (live ? 'Waiting for an answer…' : 'No answer returned.')}
            </pre>
          </details>
        ))}
      </div>
    </section>
  );
}
