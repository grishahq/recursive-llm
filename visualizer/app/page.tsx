'use client';
import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import {
  ArrowDown,
  ArrowUp,
  ArrowUpRight,
  Braces,
  Check,
  ChevronRight,
  Download,
  FileText,
  GitBranch,
  Code2,
  Info,
  Pause,
  Play,
  RotateCcw,
  Star,
  Terminal,
  Upload,
  X,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import {
  NativeSelect,
  NativeSelectOption,
} from '@/components/ui/native-select';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Comparison, Lane, validateComparison } from '@/lib/comparison';
import { ExperimentWorkbench } from '@/components/experiment-workbench';
import { QualityResults } from '@/components/quality-results';

declare global {
  interface Window {
    __rlmSetVideoTime?: (s: number) => Promise<void>;
    __rlmReady?: boolean;
  }
}
const fmt = (n: number | null) =>
  n === null
    ? '—'
    : new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(n);
const clock = (s: number) =>
  `${Math.floor(s / 60)
    .toString()
    .padStart(2, '0')}:${Math.floor(s % 60)
    .toString()
    .padStart(2, '0')}`;
const EMPTY: Lane = {
  tokens: null,
  cost: null,
  seconds: 0,
  calls: 0,
  result: '',
  passed: null,
  steps: [],
};
const INITIAL: Comparison = {
  schema_version: 1,
  kind: 'recorded',
  title: 'Transaction ledger',
  model: 'gpt-5.6-luna',
  query:
    'Find the exact count, total amount, and largest settled transaction in EMEA.',
  context_chars: 100098,
  context_preview: '',
  sample_note: 'Codex subscription · recorded comparison',
  rlm: EMPTY,
  direct: EMPTY,
};

function Flow({
  side,
  lane,
  cursor,
  chars,
  model,
}: {
  side: 'rlm' | 'direct';
  lane: Lane;
  cursor: number;
  chars: number;
  model: string;
}) {
  const started = cursor > 0,
    finished =
      Boolean(lane.result || lane.error) &&
      cursor >= lane.seconds &&
      lane.seconds > 0;
  return (
    <div
      className={`flow flow-${side}`}
      aria-label={
        side === 'rlm'
          ? 'Question to model with source in Python REPL'
          : 'Question and full source to model'
      }
    >
      <div className="flow-top">
        <span className="flow-pill">
          <FileText size={14} />
          Question
        </span>
        {side === 'direct' && (
          <span className="flow-pill">+ entire source</span>
        )}
      </div>
      <div className={`connector ${started ? 'active' : ''}`}>
        <ArrowDown size={14} />
      </div>
      <div className={`model-node ${started ? 'active' : ''}`}>
        <span className="node-symbol">
          <Braces size={24} />
        </span>
        <div>
          <strong>
            {side === 'rlm' ? 'Root language model' : 'Language model'}
          </strong>
          <span>{model.replace('gpt-5.6-luna', 'GPT-5.6 Luna')} · Codex</span>
        </div>
        <i />
      </div>
      {side === 'rlm' ? (
        <>
          <div className="loop">
            <span>inspect</span>
            <ArrowDown size={14} />
            <span>observe</span>
          </div>
          <div className="repl-node">
            <Terminal size={18} />
            <div>
              <strong>Python REPL</strong>
              <span>{fmt(chars)} characters · local computation</span>
            </div>
            <span className="local-tag">LOCAL</span>
          </div>
        </>
      ) : (
        <div className="context-block">
          <div className="context-lines">
            {Array.from({ length: 8 }, (_, i) => (
              <span key={i} style={{ width: `${58 + ((i * 17) % 43)}%` }} />
            ))}
          </div>
          <div className="context-caption">
            <FileText size={13} />
            {fmt(chars)} characters in the prompt
          </div>
        </div>
      )}
      <div className={`connector last ${finished ? 'active' : ''}`}>
        <ArrowDown size={14} />
      </div>
      <div className={`answer-node ${finished ? 'ready' : ''}`}>
        <Check size={14} />
        {finished ? 'Answer received' : 'Waiting for the answer'}
      </div>
    </div>
  );
}
function Panel({
  side,
  lane,
  demo,
  cursor,
  selected,
  onSelect,
}: {
  side: 'rlm' | 'direct';
  lane: Lane;
  demo: Comparison;
  cursor: number;
  selected: number | null;
  onSelect: (n: number | null) => void;
}) {
  const firstCall = lane.steps.find((s) => s.kind === 'model_call_start');
  const visibleSteps = lane.steps.filter(
    (s) =>
      !['run_start', 'rlm_start', 'run_end', 'rlm_end'].includes(
        s.kind || '',
      ) &&
      (s.kind !== 'model_call_start' || s === firstCall),
  );
  const steps = visibleSteps.length ? visibleSteps : lane.steps;
  const active = Math.max(
    0,
    steps.reduce((acc, s, i) => (s.at <= cursor ? i : acc), 0),
  );
  const index = Math.min(selected ?? active, Math.max(0, steps.length - 1));
  const step = steps[index];
  const nodes = Array.from(
    new Map(
      lane.steps.filter((s) => s.node_id).map((s) => [s.node_id!, s]),
    ).values(),
  );
  return (
    <section className={`run-panel ${side}`}>
      <header className="panel-header">
        <div className="panel-name">
          <span className="mode-icon">
            {side === 'rlm' ? <GitBranch size={20} /> : <Braces size={20} />}
          </span>
          <div>
            <h2>{side === 'rlm' ? 'With RLM' : 'Direct LLM'}</h2>
            <p>
              {side === 'rlm'
                ? 'Explore. Compute. Answer.'
                : 'One prompt. One model call.'}
            </p>
          </div>
        </div>
        <span className="mode-tag">
          {side === 'rlm' ? 'RECURSIVE' : 'BASELINE'}
        </span>
      </header>
      <Flow
        side={side}
        lane={lane}
        cursor={cursor}
        chars={demo.context_chars}
        model={demo.model}
      />
      <div className="step-view">
        <div className="step-view-top">
          <span>
            <Terminal size={14} />
            EXECUTION TRACE
          </span>
          <span>
            {steps.length ? index + 1 : 0} / {steps.length}
          </span>
        </div>
        <div className="step-dots">
          {steps.map((s, i) => (
            <button
              key={i}
              title={s.title}
              aria-label={`Step ${i + 1}: ${s.title}`}
              className={`${i === index ? 'selected' : ''} ${s.at <= cursor ? 'reached' : ''}`}
              onClick={() => onSelect(i === selected ? null : i)}
            />
          ))}
        </div>
        <div className="step-heading">
          <span className="step-number">
            {String(index + 1).padStart(2, '0')}
          </span>
          <h3>{step?.title || 'Preparing the comparison'}</h3>
          <span className="step-time">{step ? clock(step.at) : ''}</span>
          {selected !== null && (
            <Button
              size="icon-sm"
              variant="ghost"
              aria-label="Follow playback"
              onClick={() => onSelect(null)}
            >
              <X size={13} />
            </Button>
          )}
        </div>
        <p className="step-detail">
          {step?.detail || 'The recorded run will appear here.'}
        </p>
        {lane.error && (
          <output className="error-message">Run failed: {lane.error}</output>
        )}
        <div className="code-window">
          {step?.code ? (
            <pre>
              <code>{step.code}</code>
            </pre>
          ) : step?.output ? (
            <pre className="output-code">
              <code>{step.output}</code>
            </pre>
          ) : (
            <div className="waiting-output">
              <span />
              <span />
              <span />
              <p>
                {lane.steps.length
                  ? 'Awaiting the next recorded event'
                  : 'Loading recording'}
              </p>
            </div>
          )}
        </div>
        {step?.code && step.output && (
          <details className="output-details">
            <summary>
              View output
              <ChevronRight size={14} />
            </summary>
            <pre>{step.output}</pre>
          </details>
        )}
        {side === 'rlm' && nodes.length > 0 && (
          <details className="tree-details">
            <summary>
              <GitBranch size={13} />
              Call tree · {nodes.length} nodes
              <ChevronRight size={13} />
            </summary>
            <div className="tree-list">
              {nodes.map((n) => (
                <div
                  key={n.node_id}
                  style={{ marginLeft: n.parent_id ? 18 : 0 }}
                >
                  <span>{n.node_id}</span>
                  <small>{n.parent_id ? `from ${n.parent_id}` : 'root'}</small>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
      <div className="panel-metrics">
        <div>
          <span>Total model tokens</span>
          <strong>{fmt(lane.tokens)}</strong>
        </div>
        <div>
          <span>Model calls</span>
          <strong>{fmt(lane.calls)}</strong>
        </div>
        <div>
          <span>Exact fields</span>
          <strong className={side === 'rlm' ? 'positive' : ''}>
            {lane.passed?.replace(/ fields$/, '') || 'Not graded'}
          </strong>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const [live, setLive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [experimentMode, setExperimentMode] = useState('new');
  const [demo, setDemo] = useState<Comparison>(INITIAL);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [selected, setSelected] = useState<[number | null, number | null]>([
    null,
    null,
  ]);
  const [modal, setModal] = useState<'method' | 'run' | 'video' | null>(null);
  const [error, setError] = useState('');
  const [loaded, setLoaded] = useState(false);
  const [film, setFilm] = useState(false);
  const [filmTime, setFilmTime] = useState(0);
  const [recordings, setRecordings] = useState<
    { name: string; file: string }[]
  >([]);
  const [choice, setChoice] = useState('');
  const input = useRef<HTMLInputElement>(null);
  const total = Math.max(
    1,
    demo.rlm.seconds,
    demo.direct.seconds,
    ...demo.rlm.steps.map((s) => s.at),
    ...demo.direct.steps.map((s) => s.at),
  );
  const savings =
    !demo.rlm.error &&
    !demo.direct.error &&
    demo.rlm.tokens !== null &&
    demo.direct.tokens
      ? Math.round((1 - demo.rlm.tokens / demo.direct.tokens) * 100)
      : null;
  const setComparison = (v: Comparison) => {
    setDemo(v);
    setCursor(v.rlm.steps.find((s) => s.kind === 'repl_step')?.at ?? 0);
    setSelected([null, null]);
    setPlaying(false);
    setLoaded(true);
    setError('');
  };
  const loadRecording = async (file: string) => {
    try {
      const res = await fetch(file);
      if (!res.ok) throw new Error('This recording could not be loaded.');
      setComparison(validateComparison(await res.json()));
      setChoice(file);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load recording.');
    }
  };
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    requestAnimationFrame(() => setFilm(params.has('film')));
    fetch('/recordings/index.json')
      .then((r) => {
        if (!r.ok) throw new Error('Recording manifest unavailable');
        return r.json();
      })
      .then(async (v) => {
        if (
          !Array.isArray(v) ||
          v.some(
            (r) =>
              typeof r?.name !== 'string' ||
              typeof r?.file !== 'string' ||
              !r.file.startsWith('/recordings/'),
          )
        )
          throw new Error('Invalid recording manifest');
        const entries = v as { name: string; file: string }[];
        setRecordings(entries);
        if (entries.length)
          await loadRecording(
            params.has('film') ? '/recordings/luna-100k.json' : entries[0].file,
          );
      })
      .catch(() =>
        setError(
          'The recording could not be loaded. You can still open an exported comparison.',
        ),
      );
  }, []);
  useEffect(() => {
    window.__rlmReady = loaded;
    window.__rlmSetVideoTime = async (s: number) => {
      setPlaying(false);
      setFilmTime(s);
      setCursor(Math.min(total, Math.max(0, ((s - 3) / 25) * total)));
      setSelected([null, null]);
      await new Promise<void>((r) =>
        requestAnimationFrame(() => requestAnimationFrame(() => r())),
      );
    };
    return () => {
      delete window.__rlmSetVideoTime;
      delete window.__rlmReady;
    };
  }, [total, loaded]);
  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const id = setInterval(() => {
      const now = performance.now();
      const delta = ((now - last) / 1000) * speed;
      last = now;
      setCursor((c) => Math.min(total, c + delta));
    }, 40);
    return () => clearInterval(id);
  }, [playing, speed, total]);
  useEffect(() => {
    if (cursor >= total) {
      const id = requestAnimationFrame(() => setPlaying(false));
      return () => cancelAnimationFrame(id);
    }
  }, [cursor, total]);
  const openFile = async (file?: File) => {
    if (!file) return;
    try {
      if (file.size > 10_000_000)
        throw new Error('Choose a comparison smaller than 10 MB.');
      setComparison(validateComparison(JSON.parse(await file.text())));
      setChoice('custom');
      setModal(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not read file.');
    } finally {
      if (input.current) input.current.value = '';
    }
  };
  const download = () => {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(demo, null, 2)], { type: 'application/json' }),
    );
    const a = document.createElement('a');
    a.href = url;
    a.download = 'rlm-comparison.json';
    a.click();
    URL.revokeObjectURL(url);
  };
  return (
    <main className={`app ${film ? 'film-mode' : ''}`}>
      <nav className="topbar">
        <Link href="/" className="brand">
          <span className="brand-mark">
            <GitBranch size={23} />
          </span>
          <strong>
            recursive<span>-</span>llm
          </strong>
          <span className="brand-divider" />
          <span className="brand-sub">The comparison lab</span>
        </Link>
        <div className="nav-actions">
          <span className="milestone">
            <Star size={14} fill="currentColor" />
            600 stars. Thank you.
          </span>
          <a
            className="github-link"
            href="https://github.com/grishahq/recursive-llm"
            target="_blank"
            rel="noreferrer"
          >
            <Code2 size={17} />
            <span>View project</span>
            <ArrowUpRight size={14} />
          </a>
        </div>
      </nav>
      <div className="workspace">
        <header className="intro">
          <div>
            <div className="eyebrow">
              <span className="live-dot" /> A CLOSER LOOK AT LONG CONTEXT
            </div>
            <h1>
              One question. <span>Two approaches.</span>
            </h1>
            <p>
              Watch a recursive language model and a direct call work through
              the same source.
            </p>
          </div>
          <Button
            variant="outline"
            className="run-own"
            onClick={() => {
              setExperimentMode('new');
              document
                .getElementById('new-experiment')
                ?.scrollIntoView({ behavior: 'smooth' });
            }}
          >
            <Terminal size={16} />
            Run with Codex
            <ArrowUpRight size={15} />
          </Button>
        </header>
        <div className="experiment-switch">
          <Tabs
            value={experimentMode}
            onValueChange={(v) => !busy && setExperimentMode(v)}
          >
            <TabsList>
              <TabsTrigger value="new">New experiment</TabsTrigger>
              <TabsTrigger value="recorded" disabled={busy}>
                Recorded runs
              </TabsTrigger>
            </TabsList>
          </Tabs>
          <Button
            variant="ghost"
            disabled={busy}
            onClick={() => setModal('run')}
          >
            <Upload size={14} />
            Open a recording
          </Button>
        </div>
        <div
          className={
            experimentMode === 'new'
              ? 'workbench-container'
              : 'workbench-container hidden'
          }
        >
          <ExperimentWorkbench
            onBusy={(b) => {
              setBusy(b);
              if (!b) setLive(false);
              if (b) {
                setPlaying(false);
                setSelected([null, null]);
              }
            }}
            onComparison={(v, running) => {
              setDemo(v);
              setCursor(
                Math.max(
                  v.rlm.seconds,
                  v.direct.seconds,
                  ...v.rlm.steps.map((s) => s.at),
                  ...v.direct.steps.map((s) => s.at),
                ),
              );
              setSelected([null, null]);
              setPlaying(false);
              setLoaded(true);
              setChoice('custom');
              setLive(running);
            }}
          />
        </div>
        <section className="query-bar">
          <div className="document-icon">
            <FileText size={23} />
          </div>
          <div className="query-content">
            <div className="query-meta">
              <strong>{demo.title}</strong>
              <span>{fmt(demo.context_chars)} characters</span>
              <span className="dataset-label">SAME SOURCE</span>
            </div>
            <span className="question-caption">
              QUESTION ASKED TO BOTH APPROACHES
            </span>
            <p>{demo.original_query ?? demo.query}</p>
            {demo.original_query && demo.original_query !== demo.query && (
              <details className="prompt-protocol">
                <summary>
                  View the full question and answer-format instructions
                </summary>
                <pre>{demo.query}</pre>
              </details>
            )}
          </div>
          <div className="model-picker">
            <label htmlFor="recording">RECORDED EXPERIMENT</label>
            <NativeSelect
              id="recording"
              disabled={busy}
              value={choice}
              onChange={(e) => loadRecording(e.target.value)}
            >
              {!recordings.length && (
                <NativeSelectOption value="">GPT-5.6 Luna</NativeSelectOption>
              )}
              {recordings.map((r) => (
                <NativeSelectOption key={r.file} value={r.file}>
                  {r.name}
                </NativeSelectOption>
              ))}
              {choice === 'custom' && (
                <NativeSelectOption value="custom">
                  Current comparison
                </NativeSelectOption>
              )}
            </NativeSelect>
          </div>
        </section>
        <div className="workspace-label">
          <span>
            <span className="status-dot" />
            {live ? 'Live local experiment' : 'Recorded experiment'}
            <span className="label-divider">/</span>
            {demo.model.replace('gpt-5.6-luna', 'GPT-5.6 Luna')} · Codex
            subscription
          </span>
          <Button variant="ghost" onClick={() => setModal('method')}>
            <Info size={14} />
            About this comparison
          </Button>
        </div>
        <div className="comparison">
          <Panel
            side="rlm"
            lane={demo.rlm}
            cursor={cursor}
            demo={demo}
            selected={selected[0]}
            onSelect={(v) => setSelected([v, selected[1]])}
          />
          <div className="versus">VS</div>
          <Panel
            side="direct"
            lane={demo.direct}
            cursor={cursor}
            demo={demo}
            selected={selected[1]}
            onSelect={(v) => setSelected([selected[0], v])}
          />
        </div>
        <div className={`playback ${live ? 'is-live' : ''}`}>
          <Button
            size="icon"
            className="play-button"
            aria-label={playing ? 'Pause replay' : 'Play replay'}
            disabled={!loaded || busy}
            onClick={() => {
              if (cursor >= total) setCursor(0);
              setSelected([null, null]);
              setPlaying(!playing);
            }}
          >
            {playing || film ? (
              <Pause fill="currentColor" />
            ) : (
              <Play fill="currentColor" />
            )}
          </Button>
          <Button
            size="icon"
            variant="ghost"
            disabled={busy}
            aria-label="Restart replay"
            onClick={() => {
              setCursor(0);
              setSelected([null, null]);
              setPlaying(false);
            }}
          >
            <RotateCcw size={16} />
          </Button>
          <span className="time-code">
            {clock(cursor)} <span>/ {clock(total)}</span>
          </span>
          <Slider
            className="timeline"
            disabled={busy}
            aria-label="Replay position"
            min={0}
            max={total}
            step={0.1}
            value={[cursor]}
            onValueChange={(v) => {
              setCursor(Array.isArray(v) ? v[0] : v);
              setSelected([null, null]);
            }}
          />
          <NativeSelect
            className={`speed-picker ${film ? 'hidden' : ''}`}
            disabled={busy}
            aria-label="Playback speed"
            value={String(speed)}
            onChange={(e) => setSpeed(Number(e.target.value))}
          >
            {[1, 2, 4, 8].map((v) => (
              <NativeSelectOption key={v} value={String(v)}>
                {v}×
              </NativeSelectOption>
            ))}
          </NativeSelect>
          <span className="playback-label">
            {film
              ? `${(total / 25).toFixed(2)}× · synchronized replay`
              : live
                ? 'Following live execution'
                : 'Synchronized replay'}
          </span>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Download comparison JSON"
            onClick={download}
          >
            <Download size={16} />
          </Button>
        </div>
        {error && (
          <div role="alert" className="error-banner">
            {error}
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Dismiss error"
              onClick={() => setError('')}
            >
              <X />
            </Button>
          </div>
        )}
        <section className="takeaway">
          <div className="takeaway-main">
            <span className="takeaway-icon">
              {savings !== null && savings < 0 ? (
                <ArrowUp size={22} />
              ) : (
                <ArrowDown size={22} />
              )}
            </span>
            <div>
              <strong>
                {savings !== null && savings > 0
                  ? `${savings}% fewer model tokens`
                  : savings !== null
                    ? `${Math.abs(savings)}% more model tokens`
                    : 'A real comparison'}
                <span> {loaded ? 'in this RLM run.' : 'through Codex.'}</span>
              </strong>
              <p>{demo.sample_note.split('. ')[0]}</p>
            </div>
          </div>
          <div className="latency-note">
            <span>Elapsed time, including Codex overhead</span>
            <strong>
              <i className="rlm-swatch" />
              {demo.rlm.seconds.toFixed(1)}s <span>vs</span>
              <i className="direct-swatch" />
              {demo.direct.seconds.toFixed(1)}s
            </strong>
          </div>
        </section>
        <QualityResults comparison={demo} live={live} />
        <footer>
          <p>
            One experiment, not a universal performance claim. Tokens include
            CLI overhead and cached input; subscription use is not a dollar
            cost.
          </p>
          <div>
            <Button variant="ghost" onClick={() => setModal('method')}>
              Methodology
              <ArrowUpRight size={14} />
            </Button>
            <Button variant="ghost" onClick={() => setModal('video')}>
              <Play size={13} />
              Watch the film
            </Button>
          </div>
        </footer>
      </div>
      <input
        ref={input}
        type="file"
        accept=".json,application/json"
        className="sr-only"
        aria-label="Import recorded comparison"
        onChange={(e) => openFile(e.target.files?.[0])}
      />
      <Dialog
        open={modal === 'method'}
        onOpenChange={(o) => !o && setModal(null)}
      >
        <DialogContent className="information-dialog">
          <DialogTitle>A fair view of this experiment</DialogTitle>
          <DialogDescription>
            Actual Codex calls, actual RLM execution, and the complete outcomes.
          </DialogDescription>
          <div className="method-body">
            <p className="sample-note">{demo.sample_note}</p>
            <p>
              <strong>Same model and source.</strong> Both approaches use{' '}
              {demo.model.replace('gpt-5.6-luna', 'GPT-5.6 Luna')} through Codex,
              signed in with ChatGPT. The runs are
              captured sequentially and aligned by elapsed time for replay. The
              direct mode receives the question and full source. RLM keeps the
              source in its Python REPL and chooses what to inspect.
            </p>
            <p>
              <strong>No Codex tools in either mode.</strong> The model
              generates text. Only the RLM library executes the Python it
              requests. The recorded call tree shows whether the run used
              child calls. Each model request starts a fresh Codex session
              with the conversation supplied by the runner.
            </p>
            <p>
              <strong>What the numbers mean.</strong> Total tokens are
              Codex-reported input plus output across all calls, including
              cached input and CLI instructions. Elapsed time includes process
              startup and network overhead. The subscription does not provide a
              per-run dollar charge.
            </p>
            <div className="method-callout">
              <Info size={20} />
              <p>
                Each recording is a single paired experiment. Correctness is
                checked against reference fields when an answer key is
                available; otherwise the answers need manual review. RLM can use
                fewer tokens yet take longer; small contexts can favor a direct
                call.
              </p>
            </div>
            <details className="source-details">
              <summary>Full question, answer format and source preview</summary>
              <p>{demo.query}</p>
              <pre>{demo.context_preview}</pre>
            </details>
            <p className="mono">
              Cached input: RLM {fmt(demo.rlm.cached_tokens ?? null)} · Direct{' '}
              {fmt(demo.direct.cached_tokens ?? null)}
            </p>
            <a
              href="https://github.com/grishahq/recursive-llm"
              target="_blank"
              rel="noreferrer"
            >
              Explore the open-source project
              <ArrowUpRight size={14} />
            </a>
          </div>
        </DialogContent>
      </Dialog>
      <Dialog open={modal === 'run'} onOpenChange={(o) => !o && setModal(null)}>
        <DialogContent className="run-dialog">
          <DialogTitle>Compare with your Codex subscription</DialogTitle>
          <DialogDescription>
            Run an experiment on your computer, then open its recording here.
          </DialogDescription>
          <Tabs defaultValue="run">
            <TabsList>
              <TabsTrigger value="run">Run locally</TabsTrigger>
              <TabsTrigger value="open">Open recording</TabsTrigger>
            </TabsList>
            <TabsContent value="run">
              <div className="local-instructions">
                <Terminal size={25} />
                <h3>Use your existing Codex login</h3>
                <p>
                  From the project directory, run the comparison script. It uses
                  GPT-5.6 Luna for both sides and saves the full result.
                </p>
                <pre>
                  <code>
                    pip install -e .{'\n'}codex login{'\n'}python
                    examples/capture_codex_comparison.py \\{'\n'} --chars 100000
                    \\{'\n'} --output comparison.json
                  </code>
                </pre>
                <p>
                  Real runs consume your Codex subscription allowance. Your
                  login stays in Codex. The source is sent through Codex to the
                  model; this site does not store your document.
                </p>
                <Button onClick={() => input.current?.click()}>
                  <Upload size={16} />
                  Open the comparison JSON
                </Button>
              </div>
            </TabsContent>
            <TabsContent value="open">
              <div className="import-panel">
                <Upload size={28} />
                <h3>Open a recorded comparison</h3>
                <p>
                  Choose the JSON saved by the runner. It is read locally in
                  your browser.
                </p>
                <Button onClick={() => input.current?.click()}>
                  Choose comparison file
                </Button>
              </div>
            </TabsContent>
          </Tabs>
          {error && (
            <p role="alert" className="error-message">
              {error}
            </p>
          )}
        </DialogContent>
      </Dialog>
      <Dialog
        open={modal === 'video'}
        onOpenChange={(o) => !o && setModal(null)}
      >
        <DialogContent className="video-dialog">
          <DialogTitle>One question. Two approaches.</DialogTitle>
          <DialogDescription>
            A recorded comparison using GPT-5.6 Luna through Codex.
          </DialogDescription>
          <video
            controls
            playsInline
            preload="metadata"
            poster="/rlm-comparison-poster.jpg"
            src="/rlm-comparison.mp4"
          >
            <track
              kind="captions"
              src="/rlm-comparison.vtt"
              srcLang="en"
              label="English"
            />
          </video>
          <a href="/rlm-comparison.mp4" download>
            Download the film
            <Download size={14} />
          </a>
        </DialogContent>
      </Dialog>
      {film && (
        <>
          <div className={`film-opening ${filmTime >= 3 ? 'away' : ''}`}>
            <div className="film-kicker">
              <GitBranch /> RECURSIVE LANGUAGE MODELS
            </div>
            <h2>
              One question.
              <br />
              <span>Two approaches.</span>
            </h2>
            <p>GPT-5.6 Luna. A real experiment through Codex.</p>
            <div className="film-legend">
              <span>
                <i className="rlm-swatch" />
                With RLM
              </span>
              <span>
                <i className="direct-swatch" />
                Direct LLM
              </span>
            </div>
          </div>
          <div className={`film-ending ${filmTime < 30 ? 'away' : ''}`}>
            <span className="milestone">
              <Star size={19} fill="currentColor" />
              600 stars. Thank you.
            </span>
            <h2>
              Explore the source.
              <br />
              <span>See the difference.</span>
            </h2>
            <p>github.com/grishahq/recursive-llm</p>
            <small>
              Actual recorded calls · synchronized replay accelerated for this
              film · one paired experiment
            </small>
          </div>
        </>
      )}
    </main>
  );
}
