'use client';
import { useEffect, useRef, useState } from 'react';
import {
  ArrowRight,
  Check,
  Download,
  FileText,
  LoaderCircle,
  Plug,
  Square,
  Terminal,
  Upload,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import catalog from '@/lib/presets.json';
import { type Comparison } from '@/lib/comparison';
import {
  localAddress,
  parseReferences,
  runnerRequest,
  validateRun,
  type Connection,
  type DocumentInfo,
  type Run,
} from '@/lib/runner';

export function ExperimentWorkbench({
  onComparison,
  onBusy,
}: {
  onComparison: (comparison: Comparison, live: boolean) => void;
  onBusy: (busy: boolean) => void;
}) {
  const [sourceMode, setSourceMode] = useState('preset');
  const [presetId, setPresetId] = useState('python-csv-docs');
  const preset = catalog.find((p) => p.id === presetId)!;
  const [question, setQuestion] = useState(preset.suggested_question);
  const [file, setFile] = useState<File | null>(null);
  const [reference, setReference] = useState('');
  const [connection, setConnection] = useState<Connection | null>(null);
  const [address, setAddress] = useState('http://127.0.0.1:8766');
  const [token, setToken] = useState('');
  const [pairing, setPairing] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [document, setDocument] = useState<DocumentInfo | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [lastRun, setLastRun] = useState<Run | null>(null);
  const alive = useRef(true);
  const callbacks = useRef({ onComparison, onBusy });
  useEffect(() => {
    callbacks.current = { onComparison, onBusy };
  }, [onComparison, onBusy]);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const presetGraded =
    sourceMode === 'preset' && question.trim() === preset.suggested_question;

  useEffect(() => {
    if (!runId || !connection || !busy) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const r = validateRun(
          await runnerRequest<Run>(connection!, `/runs/${runId}`),
        );
        if (stopped) return;
        setLastRun(r);
        const running = r.state === 'running' || r.state === 'cancelling';
        setStatus(
          r.state === 'cancelled'
            ? 'Run cancelled'
            : r.state === 'failed'
              ? 'Run failed'
              : r.state === 'completed'
                ? 'Comparison complete'
                : r.phase === 'direct'
                  ? 'Direct LLM is reading the full document…'
                  : r.phase === 'rlm'
                    ? 'RLM is exploring the document in Python…'
                    : r.state === 'cancelling'
                      ? 'Stopping model calls…'
                      : 'Checking Codex login…',
        );
        if (r.comparison) callbacks.current.onComparison(r.comparison, running);
        if (!running) {
          setBusy(false);
          callbacks.current.onBusy(false);
          if (r.error) setError(r.error);
          return;
        }
      } catch (e) {
        if (stopped) return;
        setError(
          `${e instanceof Error ? e.message : 'Connection interrupted.'} The local run may still be active; reconnect to resume its status.`,
        );
        setBusy(false);
        callbacks.current.onBusy(false);
        return;
      }
      timer = setTimeout(poll, 1000);
    }
    void poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [runId, connection, busy]);

  async function connect() {
    setError('');
    setConnecting(true);
    try {
      if (!token.trim())
        throw new Error(
          'Paste the pairing code printed by the local companion.',
        );
      const c = { address: localAddress(address), token: token.trim() };
      const health = await runnerRequest<{
        service: string;
        authenticated: boolean;
        model: string;
      }>(c, '/health');
      if (health.service !== 'rlm-compare')
        throw new Error('This address is not an RLM comparison companion.');
      if (!health.authenticated)
        throw new Error(
          'Companion pairing was not accepted. Check the code in your terminal.',
        );
      setConnection(c);
      setPairing(false);
      setToken('');
      setStatus(
        'Local runner connected · Codex login is checked before each run',
      );
      if (
        runId &&
        lastRun?.state !== 'completed' &&
        lastRun?.state !== 'cancelled' &&
        lastRun?.state !== 'failed'
      ) {
        setBusy(true);
        callbacks.current.onBusy(true);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Connection failed.');
    } finally {
      setConnecting(false);
    }
  }
  async function start() {
    setError('');
    if (!connection) {
      setPairing(true);
      return;
    }
    try {
      if (!question.trim())
        throw new Error(
          'Write a question so both approaches have the same task.',
        );
      if (sourceMode === 'upload' && !file)
        throw new Error('Choose a document first.');
      const expected = parseReferences(reference);
      setBusy(true);
      callbacks.current.onBusy(true);
      setRunId(null);
      setLastRun(null);
      setDocument(null);
      setStatus(
        sourceMode === 'preset'
          ? 'Downloading the original document and checking its SHA-256…'
          : 'Extracting text from your document…',
      );
      let doc: DocumentInfo;
      if (sourceMode === 'preset')
        doc = await runnerRequest(
          connection,
          `/presets/${preset.id}/download`,
          {},
        );
      else {
        if (file!.size > 5_000_000)
          throw new Error('Choose a document smaller than 5 MB.');
        const bytes = new Uint8Array(await file!.arrayBuffer());
        let binary = '';
        for (let i = 0; i < bytes.length; i += 8192)
          binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
        doc = await runnerRequest(connection, '/documents', {
          filename: file!.name,
          content_base64: btoa(binary),
        });
      }
      if (!alive.current) return;
      if (
        sourceMode === 'preset' &&
        doc.suggested_question !== preset.suggested_question
      )
        throw new Error(
          'Your local companion has an older example catalog. Download the current companion from Connect Codex, restart it and connect again.',
        );
      setDocument(doc);
      setStatus('Starting the paired experiment…');
      const run = validateRun(
        await runnerRequest<Run>(connection, '/runs', {
          document_id: doc.document_id,
          question: question.trim(),
          ...(expected ? { expected_fields: expected } : {}),
        }),
      );
      setRunId(run.run_id);
      setLastRun(run);
      if (run.comparison) callbacks.current.onComparison(run.comparison, true);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : 'Could not start the experiment.',
      );
      setBusy(false);
      callbacks.current.onBusy(false);
      setStatus('');
    }
  }
  async function cancel() {
    if (!connection || !runId) return;
    try {
      await runnerRequest(connection, `/runs/${runId}/cancel`, {});
      setStatus('Stopping model calls…');
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : 'Could not cancel. Stop the companion in your terminal.',
      );
    }
  }
  return (
    <section className="experiment-workbench" id="new-experiment">
      <div className="workbench-heading">
        <div>
          <span className="eyebrow">YOUR DOCUMENT. YOUR QUESTION.</span>
          <h2>Put both approaches to the test.</h2>
        </div>
        <Button
          variant="outline"
          onClick={() => setPairing(true)}
          disabled={busy}
          className={connection ? 'connected-button' : ''}
        >
          {connection ? <Check size={15} /> : <Plug size={15} />}
          {connection ? 'Local runner connected' : 'Connect Codex'}
        </Button>
      </div>
      <div className="experiment-grid">
        <div className="source-picker">
          <div className="field-heading">
            <span>01</span> Choose a document
          </div>
          <Tabs
            value={sourceMode}
            onValueChange={(v) => {
              if (busy) return;
              setSourceMode(v);
              setQuestion(v === 'preset' ? preset.suggested_question : '');
              setReference('');
              setDocument(null);
            }}
          >
            <TabsList>
              <TabsTrigger value="preset" disabled={busy}>
                From the web
              </TabsTrigger>
              <TabsTrigger value="upload" disabled={busy}>
                Your document
              </TabsTrigger>
            </TabsList>
            <TabsContent value="preset">
              <fieldset className="preset-list" aria-label="Example documents">
                {catalog.map((p) => (
                  <button
                    key={p.id}
                    className={`preset-card ${presetId === p.id ? 'chosen' : ''}`}
                    disabled={busy}
                    aria-pressed={presetId === p.id}
                    onClick={() => {
                      setPresetId(p.id);
                      setQuestion(p.suggested_question);
                      setReference('');
                      setDocument(null);
                    }}
                  >
                    <span className="preset-icon">
                      <FileText size={19} />
                    </span>
                    <span>
                      <strong>{p.title}</strong>
                      <small>{p.description}</small>
                      <em>
                        {Math.round(p.context_chars / 1000)}k characters ·{' '}
                        {p.filename.split('.').pop()?.toUpperCase()}
                      </em>
                    </span>
                    <span className="choice-circle">
                      {presetId === p.id && <Check size={12} />}
                    </span>
                  </button>
                ))}
              </fieldset>
              <p className="field-help">
                <Download size={13} />
                Downloaded fresh when you run. Only the catalog lives in this
                app.
              </p>
              <a
                className="source-link"
                href={preset.source_url}
                target="_blank"
                rel="noreferrer"
              >
                View original document <ArrowRight size={12} />
              </a>
            </TabsContent>
            <TabsContent value="upload">
              <label className={`document-drop ${busy ? 'disabled' : ''}`}>
                <Upload size={27} />
                <strong>{file?.name || 'Choose your document'}</strong>
                <span>
                  {file
                    ? `${(file.size / 1000).toFixed(1)} KB · click to replace`
                    : 'TXT, Markdown, CSV, JSON, PDF or DOCX'}
                </span>
                <small>Up to 5 MB · 300k extracted characters</small>
                <input
                  type="file"
                  disabled={busy}
                  accept=".txt,.md,.csv,.json,.pdf,.docx"
                  onChange={(e) => {
                    setFile(e.target.files?.[0] || null);
                    setDocument(null);
                  }}
                />
              </label>
              <p className="field-help">
                Your local companion extracts the text. Scanned PDFs need
                selectable text; images and page layout are not analyzed.
              </p>
            </TabsContent>
          </Tabs>
        </div>
        <div className="question-editor">
          <label className="field-heading" htmlFor="experiment-question">
            <span>02</span> Ask a question
          </label>
          <Textarea
            id="experiment-question"
            value={question}
            maxLength={8000}
            disabled={busy}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What would you like to find out about this document?"
          />
          <div
            className={`grading-explainer ${presetGraded && !reference.trim() ? 'has-reference' : ''}`}
          >
            <Check size={16} />
            <div>
              <strong>
                {presetGraded && !reference.trim()
                  ? `${Object.keys(preset.expected_fields).length} reference fields · automatically checked`
                  : reference.trim()
                    ? 'Your reference answers · checked after the run'
                    : 'No reference answer · manual review'}
              </strong>
              <p>
                {presetGraded && !reference.trim()
                  ? 'Both approaches answer your question. We check the requested facts against a reference they never see.'
                  : 'Efficiency measures tokens and time. Correctness needs an answer key; without one, compare the full answers yourself.'}
              </p>
            </div>
          </div>
          <details className="reference-editor">
            <summary>
              {presetGraded
                ? 'View the reference or supply your own'
                : 'Add reference answers (optional)'}
            </summary>
            {presetGraded && (
              <pre>{JSON.stringify(preset.expected_fields, null, 2)}</pre>
            )}
            <label htmlFor="reference-json">Optional answer key as JSON</label>
            <Textarea
              id="reference-json"
              value={reference}
              disabled={busy}
              onChange={(e) => setReference(e.target.value)}
              placeholder={'{"count": 12, "title": "Expected title"}'}
            />
            <p>
              Use exact facts, numbers or booleans. Editing a preset question
              disables its original answer key. Field names are added to both
              prompts; expected values stay outside them.
            </p>
            <p>
              You can ask any question in ordinary language. For automatic
              checking, the runner adds the same answer-format instructions to
              both models; the full prompt is available with the result.
            </p>
          </details>
        </div>
      </div>
      <div className="experiment-actions">
        <p>
          <Terminal size={16} />
          <span>
            <strong>GPT-5.6 Luna · your Codex subscription</strong>
            <small>
              One source and question. Direct first, then RLM. Real runs use
              your allowance.
            </small>
          </span>
        </p>
        {busy ? (
          <Button variant="outline" onClick={cancel} disabled={!runId}>
            <Square size={13} />
            Stop run
          </Button>
        ) : (
          <Button
            className="start-experiment"
            onClick={start}
            disabled={!question.trim() || (sourceMode === 'upload' && !file)}
          >
            {connection ? <PlayIcon /> : <Plug size={16} />}
            {connection
              ? sourceMode === 'preset'
                ? 'Download & run'
                : 'Run comparison'
              : 'Connect to run'}
            <ArrowRight size={15} />
          </Button>
        )}
      </div>
      {status && (
        <output className="runner-status">
          {busy ? (
            <LoaderCircle className="spin" size={15} />
          ) : (
            <Check size={15} />
          )}
          {status}
        </output>
      )}
      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}
      {document && (
        <details className="document-preview">
          <summary>
            Source received · {document.filename} ·{' '}
            {document.context_chars.toLocaleString()} characters
          </summary>
          <p className="mono">SHA-256: {document.sha256}</p>
          {document.warnings.map((w, i) => (
            <p key={i}>{w}</p>
          ))}
          <pre>{document.preview}</pre>
        </details>
      )}
      <Dialog open={pairing} onOpenChange={setPairing}>
        <DialogContent className="pairing-dialog">
          <DialogTitle>Connect your local Codex</DialogTitle>
          <DialogDescription>
            The page controls a small companion on your computer. Your ChatGPT
            login stays in Codex; model requests use your subscription.
          </DialogDescription>
          <div className="pairing-steps">
            <p>
              Download the companion, unzip it, and open a terminal in its
              folder.
            </p>
            <a
              className="companion-download"
              href="/rlm-local-companion.zip"
              download
            >
              <Download size={15} />
              Download local companion <span>Python source · ZIP</span>
            </a>
            <p>Create a virtual environment, then run:</p>
            <pre>
              <code>
                {'python3 -m venv .venv'}
                {'\n'}
                {'source .venv/bin/activate'}
                {'\n'}
                {"python -m pip install -e '.[visualizer]'"}
                {'\n'}codex login{'\n'}
                python examples/serve_codex_comparison.py
              </code>
            </pre>
            <p>
              Keep that terminal open and paste its pairing code below. The code
              is held only in this tab’s memory.
            </p>
          </div>
          <label htmlFor="runner-address">Companion address</label>
          <Input
            id="runner-address"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            autoComplete="off"
          />
          <label htmlFor="runner-code">Pairing code</label>
          <Input
            id="runner-code"
            type="password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            autoComplete="off"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !connecting) void connect();
            }}
          />
          <p className="field-help">
            Your browser may ask for local network access.{' '}
            <a
              href="https://developer.chrome.com/blog/local-network-access"
              target="_blank"
              rel="noreferrer"
            >
              How it works
            </a>
            . Documents go through your local companion to Codex, not to a
            shared runner.
          </p>
          {error && (
            <div className="error-banner" role="alert">
              {error}
            </div>
          )}
          <Button onClick={connect} disabled={connecting}>
            {connecting ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <Plug size={16} />
            )}
            {connecting ? 'Connecting…' : 'Connect'}
          </Button>
        </DialogContent>
      </Dialog>
    </section>
  );
}
function PlayIcon() {
  return <ArrowRight size={16} />;
}
