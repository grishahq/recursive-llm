#!/usr/bin/env node
import { createRequire } from 'node:module';
import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { spawn } from 'node:child_process';
import { once } from 'node:events';

const require = createRequire(import.meta.url);
const playwrightPath = process.env.RLM_PLAYWRIGHT_PATH || 'playwright';
const { chromium } = require(playwrightPath);
const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, arg, i, all) => {
    if (arg.startsWith('--'))
      pairs.push([
        arg.slice(2),
        all[i + 1]?.startsWith('--') ? true : (all[i + 1] ?? true),
      ]);
    return pairs;
  }, []),
);

if (!args.url || !args.output || args.help) {
  console.log(`Usage: node render-ui-video.mjs --url http://127.0.0.1:5173/ --output /absolute/output.mp4
  [--width 1920] [--height 1080] [--duration 24] [--fps 30]
  [--seek-function __rlmSetVideoTime] [--start-selector '[data-record-start]']
  [--warmup-ms 1000] [--trim-start 0] [--keep-source]

When --seek-function is provided, the page must expose window[functionName](seconds).
The helper calls it for each exact frame timestamp, waits for browser painting,
and pipes PNG screenshots to FFmpeg. The hook should pause the page's own playback,
set all animation state directly, and return after state is committed. Disable
CSS transitions in capture mode so a frame represents the requested timestamp.

Without a seek function, Playwright records real time, then FFmpeg encodes MP4.
Optional --start-selector is clicked before the capture interval; --trim-start
removes that many extra seconds from the beginning of the final real-time clip.

Only localhost, 127.0.0.1, and ::1 URLs are accepted. Output defaults to silent
H.264 (libx264), CRF 18, yuv420p, and faststart. Existing output is overwritten.
No audio is generated or recorded.`);
  process.exit(args.help ? 0 : 1);
}

const url = new URL(String(args.url));
if (
  !['localhost', '127.0.0.1', '[::1]'].includes(url.hostname) ||
  !['http:', 'https:'].includes(url.protocol)
) {
  throw new Error('Capture URL must be a local HTTP(S) server.');
}
const width = Number(args.width || 1920);
const height = Number(args.height || 1080);
const fps = Number(args.fps || 30);
const duration = Number(args.duration || 24);
const warmup = Number(args['warmup-ms'] || 1000);
const trimStart = Number(args['trim-start'] || 0);
for (const [key, value] of Object.entries({ width, height, fps, duration })) {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid ${key}`);
}
if (width % 2 || height % 2)
  throw new Error('Width and height must be even for yuv420p.');
const output = resolve(String(args.output));
await mkdir(dirname(output), { recursive: true });
const scratch = await mkdtemp(`${tmpdir()}/rlm-ui-capture-`);
const ffmpeg = process.env.RLM_FFMPEG || 'ffmpeg';
const commonOutput = [
  '-an',
  '-c:v',
  'libx264',
  '-preset',
  'slow',
  '-crf',
  '18',
  '-pix_fmt',
  'yuv420p',
  '-movflags',
  '+faststart',
  output,
];
const browser = await chromium.launch({ headless: true });
let context;

function encode(commandArgs, inputPipe = false) {
  const child = spawn(ffmpeg, ['-hide_banner', '-y', ...commandArgs], {
    stdio: [inputPipe ? 'pipe' : 'ignore', 'inherit', 'inherit'],
  });
  const completion = new Promise((resolvePromise, reject) => {
    child.once('error', reject);
    child.once('close', (code) =>
      code === 0
        ? resolvePromise()
        : reject(new Error(`FFmpeg failed (${code})`)),
    );
  });
  return { child, completion };
}

try {
  context = await browser.newContext({
    viewport: { width, height },
    deviceScaleFactor: 1,
    colorScheme: 'dark',
    ...(args['seek-function']
      ? {}
      : { recordVideo: { dir: scratch, size: { width, height } } }),
  });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(url.href, { waitUntil: 'networkidle' });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(warmup);
  if (args['seek-function'])
    await page.waitForFunction(() => window.__rlmReady === true);
  if (args['start-selector'])
    await page.locator(String(args['start-selector'])).click();

  if (args['seek-function']) {
    const functionName = String(args['seek-function']);
    if (
      !(await page.evaluate(
        (name) => typeof window[name] === 'function',
        functionName,
      ))
    ) {
      throw new Error(`Page does not expose window.${functionName}(seconds)`);
    }
    const { child, completion } = encode(
      [
        '-f',
        'image2pipe',
        '-vcodec',
        'png',
        '-framerate',
        String(fps),
        '-i',
        'pipe:0',
        ...commonOutput,
      ],
      true,
    );
    const frames = Math.ceil(duration * fps);
    for (let frame = 0; frame < frames; frame += 1) {
      await page.evaluate(
        async ({ name, seconds }) => {
          await window[name](seconds);
          await new Promise((resolveFrame) =>
            requestAnimationFrame(() => requestAnimationFrame(resolveFrame)),
          );
        },
        { name: functionName, seconds: frame / fps },
      );
      const png = await page.screenshot({
        type: 'png',
        fullPage: false,
        animations: 'allow',
      });
      if (!child.stdin.write(png)) await once(child.stdin, 'drain');
      if (frame % Math.round(fps * 5) === 0)
        console.log(`Frame ${frame}/${frames}`);
    }
    child.stdin.end();
    await completion;
  } else {
    // Playwright video starts at page creation. This approximates its navigation
    // and warmup lead-in; --trim-start can remove any extra lead-in after review.
    const startStamp = performance.now();
    const startVideo = await page.video();
    await page.waitForTimeout((duration + trimStart) * 1000);
    const closingStamp = performance.now();
    await context.close();
    context = null;
    const source = await startVideo.path();
    const probe = spawn(ffmpeg.replace(/ffmpeg$/, 'ffprobe'), [
      '-v',
      'error',
      '-show_entries',
      'format=duration',
      '-of',
      'default=noprint_wrappers=1:nokey=1',
      source,
    ]);
    let sourceLength = '';
    probe.stdout.on('data', (chunk) => {
      sourceLength += chunk;
    });
    const [probeCode] = await once(probe, 'close');
    if (probeCode !== 0)
      throw new Error('Could not measure recorded video duration.');
    const elapsed = (closingStamp - startStamp) / 1000;
    const start = Math.max(
      0,
      Number(sourceLength.trim()) - elapsed + trimStart,
    );
    const { completion } = encode([
      '-ss',
      start.toFixed(3),
      '-i',
      source,
      '-t',
      String(duration),
      '-vf',
      `fps=${fps}`,
      ...commonOutput,
    ]);
    await completion;
  }
  if (errors.length) console.warn('Page errors during capture:', errors);
  console.log(`Saved ${output}`);
} finally {
  if (context) await context.close();
  await browser.close();
  if (!args['keep-source']) await rm(scratch, { recursive: true, force: true });
  else console.log(`Source directory: ${scratch}`);
}
