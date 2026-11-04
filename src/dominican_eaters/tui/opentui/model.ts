export type Workflow =
  | "config-validate"
  | "books-preflight"
  | "books-run"
  | "lyrics-preflight"
  | "lyrics-run"
  | "lyrics-download-audio"
  | "poems-preflight"
  | "poems-run"
  | "stt-preflight"
  | "stt-manifest-build"
  | "stt-benchmark"
  | "stt-suite"

export interface Preset {
  id: string
  label: string
  status: string
  reason: string
  runnable: boolean
  execution: "inline" | "worker"
  precision: string
  devices: string[]
  workerPython: string
}

export interface Dataset {
  id: string
  label: string
  manifest: string
  sampleCount?: number
}

export interface Bootstrap {
  python: string
  presets: Preset[]
  datasets?: Dataset[]
  suitePresets?: string[]
  initialState?: Partial<FormState>
  autoRun?: boolean
}

export const defaultDatasets: Dataset[] = [
  {
    id: "music",
    label: "Music / collected lyrics",
    manifest: "data/manifests/stt-all.json",
  },
]

export const t4SuitePresetIds = [
  "whisper-large-v3",
  "parakeet-tdt-0.6b-v3",
  "canary-1b-v2",
  "granite-speech-4.1-2b",
  "qwen3-asr-1.7b",
  "voxtral-mini-3b-2507",
] as const

export const workflows: { name: string; value: Workflow }[] = [
  ["Validate configuration", "config-validate"],
  ["Preflight books manifest", "books-preflight"],
  ["Collect books", "books-run"],
  ["Preflight lyrics manifest", "lyrics-preflight"],
  ["Collect lyrics", "lyrics-run"],
  ["Download collected lyrics audio", "lyrics-download-audio"],
  ["Preflight poems manifest", "poems-preflight"],
  ["Collect poems", "poems-run"],
  ["Preflight speech-to-text manifest", "stt-preflight"],
  ["Build speech-to-text manifest", "stt-manifest-build"],
  ["Run one speech-to-text model", "stt-benchmark"],
  ["Run full speech-to-text pipeline", "stt-suite"],
].map(([name, value]) => ({ name, value: value as Workflow }))

export const sources: Record<Exclude<Workflow, "stt-suite">, string> = {
  "config-validate": "config/default.yaml",
  "books-preflight": "data/manifests/books.json",
  "books-run": "data/manifests/books.json",
  "lyrics-preflight": "data/manifests/lyrics.json",
  "lyrics-run": "data/manifests/lyrics.json",
  "lyrics-download-audio": "artifacts/lyrics/lyrics-collection.json",
  "poems-preflight": "data/manifests/poems.json",
  "poems-run": "data/manifests/poems.json",
  "stt-preflight": "data/manifests/stt.json",
  "stt-manifest-build": "data/audio",
  "stt-benchmark": "data/manifests/stt.json",
}

export const outputs: Partial<Record<Workflow, string>> = {
  "books-run": "artifacts/books",
  "lyrics-run": "artifacts/lyrics",
  "lyrics-download-audio": "data/audio/lyrics",
  "poems-run": "artifacts/poems",
  "stt-manifest-build": "data/manifests/stt-all.json",
  "stt-benchmark": "artifacts/stt-run",
  "stt-suite": "artifacts/stt-suite",
}

export interface FormState {
  workflow: Workflow
  dataset: string
  source: string
  output: string
  dataRoot: string
  artifactsRoot: string
  preset: string
  device: string
  precision: string
  workerPython: string
  warmupRuns: string
  requestTimeout: string
  shortAudioPolicy: string
  minimumAudioSeconds: string
  force: boolean
  verifyHashes: boolean
  timestamps: boolean
}

export function initialState(overrides: Partial<FormState> = {}): FormState {
  const state: FormState = {
    workflow: "config-validate",
    dataset: "all",
    source: sources["config-validate"],
    output: "",
    dataRoot: "",
    artifactsRoot: "",
    preset: "whisper-large-v3",
    device: "cuda",
    precision: "fp16",
    workerPython: "",
    warmupRuns: "1",
    requestTimeout: "1800",
    shortAudioPolicy: "reject",
    minimumAudioSeconds: "0.1",
    force: false,
    verifyHashes: true,
    timestamps: false,
  }

  const workflow = overrides.workflow
  if (workflows.some((candidate) => candidate.value === workflow)) {
    changeWorkflow(state, workflow as Workflow)
  }
  for (const [key, value] of Object.entries(overrides)) {
    if (key === "workflow" || !(key in state)) continue
    const stateKey = key as keyof FormState
    if (typeof value === typeof state[stateKey]) {
      ;(state as unknown as Record<string, unknown>)[key] = value
    }
  }
  return state
}

export const hasOutput = (workflow: Workflow) => outputs[workflow] !== undefined
export const isStt = (workflow: Workflow) =>
  workflow === "stt-preflight" ||
  workflow === "stt-benchmark" ||
  workflow === "stt-suite"

export function changeWorkflow(state: FormState, workflow: Workflow) {
  const oldSource = state.workflow === "stt-suite" ? "" : sources[state.workflow]
  const oldOutput = outputs[state.workflow] ?? ""
  const newSource = workflow === "stt-suite" ? "" : sources[workflow]
  if (!state.source.trim() || state.source === oldSource) state.source = newSource
  if (!state.output.trim() || state.output === oldOutput) state.output = outputs[workflow] ?? ""
  state.workflow = workflow
}

const need = (value: string, message: string) => {
  const normalized = value.trim()
  if (!normalized) throw Error(message)
  return normalized
}

const positive = (value: string, label: string) => {
  if (!Number.isFinite(Number(value)) || Number(value) <= 0) {
    throw Error(`${label} must be a positive number.`)
  }
  return value.trim()
}

const quote = (value: string) =>
  /^[\w@%+=:,./-]+$/.test(value) ? value : `'${value.replaceAll("'", "'\\''")}'`

export function selectedDatasets(state: FormState, datasets: Dataset[]): Dataset[] {
  if (state.dataset === "all") return datasets
  const selected = datasets.find((dataset) => dataset.id === state.dataset)
  if (!selected) throw Error(`Unsupported dataset: ${state.dataset}`)
  return [selected]
}

export function buildArgs(
  state: FormState,
  presets: Preset[],
  datasets: Dataset[] = defaultDatasets,
  suitePresetIds: readonly string[] = t4SuitePresetIds,
): string[] {
  if (state.workflow === "stt-suite") {
    const selected = selectedDatasets(state, datasets)
    if (!selected.length) throw Error("No speech-to-text datasets are configured.")
    if (!/^\d+$/.test(state.warmupRuns.trim())) {
      throw Error("Warmup runs must be a nonnegative integer.")
    }
    const suitePresets = [...new Set(suitePresetIds)]
      .map((id) => presets.find((preset) => preset.id === id))
      .filter((preset): preset is Preset => preset !== undefined && preset.runnable)
    if (!suitePresets.length) {
      throw Error("No runnable speech-to-text model presets are configured.")
    }
    const missingWorkers = suitePresets.filter(
      (preset) => preset.execution === "worker" && !preset.workerPython.trim(),
    )
    if (missingWorkers.length) {
      throw Error(
        `Missing worker environments: ${missingWorkers.map((preset) => preset.id).join(", ")}.`,
      )
    }
    const args = ["stt", "suite"]
    for (const dataset of selected) {
      args.push(
        "--dataset",
        `${dataset.id}=${need(dataset.manifest, `Dataset ${dataset.id} has no manifest.`)}`,
      )
    }
    for (const preset of suitePresets) args.push("--preset", preset.id)
    args.push(
      "--output-dir",
      need(state.output, "Select an output directory."),
      "--device",
      state.device,
      "--precision",
      state.precision,
      "--warmup-runs",
      state.warmupRuns.trim(),
      "--request-timeout",
      positive(state.requestTimeout, "Request timeout"),
    )
    args.push(state.verifyHashes ? "--verify-hashes" : "--no-verify-hashes")
    if (state.timestamps) args.push("--timestamps")
    return args
  }

  const source = need(
    state.source,
    state.workflow === "config-validate"
      ? "Select a configuration file."
      : "Select a manifest file.",
  )
  if (state.workflow === "config-validate") {
    const args = ["config", "validate", source]
    if (state.dataRoot.trim()) args.push("--data-root", state.dataRoot.trim())
    if (state.artifactsRoot.trim()) args.push("--artifacts-root", state.artifactsRoot.trim())
    return args
  }

  const domain: Partial<Record<Workflow, string[]>> = {
    "books-preflight": ["collect", "books", "preflight"],
    "books-run": ["collect", "books", "run"],
    "lyrics-preflight": ["collect", "lyrics", "preflight"],
    "lyrics-run": ["collect", "lyrics", "run"],
    "poems-preflight": ["collect", "poems", "preflight"],
    "poems-run": ["collect", "poems", "run"],
  }
  if (domain[state.workflow]) {
    const args = [...domain[state.workflow]!, source]
    if (["books-run", "lyrics-run", "poems-run"].includes(state.workflow)) {
      args.push("--output-dir", need(state.output, "Select an output directory."))
      if (state.force) args.push("--force")
    }
    return args
  }
  if (state.workflow === "lyrics-download-audio") {
    const args = [
      "collect",
      "lyrics",
      "download-audio",
      source,
      "--output-dir",
      need(state.output, "Select an output directory."),
    ]
    if (state.force) args.push("--force")
    return args
  }
  if (state.workflow === "stt-manifest-build") {
    return [
      "stt",
      "manifest",
      "build",
      source,
      "--output-file",
      need(state.output, "Select an output file."),
    ]
  }

  const preset = presets.find((candidate) => candidate.id === state.preset)
  if (!preset) throw Error(`Unsupported preset: ${state.preset}`)
  if (state.workflow === "stt-benchmark" && !preset.runnable) {
    throw Error(
      `Preset ${preset.id} is ${preset.status} and cannot be benchmarked: ${preset.reason}`,
    )
  }
  if (preset.execution === "worker" && !state.workerPython.trim()) {
    throw Error("Select the isolated worker Python executable.")
  }

  const args = [
    "stt",
    state.workflow === "stt-preflight" ? "preflight" : "benchmark",
    source,
  ]
  if (state.workflow === "stt-preflight" && state.dataRoot.trim()) {
    args.push("--dataset-root", state.dataRoot.trim())
  }
  if (state.workflow === "stt-benchmark") {
    args.push("--output-dir", need(state.output, "Select an output directory."))
  }
  args.push("--preset", preset.id, "--device", state.device, "--precision", state.precision)
  if (state.workflow === "stt-benchmark") {
    if (!/^\d+$/.test(state.warmupRuns.trim())) {
      throw Error("Warmup runs must be a nonnegative integer.")
    }
    args.push(
      "--warmup-runs",
      state.warmupRuns.trim(),
      "--request-timeout",
      positive(state.requestTimeout, "Request timeout"),
      "--short-audio-policy",
      state.shortAudioPolicy,
      "--minimum-audio-seconds",
      positive(state.minimumAudioSeconds, "Minimum audio duration"),
    )
  }
  if (state.workerPython.trim()) args.push("--worker-python", state.workerPython.trim())
  if (state.verifyHashes) args.push("--verify-hashes")
  if (state.timestamps && state.workflow === "stt-benchmark") args.push("--timestamps")
  return args
}

export function commandPreview(
  state: FormState,
  presets: Preset[],
  datasets: Dataset[] = defaultDatasets,
  suitePresetIds: readonly string[] = t4SuitePresetIds,
) {
  try {
    return ["dominican-eaters", ...buildArgs(state, presets, datasets, suitePresetIds)]
      .map(quote)
      .join(" ")
  } catch (error) {
    return `Complete the form: ${(error as Error).message}`
  }
}

export type SuiteEventName =
  | "suite_started"
  | "suite_finished"
  | "suite_interrupted"
  | "run_started"
  | "run_resumed"
  | "phase_started"
  | "sample_started"
  | "sample_completed"
  | "sample_restored"
  | "run_finished"
  | "run_interrupted"
  | "task_started"
  | "task_skipped"
  | "phase_changed"
  | "task_completed"
  | "suite_completed"

export interface SuiteEvent {
  event: SuiteEventName
  dataset?: string | null
  dataset_id?: string | null
  preset?: string | null
  preset_id?: string | null
  model?: string | null
  phase?: string | null
  completed?: number | null
  succeeded?: number | null
  failed?: number | null
  total?: number | null
  index?: number | null
  sample_id?: string | null
  audio_path?: string | null
  status?: string | null
  run_status?: string | null
  suite_status?: string | null
  task_status?: string | null
  task_index?: number | null
  task_total?: number | null
  elapsed_seconds?: number | null
  audio_seconds?: number | null
  error_type?: string | null
  error_code?: string | null
  message?: string | null
}

export interface ModelProgress {
  key: string
  dataset: string
  preset: string
  model: string
  phase: string
  status: string
  completed: number
  succeeded: number
  failed: number
  total: number
  activeSample: string
  elapsedSeconds: number | null
  startedAtMilliseconds: number | null
  failureType: string
  failureMessage: string
}

export interface SampleProgress {
  key: string
  dataset: string
  preset: string
  sampleId: string
  audioPath: string
  index: number
  total: number
  status: string
  elapsedSeconds: number
  errorType: string
  errorCode: string
  message: string
}

export interface SuiteProgress {
  status: string
  models: Map<string, ModelProgress>
  samples: Map<string, SampleProgress>
}

export const initialSuiteProgress = (): SuiteProgress => ({
  status: "waiting",
  models: new Map(),
  samples: new Map(),
})

export function seedSuiteProgress(datasets: Dataset[], presets: Preset[]): SuiteProgress {
  const progress = initialSuiteProgress()
  for (const dataset of datasets) {
    for (const preset of presets) {
      const key = `${dataset.id}::${preset.id}`
      progress.models.set(key, {
        key,
        dataset: dataset.id,
        preset: preset.id,
        model: preset.label,
        phase: preset.execution === "worker" && !preset.workerPython ? "setup" : "queued",
        status:
          preset.execution === "worker" && !preset.workerPython ? "missing-worker" : "queued",
        completed: 0,
        succeeded: 0,
        failed: 0,
        total: dataset.sampleCount ?? 0,
        activeSample: "",
        elapsedSeconds: null,
        startedAtMilliseconds: null,
        failureType: "",
        failureMessage: "",
      })
    }
  }
  return progress
}

const textValue = (value: unknown) => (typeof value === "string" ? value : "")
const numberValue = (value: unknown) =>
  typeof value === "number" && Number.isFinite(value) ? value : 0

export function parseSuiteEvent(line: string): SuiteEvent | null {
  const marker = "stt_event="
  const offset = line.indexOf(marker)
  if (offset < 0) return null
  try {
    const payload = JSON.parse(line.slice(offset + marker.length)) as Record<string, unknown>
    const event = textValue(payload.event || payload.kind) as SuiteEventName
    if (!event) return null
    return { ...payload, event } as SuiteEvent
  } catch {
    return null
  }
}

export function applySuiteEvent(
  progress: SuiteProgress,
  event: SuiteEvent,
  nowMilliseconds = Date.now(),
): SuiteProgress {
  const dataset = textValue(event.dataset_id) || textValue(event.dataset) || "dataset"
  const preset = textValue(event.preset_id) || textValue(event.preset) || "model"
  const modelKey = `${dataset}::${preset}`
  const prior = progress.models.get(modelKey)
  const model: ModelProgress = prior
    ? { ...prior }
    : {
        key: modelKey,
        dataset,
        preset,
        model: textValue(event.model),
        phase: "queued",
        status: "queued",
        completed: 0,
        succeeded: 0,
        failed: 0,
        total: numberValue(event.total),
        activeSample: "",
        elapsedSeconds: null,
        startedAtMilliseconds: null,
        failureType: "",
        failureMessage: "",
      }
  model.model = textValue(event.model) || model.model
  model.phase = textValue(event.phase) || model.phase
  const suiteTaskEvent = ["task_started", "task_skipped", "task_completed"].includes(
    event.event,
  )
  if (!suiteTaskEvent) {
    model.total = numberValue(event.total) || model.total
    model.completed = numberValue(event.completed)
    model.succeeded = numberValue(event.succeeded)
    model.failed = numberValue(event.failed)
  }

  if (["run_started", "run_resumed", "task_started"].includes(event.event)) {
    model.status = "running"
    if (model.startedAtMilliseconds == null) {
      model.startedAtMilliseconds = nowMilliseconds
      model.elapsedSeconds = 0
    }
    if (event.event === "task_started" && model.phase === "queued") model.phase = "preflight"
  } else if (["phase_started", "phase_changed"].includes(event.event)) {
    model.status = "running"
  } else if (event.event === "sample_started") {
    model.status = "running"
    model.activeSample = textValue(event.sample_id)
  } else if (["sample_completed", "sample_restored"].includes(event.event)) {
    model.status = textValue(event.status) === "failed" ? "warning" : "running"
    model.activeSample = ""
    model.completed = numberValue(event.completed) || numberValue(event.index) || model.completed
    if (event.succeeded == null && event.failed == null) {
      if (textValue(event.status) === "failed") model.failed += 1
      else model.succeeded += 1
    }
  } else if (event.event === "task_skipped") {
    model.status = textValue(event.task_status) || "complete"
    model.phase = "resumed"
    model.succeeded = numberValue(event.succeeded)
    model.failed = numberValue(event.failed)
    model.total = numberValue(event.total) || model.total
    model.completed = model.succeeded + model.failed
    model.activeSample = ""
  } else if (event.event === "task_completed") {
    model.status = textValue(event.task_status) || "complete"
    model.succeeded = numberValue(event.succeeded)
    model.failed = numberValue(event.failed)
    model.total = numberValue(event.total) || model.total
    model.completed = model.succeeded + model.failed
    model.activeSample = ""
    model.failureType = textValue(event.error_type)
    model.failureMessage = textValue(event.message)
  } else if (event.event === "run_finished") {
    model.status = textValue(event.run_status) || textValue(event.status) || "complete"
    model.activeSample = ""
  } else if (event.event === "run_interrupted") {
    model.status = "interrupted"
    model.activeSample = ""
  }

  if (model.startedAtMilliseconds != null) {
    model.elapsedSeconds = Math.max(0, (nowMilliseconds - model.startedAtMilliseconds) / 1000)
  }

  const models = new Map(progress.models)
  if (!event.event.startsWith("suite_")) models.set(modelKey, model)

  const samples = new Map(progress.samples)
  if (["sample_started", "sample_completed", "sample_restored"].includes(event.event)) {
    const sampleId = textValue(event.sample_id) || "sample"
    const sample: SampleProgress = {
      key: `${modelKey}::${sampleId}`,
      dataset,
      preset,
      sampleId,
      audioPath: textValue(event.audio_path),
      index: numberValue(event.index),
      total: numberValue(event.total),
      status: event.event === "sample_started" ? "running" : textValue(event.status) || "ok",
      elapsedSeconds: numberValue(event.elapsed_seconds),
      errorType: textValue(event.error_type),
      errorCode: textValue(event.error_code),
      message: textValue(event.message) || textValue(event.error_type),
    }
    samples.delete(sample.key)
    samples.set(sample.key, sample)
  }

  let status = progress.status
  if (event.event === "suite_started") status = "running"
  if (["suite_finished", "suite_completed"].includes(event.event)) {
    status =
      textValue(event.suite_status) ||
      textValue(event.status) ||
      textValue(event.run_status) ||
      "complete"
  }
  if (event.event === "suite_interrupted") status = "interrupted"
  return { status, models, samples }
}

const compact = (value: string, width: number) =>
  value.length <= width ? value.padEnd(width) : `${value.slice(0, Math.max(1, width - 1))}…`
const duration = (seconds: number | null) => {
  if (seconds == null) return "--:--"
  const value = Math.max(0, Math.floor(seconds))
  return `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`
}

export function renderModelProgress(progress: SuiteProgress, width = 100): string {
  const nameWidth = Math.max(18, Math.min(42, width - 48))
  const header = `${compact("DATASET / MODEL", nameWidth)}  ${compact("PHASE", 13)}  PROGRESS       FAIL  TIME`
  const rows = [...progress.models.values()].map((model) => {
    const label = `${model.dataset} / ${model.preset}`
    const count = `${model.completed}/${model.total || "?"}`.padEnd(13)
    const terminal = ["complete", "partial", "failed", "interrupted", "missing-worker"].includes(
      model.status,
    )
    return `${compact(label, nameWidth)}  ${compact(terminal ? model.status : model.phase || model.status, 13)}  ${count}  ${String(model.failed).padStart(4)}  ${duration(model.elapsedSeconds)}`
  })
  const failures = dominantFailures(progress)
  const failureRows = failures.map((failure) => {
    const label = compact(`${failure.dataset} / ${failure.preset}`, nameWidth)
    const detailWidth = Math.max(20, width - nameWidth - 10)
    return `${label}  ${String(failure.count).padStart(4)}x  ${compact(failure.detail, detailWidth).trimEnd()}`
  })
  return [
    header,
    ...(rows.length ? rows : ["Waiting for the pipeline to start..."]),
    ...(failureRows.length ? ["", "DOMINANT FAILURES", ...failureRows] : []),
  ].join("\n")
}

interface FailureSummary {
  dataset: string
  preset: string
  count: number
  detail: string
}

function dominantFailures(progress: SuiteProgress): FailureSummary[] {
  const grouped = new Map<string, Map<string, FailureSummary>>()
  for (const sample of progress.samples.values()) {
    if (sample.status !== "failed") continue
    const modelKey = `${sample.dataset}::${sample.preset}`
    const detail = [sample.errorCode || sample.errorType, sample.message]
      .filter((part, index, all) => part && all.indexOf(part) === index)
      .join(": ") || "Unknown sample failure"
    const failures = grouped.get(modelKey) ?? new Map<string, FailureSummary>()
    const current = failures.get(detail)
    failures.set(detail, {
      dataset: sample.dataset,
      preset: sample.preset,
      count: (current?.count ?? 0) + 1,
      detail,
    })
    grouped.set(modelKey, failures)
  }
  const summaries: FailureSummary[] = []
  for (const model of progress.models.values()) {
    const failures = grouped.get(model.key)
    const dominant = failures
      ? [...failures.values()].sort(
          (left, right) => right.count - left.count || left.detail.localeCompare(right.detail),
        )[0]
      : undefined
    if (dominant) {
      summaries.push(dominant)
    } else if (model.failureMessage || model.failureType) {
      summaries.push({
        dataset: model.dataset,
        preset: model.preset,
        count: Math.max(1, model.failed),
        detail: [model.failureType, model.failureMessage]
          .filter((part, index, all) => part && all.indexOf(part) === index)
          .join(": "),
      })
    }
  }
  return summaries
}

export function renderSampleProgress(progress: SuiteProgress, width = 100): string {
  if (!progress.samples.size) return "No samples processed yet."
  const presetWidth = Math.max(16, Math.min(28, width - 76))
  const sampleWidth = Math.max(18, Math.min(42, width - presetWidth - 52))
  const header = `${compact("DATASET", 12)}  ${compact("MODEL", presetWidth)}  ${compact("SAMPLE", sampleWidth)}  POSITION     STATUS    TIME`
  const detailedFailures = new Set<string>()
  const rows = [...progress.samples.values()]
    .reverse()
    .map((sample) => {
      const position = sample.total ? `${sample.index}/${sample.total}` : String(sample.index || "-")
      const failureKey = `${sample.dataset}::${sample.preset}::${sample.errorCode || sample.errorType}::${sample.message}`
      const showDetail =
        Boolean(sample.message) &&
        (sample.status !== "failed" || !detailedFailures.has(failureKey))
      if (sample.status === "failed") detailedFailures.add(failureKey)
      const detail = showDetail ? ` · ${sample.message}` : ""
      return `${compact(sample.dataset, 12)}  ${compact(sample.preset, presetWidth)}  ${compact(sample.sampleId, sampleWidth)}  ${position.padEnd(11)}  ${sample.status.padEnd(8)}  ${duration(sample.elapsedSeconds)}${detail}`
    })
  return [header, ...rows].join("\n")
}
