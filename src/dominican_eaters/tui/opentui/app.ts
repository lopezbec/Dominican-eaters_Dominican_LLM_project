import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process"
import {
  BoxRenderable,
  CliRenderEvents,
  InputRenderable,
  InputRenderableEvents,
  ScrollBoxRenderable,
  TextRenderable,
  createCliRenderer,
  type CliRenderer,
  type KeyEvent,
  type RenderContext,
} from "@opentui/core"
import {
  applySuiteEvent,
  buildArgs,
  changeWorkflow,
  commandPreview,
  defaultDatasets,
  hasOutput,
  initialState,
  isStt,
  parseSuiteEvent,
  renderModelProgress,
  renderSampleProgress,
  seedSuiteProgress,
  selectedDatasets,
  t4SuitePresetIds,
  workflows,
  type Bootstrap,
  type Dataset,
  type FormState,
  type Preset,
  type SuiteEvent,
  type SuiteProgress,
} from "./model.ts"

const C = {
  bg: "#0c0e0f",
  panel: "#121516",
  raised: "#181c1d",
  line: "#303637",
  muted: "#899192",
  fg: "#e6e9e7",
  green: "#77c38b",
  amber: "#d6ad65",
  red: "#e07373",
}

type Field = {
  key: keyof FormState
  box: BoxRenderable
  value: TextRenderable | InputRenderable
  kind: "input" | "choice" | "toggle"
  options?: string[]
}

export class DominicanEatersTui {
  readonly renderer: CliRenderer
  readonly bootstrap: Bootstrap
  readonly state = initialState()
  readonly fields: Field[] = []
  readonly presets: Preset[]
  readonly datasets: Dataset[]
  readonly suitePresets: Preset[]

  private workspace!: BoxRenderable
  private controls!: BoxRenderable
  private pinned!: BoxRenderable
  private form!: ScrollBoxRenderable
  private preview!: TextRenderable
  private status!: TextRenderable
  private runState!: TextRenderable
  private consoleBox!: BoxRenderable
  private progressBox!: ScrollBoxRenderable
  private modelProgress!: TextRenderable
  private sampleProgress!: TextRenderable
  private logBox!: ScrollBoxRenderable
  private output!: TextRenderable
  private outputState!: TextRenderable
  private focus = 0
  private maximized = false
  private progressVisible = true
  private child: ChildProcessWithoutNullStreams | null = null
  private logs: string[] = []
  private started = 0
  private timer?: NodeJS.Timeout
  private progress: SuiteProgress
  private activeTask: { dataset: string; preset: string } | null = null

  constructor(renderer: CliRenderer, bootstrap: Bootstrap) {
    this.renderer = renderer
    this.bootstrap = bootstrap
    this.presets = bootstrap.presets
    this.datasets = bootstrap.datasets?.length ? bootstrap.datasets : defaultDatasets
    const suiteIds = bootstrap.suitePresets?.length
      ? bootstrap.suitePresets
      : t4SuitePresetIds
    this.suitePresets = suiteIds
      .map((id) => this.presets.find((preset) => preset.id === id))
      .filter((preset): preset is Preset => preset !== undefined && preset.runnable)
    this.progress = seedSuiteProgress(this.datasets, this.suitePresets)
    this.build()
  }

  private text(ctx: RenderContext, content: string, fg = C.fg, options: object = {}) {
    return new TextRenderable(ctx, { content, fg, ...options })
  }

  private build() {
    const renderer = this.renderer
    renderer.root.flexDirection = "column"

    const top = new BoxRenderable(renderer, {
      height: 3,
      flexDirection: "row",
      paddingX: 2,
      alignItems: "center",
      backgroundColor: C.panel,
    })
    top.add(this.text(renderer, "DOMINICAN EATERS", C.fg, { width: 22 }))
    top.add(this.text(renderer, "WORKFLOW CONSOLE", C.muted, { flexGrow: 1 }))
    this.runState = this.text(renderer, "● READY", C.green, { width: 13 })
    top.add(this.runState)
    renderer.root.add(top)

    this.workspace = new BoxRenderable(renderer, {
      flexGrow: 1,
      flexDirection: "row",
      padding: 1,
      gap: 1,
    })
    renderer.root.add(this.workspace)

    this.controls = new BoxRenderable(renderer, {
      width: 42,
      minWidth: 34,
      flexDirection: "column",
      border: true,
      borderColor: C.line,
      backgroundColor: C.panel,
    })
    this.controls.add(this.text(renderer, " RUN CONFIGURATION", C.muted, { height: 2 }))
    this.pinned = new BoxRenderable(renderer, {
      flexDirection: "column",
      paddingX: 1,
      border: ["bottom"],
      borderColor: C.line,
    })
    this.controls.add(this.pinned)
    this.form = new ScrollBoxRenderable(renderer, {
      flexGrow: 1,
      flexDirection: "column",
      paddingX: 1,
    })
    this.controls.add(this.form)

    this.choice("workflow", "Workflow", workflows.map((workflow) => workflow.value))
    this.choice("dataset", "Dataset", ["all", ...this.datasets.map((dataset) => dataset.id)])
    this.input("source", "Source")
    this.input("output", "Output directory")
    this.input("dataRoot", "Dataset/data root override")
    this.input("artifactsRoot", "Artifacts root override")
    this.choice("preset", "Model preset", this.presets.map((preset) => preset.id))
    this.choice("device", "Device", ["auto", "cpu", "cuda"])
    this.choice("precision", "Precision", ["auto", "fp16", "fp32", "bf16"])
    this.input("workerPython", "Worker Python")
    this.input("warmupRuns", "Warmup runs")
    this.input("requestTimeout", "Timeout (seconds)")
    this.choice("shortAudioPolicy", "Short audio", ["reject", "allow"])
    this.input("minimumAudioSeconds", "Minimum seconds")
    this.toggle("force", "Force reprocessing")
    this.toggle("verifyHashes", "Verify hashes")
    this.toggle("timestamps", "Request timestamps")

    const dock = new BoxRenderable(renderer, {
      minHeight: 8,
      flexDirection: "column",
      padding: 1,
      backgroundColor: C.raised,
      border: ["top"],
      borderColor: C.line,
    })
    dock.add(this.text(renderer, "COMMAND", C.muted))
    this.preview = this.text(renderer, "", C.fg, { height: 3 })
    dock.add(this.preview)
    this.status = this.text(renderer, "Ready", C.green)
    dock.add(this.status)
    dock.add(this.text(renderer, "[ Ctrl+R Run ]  [ Ctrl+C Stop ]", C.green))
    this.controls.add(dock)

    this.consoleBox = new BoxRenderable(renderer, {
      flexGrow: 1,
      flexDirection: "column",
      border: true,
      borderColor: C.line,
      backgroundColor: C.panel,
    })

    this.progressBox = new ScrollBoxRenderable(renderer, {
      height: 20,
      flexDirection: "column",
      border: ["bottom"],
      borderColor: C.line,
      paddingX: 1,
      visible: false,
    })
    this.progressBox.add(
      this.text(renderer, "PIPELINE PROGRESS    Ctrl+J/K scroll", C.muted, { height: 2 }),
    )
    this.modelProgress = this.text(renderer, "", C.fg, { minHeight: 3 })
    this.progressBox.add(this.modelProgress)
    this.progressBox.add(this.text(renderer, "SAMPLE ROWS", C.muted, { height: 2 }))
    this.sampleProgress = this.text(renderer, "", C.fg, { minHeight: 2 })
    this.progressBox.add(this.sampleProgress)
    this.consoleBox.add(this.progressBox)

    this.consoleBox.add(
      this.text(
        renderer,
        " CONSOLE       Ctrl+P progress · Ctrl+O maximize · Ctrl+L clear",
        C.muted,
        { height: 2 },
      ),
    )
    this.logBox = new ScrollBoxRenderable(renderer, {
      flexGrow: 1,
      stickyScroll: true,
      stickyStart: "bottom",
      padding: 1,
    })
    this.output = this.text(renderer, "Waiting for a run", C.fg, {
      padding: 1,
      selectable: true,
    })
    this.logBox.add(this.output)
    this.consoleBox.add(this.logBox)
    this.outputState = this.text(renderer, "OUTPUT  waiting for a run", C.muted, {
      height: 2,
      paddingLeft: 1,
      backgroundColor: C.raised,
    })
    this.consoleBox.add(this.outputState)
    this.workspace.add(this.controls)
    this.workspace.add(this.consoleBox)
    renderer.root.add(
      this.text(
        renderer,
        " Tab navigate  ←/→ choose  Space toggle  PgUp/PgDn log  Ctrl+R run  Ctrl+P progress  Ctrl+Q quit",
        C.muted,
        { height: 1 },
      ),
    )

    renderer.keyInput.on("keypress", (key: KeyEvent) => this.key(key))
    renderer.on(CliRenderEvents.RESIZE, () => this.responsive())
    this.sync()
    this.focusField(0)
    this.responsive()
  }

  private base(
    key: keyof FormState,
    label: string,
    value: TextRenderable | InputRenderable,
    kind: Field["kind"],
    options?: string[],
  ) {
    const box = new BoxRenderable(this.renderer, {
      height: 4,
      flexDirection: "column",
      marginBottom: 1,
      border: ["left"],
      borderColor: C.line,
      paddingLeft: 1,
    })
    box.add(this.text(this.renderer, label, C.muted))
    box.add(value)
    ;(["workflow", "dataset", "preset"].includes(key) ? this.pinned : this.form).add(box)
    this.fields.push({ key, box, value, kind, options })
  }

  private input(key: keyof FormState, label: string) {
    const value = new InputRenderable(this.renderer, {
      value: String(this.state[key]),
      textColor: C.fg,
      focusedTextColor: C.green,
      placeholderColor: C.muted,
      backgroundColor: C.bg,
      focusedBackgroundColor: C.bg,
    })
    value.on(InputRenderableEvents.INPUT, (input: string) => {
      ;(this.state as any)[key] = input
      this.refresh()
    })
    this.base(key, label, value, "input")
  }

  private choice(key: keyof FormState, label: string, options: string[]) {
    this.base(key, label, this.text(this.renderer, ""), "choice", options)
  }

  private toggle(key: keyof FormState, label: string) {
    this.base(key, label, this.text(this.renderer, ""), "toggle")
  }

  private preset() {
    return this.presets.find((preset) => preset.id === this.state.preset) ?? this.presets[0]!
  }

  private visible(field: Field) {
    const workflow = this.state.workflow
    const key = String(field.key)
    if (key === "dataset") return workflow === "stt-suite"
    if (key === "source") return workflow !== "stt-suite"
    if (key === "output") return hasOutput(workflow)
    if (key === "dataRoot") return workflow === "config-validate" || workflow === "stt-preflight"
    if (key === "artifactsRoot") return workflow === "config-validate"
    if (key === "preset") return workflow === "stt-preflight" || workflow === "stt-benchmark"
    if (["device", "precision"].includes(key)) return isStt(workflow)
    if (key === "workerPython") {
      return workflow !== "stt-suite" && isStt(workflow) && this.preset().execution === "worker"
    }
    if (["warmupRuns", "requestTimeout", "timestamps"].includes(key)) {
      return workflow === "stt-benchmark" || workflow === "stt-suite"
    }
    if (["shortAudioPolicy", "minimumAudioSeconds"].includes(key)) {
      return workflow === "stt-benchmark"
    }
    if (key === "force") {
      return ["books-run", "lyrics-run", "lyrics-download-audio", "poems-run"].includes(workflow)
    }
    if (key === "verifyHashes") return isStt(workflow)
    return true
  }

  private render(field: Field) {
    let display = ""
    if (field.kind === "toggle") {
      display = (this.state[field.key] ? "[✓] " : "[ ] ") + "Space to toggle"
    } else if (field.key === "workflow") {
      display = workflows.find((workflow) => workflow.value === this.state.workflow)?.name ?? ""
    } else if (field.key === "dataset") {
      display =
        this.state.dataset === "all"
          ? `All datasets (${this.datasets.length})`
          : this.datasets.find((dataset) => dataset.id === this.state.dataset)?.label ??
            this.state.dataset
    } else if (field.key === "preset") {
      const preset = this.preset()
      display = `${preset.label} · ${preset.status.toUpperCase()}`
    } else {
      display = String(this.state[field.key])
    }
    ;(field.value as TextRenderable).content = `‹ ${display} ›`
  }

  private sync() {
    const pinnedFields = this.state.workflow === "stt-suite" ? 2 : isStt(this.state.workflow) ? 2 : 1
    this.pinned.height = pinnedFields * 5
    for (const field of this.fields) {
      field.box.visible = this.visible(field)
      if (field.kind === "input") {
        ;(field.value as InputRenderable).value = String(this.state[field.key])
      } else {
        this.render(field)
      }
    }
    this.refresh()
    this.updateProgressVisibility()
    if (!this.fields[this.focus]?.box.visible) this.focusField(this.next(1))
  }

  private refresh() {
    this.preview.content = commandPreview(this.state, this.presets, this.datasets)
  }

  private focusField(index: number) {
    const old = this.fields[this.focus]
    if (old) {
      old.box.borderColor = C.line
      old.value.blur()
    }
    this.focus = index
    const field = this.fields[index]!
    field.box.borderColor = C.green
    field.value.focus()
    if (!["workflow", "dataset", "preset"].includes(String(field.key))) {
      this.form.scrollTo(field.box)
    }
  }

  private next(direction: number) {
    let index = this.focus
    do {
      index = (index + direction + this.fields.length) % this.fields.length
    } while (!this.fields[index]!.box.visible)
    return index
  }

  private choose(direction: number) {
    const field = this.fields[this.focus]!
    if (field.kind === "toggle") {
      ;(this.state as any)[field.key] = !this.state[field.key]
      this.render(field)
      this.refresh()
      return
    }
    if (field.kind !== "choice") return
    const values = field.options ?? []
    const current = String(this.state[field.key])
    const value = values[(values.indexOf(current) + direction + values.length) % values.length]!
    if (field.key === "workflow") {
      changeWorkflow(this.state, value as FormState["workflow"])
    } else {
      ;(this.state as any)[field.key] = value
      if (field.key === "preset") {
        const preset = this.preset()
        this.state.workerPython = preset.workerPython
        this.state.precision = preset.precision || "auto"
        this.state.device = preset.devices.includes("cuda")
          ? "cuda"
          : preset.devices[0] || "auto"
      }
    }
    this.sync()
  }

  private key(key: KeyEvent) {
    if (key.ctrl && key.name === "q") {
      void this.quit()
      return
    }
    if (key.ctrl && key.name === "r") {
      void this.run()
      return
    }
    if (key.ctrl && key.name === "l") {
      this.logs = []
      this.output.content = ""
      this.outputState.content = "OUTPUT  log cleared"
      return
    }
    if (key.ctrl && key.name === "p") {
      this.progressVisible = !this.progressVisible
      this.updateProgressVisibility()
      return
    }
    if (key.ctrl && ["j", "k"].includes(key.name)) {
      this.progressBox.scrollBy(key.name === "j" ? 3 : -3, "step")
      return
    }
    if (["pageup", "pagedown"].includes(key.name)) {
      this.logBox.scrollBy(key.name === "pagedown" ? 1 : -1, "viewport")
      return
    }
    if (key.ctrl && key.name === "o") {
      this.maximized = !this.maximized
      this.controls.visible = !this.maximized
      this.workspace.padding = this.maximized ? 0 : 1
      return
    }
    if (key.ctrl && key.name === "c" && this.child) {
      this.child.kill("SIGTERM")
      this.setStatus("Stopping…", C.amber)
      return
    }
    if (key.name === "tab") {
      this.focusField(this.next(key.shift ? -1 : 1))
      return
    }
    const field = this.fields[this.focus]!
    if (field.kind !== "input" && ["left", "right", "space", "return"].includes(key.name)) {
      this.choose(key.name === "left" ? -1 : 1)
    }
  }

  private responsive() {
    const narrow = this.renderer.root.width < 86
    this.workspace.flexDirection = narrow ? "column" : "row"
    this.controls.width = narrow ? "100%" : 42
    this.controls.height = narrow ? "65%" : "100%"
    this.sampleProgress.visible = !narrow && this.renderer.root.height >= 34
    this.progressBox.height = narrow ? 10 : Math.min(22, Math.max(14, this.renderer.root.height - 20))
    this.renderProgress()
  }

  private updateProgressVisibility() {
    this.progressBox.visible = this.state.workflow === "stt-suite" && this.progressVisible
  }

  private line(message: string) {
    this.logs.push(
      `${new Date().toLocaleTimeString("en-GB", { hour12: false })}  ${message}`,
    )
    if (this.logs.length > 1000) this.logs.shift()
    this.output.content = this.logs.join("\n")
    this.metrics()
  }

  private eventLine(event: SuiteEvent) {
    const dataset = event.dataset_id || event.dataset || "dataset"
    const preset = event.preset_id || event.preset || "model"
    if (event.event === "sample_started") {
      return `SAMPLE  ${dataset} · ${preset} · ${event.index ?? "-"}/${event.total ?? "?"} · ${event.sample_id ?? ""}`
    }
    if (event.event === "sample_completed") {
      return `SAMPLE  ${dataset} · ${preset} · ${event.index ?? event.completed ?? "-"}/${event.total ?? "?"} · ${(event.status ?? "ok").toUpperCase()} · ${event.sample_id ?? ""}`
    }
    return `EVENT   ${dataset} · ${preset} · ${event.event}${event.phase ? ` · ${event.phase}` : ""}`
  }

  private handleLine(message: string) {
    let event = parseSuiteEvent(message)
    if (!event) {
      this.line(message)
      return
    }
    if (event.event === "task_started") {
      this.activeTask = {
        dataset: String(event.dataset_id || event.dataset || "dataset"),
        preset: String(event.preset_id || event.preset || "model"),
      }
    } else if (
      this.activeTask &&
      !event.dataset_id &&
      event.preset === this.activeTask.preset &&
      !event.event.startsWith("suite_") &&
      !event.event.startsWith("task_")
    ) {
      event = { ...event, dataset_id: this.activeTask.dataset }
    }
    this.progress = applySuiteEvent(this.progress, event)
    this.renderProgress()
    if (["sample_started", "sample_completed", "sample_restored"].includes(event.event)) {
      this.progressBox.scrollTo(this.sampleProgress)
    }
    this.line(this.eventLine(event))
    if (["task_completed", "task_skipped", "suite_interrupted"].includes(event.event)) {
      this.activeTask = null
    }
  }

  private consume(stream: NodeJS.ReadableStream) {
    let buffered = ""
    stream.on("data", (chunk: Buffer) => {
      buffered += chunk.toString()
      const lines = buffered.split(/\r?\n/)
      buffered = lines.pop() ?? ""
      for (const line of lines) if (line) this.handleLine(line)
    })
    return () => {
      if (buffered) this.handleLine(buffered)
      buffered = ""
    }
  }

  private renderProgress() {
    const width = Math.max(70, this.consoleBox?.width ?? this.renderer.root.width - 45)
    if (this.modelProgress) this.modelProgress.content = renderModelProgress(this.progress, width)
    if (this.sampleProgress) this.sampleProgress.content = renderSampleProgress(this.progress, width)
  }

  private metrics() {
    const elapsed = this.started ? Math.floor((Date.now() - this.started) / 1000) : 0
    this.outputState.content = `OUTPUT  ${this.logs.length} lines · ${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")} · ${this.child ? "running" : "finished"}`
  }

  private setStatus(text: string, color: string) {
    this.status.content = text
    this.status.fg = color
  }

  async run() {
    if (this.child) return
    let args: string[]
    try {
      args = buildArgs(this.state, this.presets, this.datasets)
    } catch (error) {
      const message = (error as Error).message
      this.setStatus(message, C.red)
      this.line(`ERR   ${message}`)
      return
    }

    this.started = Date.now()
    this.logs = []
    if (this.state.workflow === "stt-suite") {
      this.progress = seedSuiteProgress(
        selectedDatasets(this.state, this.datasets),
        this.suitePresets,
      )
      this.progress.status = "running"
      this.activeTask = null
      this.renderProgress()
    }
    this.runState.content = "● RUNNING"
    this.runState.fg = C.amber
    this.setStatus("Running…", C.amber)
    const child = spawn(this.bootstrap.python, ["-m", "dominican_eaters", ...args], {
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    })
    this.child = child
    this.line(`RUN   Starting ${this.state.workflow}`)
    this.line(`CMD   ${commandPreview(this.state, this.presets, this.datasets)}`)
    const flushStdout = this.consume(child.stdout)
    const flushStderr = this.consume(child.stderr)
    this.timer = setInterval(() => this.metrics(), 1000)
    child.on("error", (error) => this.line(`ERR   ${error.message}`))
    child.on("close", (code, signal) => {
      flushStdout()
      flushStderr()
      this.child = null
      if (this.timer) clearInterval(this.timer)
      const ok = code === 0
      const stopped = signal !== null
      if (this.state.workflow === "stt-suite" && !ok && this.progress.status === "running") {
        this.progress.status = stopped ? "interrupted" : "failed"
        this.renderProgress()
      }
      const outcome = ok
        ? "OK    Completed successfully"
        : stopped
          ? `STOP  Process stopped by ${signal}`
          : `ERR   Process exited with status ${code}`
      this.line(outcome)
      this.setStatus(
        ok ? "Completed successfully" : stopped ? "Stopped; run again to resume" : `Exited with status ${code}`,
        ok ? C.green : stopped ? C.amber : C.red,
      )
      this.runState.content = "● READY"
      this.runState.fg = C.green
      this.metrics()
    })
  }

  async quit() {
    if (this.child) {
      this.child.kill("SIGTERM")
      await new Promise((resolve) => setTimeout(resolve, 100))
    }
    this.renderer.destroy()
  }
}

export async function main() {
  const bootstrap = JSON.parse(process.env.DOMINICAN_EATERS_TUI_DATA ?? "") as Bootstrap
  const renderer = await createCliRenderer({
    exitOnCtrlC: false,
    targetFps: 30,
    backgroundColor: C.bg,
  })
  new DominicanEatersTui(renderer, bootstrap)
}

if (import.meta.url === `file://${process.argv[1]}`) await main()
