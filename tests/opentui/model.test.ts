import assert from "node:assert/strict"
import test from "node:test"
import {
  applySuiteEvent,
  buildArgs,
  changeWorkflow,
  commandPreview,
  initialState,
  initialSuiteProgress,
  parseSuiteEvent,
  renderModelProgress,
  renderSampleProgress,
  seedSuiteProgress,
  type Dataset,
  type Preset,
} from "../../src/dominican_eaters/tui/opentui/model.ts"
const presets:Preset[]=[{id:"whisper-large-v3",label:"Whisper Large v3",status:"current",reason:"Ready",runnable:true,execution:"inline",precision:"fp16",devices:["cpu","cuda"],workerPython:""},{id:"worker",label:"Worker",status:"current",reason:"Ready",runnable:true,execution:"worker",precision:"fp16",devices:["cuda"],workerPython:"/venv/python"}]
const datasets:Dataset[]=[
  {id:"music",label:"Music",manifest:"data/manifests/stt-all.json",sampleCount:558},
  {id:"speech",label:"Speech",manifest:"data/manifests/speech.json",sampleCount:12},
]
test("default preview",()=>assert.equal(commandPreview(initialState(),presets),"dominican-eaters config validate config/default.yaml"))
test("startup state selects a workflow and keeps valid form overrides",()=>{
 const s=initialState({workflow:"stt-suite",dataset:"music",output:"artifacts/stt/custom",timestamps:true})
 assert.equal(s.workflow,"stt-suite")
 assert.equal(s.source,"")
 assert.equal(s.output,"artifacts/stt/custom")
 assert.equal(s.dataset,"music")
 assert.equal(s.timestamps,true)
 const malformed=initialState({warmupRuns:42} as unknown as Partial<ReturnType<typeof initialState>>)
 assert.equal(malformed.warmupRuns,"1")
})
test("domain defaults",()=>{const s=initialState();changeWorkflow(s,"books-run");assert.equal(s.source,"data/manifests/books.json");assert.equal(s.output,"artifacts/books");assert.deepEqual(buildArgs(s,presets),["collect","books","run","data/manifests/books.json","--output-dir","artifacts/books"])})
test("benchmark controls",()=>{const s=initialState();changeWorkflow(s,"stt-benchmark");s.verifyHashes=true;s.timestamps=true;assert.match(commandPreview(s,presets),/--timestamps$/)})
test("worker requires interpreter",()=>{const s=initialState();changeWorkflow(s,"stt-benchmark");s.preset="worker";assert.throws(()=>buildArgs(s,presets),/isolated worker Python/)})
test("worker preflight requires interpreter",()=>{const s=initialState();changeWorkflow(s,"stt-preflight");s.preset="worker";assert.throws(()=>buildArgs(s,presets),/isolated worker Python/)})
test("suite selects every configured dataset",()=>{
 const s=initialState();changeWorkflow(s,"stt-suite")
 assert.deepEqual(buildArgs(s,presets,datasets,["whisper-large-v3","worker"]),[
  "stt","suite","--dataset","music=data/manifests/stt-all.json","--dataset","speech=data/manifests/speech.json",
  "--preset","whisper-large-v3","--preset","worker",
  "--output-dir","artifacts/stt-suite","--device","cuda","--precision","fp16","--warmup-runs","1","--request-timeout","1800","--verify-hashes",
 ])
 s.dataset="speech"
 assert.equal(buildArgs(s,presets,datasets,["whisper-large-v3"]).filter(value=>value.includes("data/manifests")).length,1)
})
test("suite blocks before a long run when a required worker is missing",()=>{
 const s=initialState();changeWorkflow(s,"stt-suite")
 const missing:Preset={id:"qwen3-asr-1.7b",label:"Qwen3 ASR",status:"candidate",reason:"Ready",runnable:true,execution:"worker",precision:"fp16",devices:["cuda"],workerPython:""}
 assert.throws(()=>buildArgs(s,[presets[0]!,missing],datasets,[missing.id]),/Missing worker environments: qwen3-asr-1.7b/)
 assert.doesNotThrow(()=>buildArgs(s,[presets[0]!,missing],datasets,["whisper-large-v3"]))
 const progress=seedSuiteProgress([datasets[0]!],[missing])
 assert.equal(progress.models.get("music::qwen3-asr-1.7b")?.status,"missing-worker")
})
test("structured progress events update model and sample rows",()=>{
 let progress=seedSuiteProgress([datasets[0]!],[presets[0]!])
 const started=parseSuiteEvent('stt_event={"event":"sample_started","dataset":"music","preset":"whisper-large-v3","phase":"transcribing","index":4,"total":558,"sample_id":"song-004"}')
 assert.ok(started)
 progress=applySuiteEvent(progress,started)
 assert.equal(progress.models.get("music::whisper-large-v3")?.activeSample,"song-004")
 const completed=parseSuiteEvent('prefix stt_event={"event":"sample_completed","dataset":"music","preset":"whisper-large-v3","phase":"transcribing","completed":4,"succeeded":3,"failed":1,"index":4,"total":558,"sample_id":"song-004","status":"failed","elapsed_seconds":2.5,"message":"decode failed"}')
 assert.ok(completed)
 progress=applySuiteEvent(progress,completed)
 assert.equal(progress.models.get("music::whisper-large-v3")?.failed,1)
 assert.match(renderModelProgress(progress),/4\/558/)
 assert.match(renderSampleProgress(progress),/song-004/)
 assert.match(renderSampleProgress(progress),/whisper-large-v3/)
 assert.match(renderSampleProgress(progress),/decode failed/)
})
test("sample progress keeps every dataset row instead of a recent-only window",()=>{
 let progress=seedSuiteProgress([datasets[0]!],[presets[0]!])
 for(let index=1;index<=12;index++){
  progress=applySuiteEvent(progress,{event:"sample_completed",dataset_id:"music",preset:"whisper-large-v3",index,total:558,sample_id:`song-${String(index).padStart(3,"0")}`,status:"ok"})
 }
 assert.equal(progress.samples.size,12)
 const table=renderSampleProgress(progress)
 assert.match(table,/song-001/)
 assert.match(table,/song-012/)
})
test("event parser ignores ordinary and malformed log lines",()=>{
 assert.equal(parseSuiteEvent("loading model"),null)
 assert.equal(parseSuiteEvent("stt_event={bad"),null)
 assert.equal(initialSuiteProgress().status,"waiting")
})
test("suite task boundaries preserve per-sample counts",()=>{
 let progress=seedSuiteProgress([datasets[0]!],[presets[0]!])
 progress=applySuiteEvent(progress,{event:"task_started",dataset_id:"music",preset:"whisper-large-v3",completed:2,total:0,task_status:"running"})
 assert.equal(progress.models.get("music::whisper-large-v3")?.completed,0)
 progress=applySuiteEvent(progress,{event:"task_completed",dataset_id:"music",preset:"whisper-large-v3",succeeded:550,failed:8,total:558,task_status:"partial"})
 const model=progress.models.get("music::whisper-large-v3")
 assert.equal(model?.completed,558)
 assert.equal(model?.status,"partial")
})
