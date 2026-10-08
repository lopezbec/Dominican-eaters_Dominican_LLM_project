import assert from "node:assert/strict"
import test from "node:test"
import {buildArgs,changeWorkflow,commandPreview,initialState,type Preset} from "../../src/dominican_eaters/tui/opentui/model.ts"
const presets:Preset[]=[{id:"whisper-base",label:"Whisper Base",status:"current",reason:"Ready",runnable:true,execution:"inline",precision:"fp16",devices:["cpu","cuda"],workerPython:""},{id:"worker",label:"Worker",status:"current",reason:"Ready",runnable:true,execution:"worker",precision:"fp16",devices:["cuda"],workerPython:"/venv/python"}]
test("default preview",()=>assert.equal(commandPreview(initialState(),presets),"dominican-eaters config validate config/default.yaml"))
test("domain defaults",()=>{const s=initialState();changeWorkflow(s,"books-run");assert.equal(s.source,"data/manifests/books.json");assert.equal(s.output,"artifacts/books");assert.deepEqual(buildArgs(s,presets),["collect","books","run","data/manifests/books.json","--output-dir","artifacts/books"])})
test("benchmark controls",()=>{const s=initialState();changeWorkflow(s,"stt-benchmark");s.verifyHashes=true;s.timestamps=true;assert.match(commandPreview(s,presets),/--timestamps$/)})
test("worker requires interpreter",()=>{const s=initialState();changeWorkflow(s,"stt-benchmark");s.preset="worker";assert.throws(()=>buildArgs(s,presets),/isolated worker Python/)})
