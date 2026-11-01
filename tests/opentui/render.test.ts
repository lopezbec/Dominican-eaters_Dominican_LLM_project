import assert from "node:assert/strict"
import test from "node:test"
import {createTestRenderer} from "@opentui/core/testing"
import {DominicanEatersTui} from "../../src/dominican_eaters/tui/opentui/app.ts"
const bootstrap={python:process.execPath,presets:[{id:"whisper-large-v3",label:"Whisper Large v3",status:"current",reason:"Ready",runnable:true,execution:"inline" as const,precision:"fp16",devices:["cpu","cuda"],workerPython:""}],datasets:[{id:"music",label:"Music / collected lyrics",manifest:"data/manifests/stt-all.json",sampleCount:558}],suitePresets:["whisper-large-v3"]}
test("desktop and compact rendering",async()=>{const s=await createTestRenderer({width:110,height:40});new DominicanEatersTui(s.renderer,bootstrap);await s.renderOnce();let f=s.captureCharFrame();assert.match(f,/DOMINICAN EATERS/);assert.match(f,/dominican-eaters config validate/);assert.match(f,/config\/default.yaml/);assert.match(f,/Workflow/);s.resize(72,28);await s.renderOnce();f=s.captureCharFrame();assert.match(f,/RUN CONFIGURATION/);assert.match(f,/Workflow/);s.renderer.destroy()})
test("suite rendering keeps dataset and progress visible",async()=>{
 const screen=await createTestRenderer({width:140,height:44})
 const tui=new DominicanEatersTui(screen.renderer,bootstrap)
 tui.state.workflow="stt-suite"
 tui.state.output="artifacts/stt-suite"
 ;(tui as any).sync()
 ;(tui as any).handleLine('stt_event={"event":"sample_completed","dataset_id":"music","preset":"whisper-large-v3","index":1,"total":558,"sample_id":"song-001","status":"ok"}')
 await screen.renderOnce()
 const frame=screen.captureCharFrame()
 assert.match(frame,/All datasets \(1\)/)
 assert.match(frame,/PIPELINE PROGRESS/)
 assert.match(frame,/music \/ whisper-large/)
 assert.match(frame,/song-001/)
 assert.match(frame,/dominican-eaters stt suite/)
 const preview=(tui as any).preview.content.chunks.map((chunk:{text:string})=>chunk.text).join("")
 assert.match(preview,/--preset whisper-large-v3/)
 screen.renderer.destroy()
})
test("bootstrap selects the lyrics suite and auto-runs through the startup hook",async()=>{
 const screen=await createTestRenderer({width:140,height:44})
 const tui=new DominicanEatersTui(screen.renderer,{
  ...bootstrap,
  initialState:{workflow:"stt-suite",dataset:"music",output:"artifacts/stt-suite/t4-all-models"},
  autoRun:true,
 })
 let runs=0
 tui.run=async()=>{runs+=1}
 await tui.start()
 await tui.start()
 await screen.renderOnce()
 const frame=screen.captureCharFrame()
 assert.equal(runs,1)
 assert.equal(tui.state.workflow,"stt-suite")
 assert.equal(tui.state.dataset,"music")
 assert.equal(tui.state.output,"artifacts/stt-suite/t4-all-models")
 assert.match(frame,/Music \/ collected lyrics/)
 assert.match(frame,/dominican-eaters stt suite/)
 screen.renderer.destroy()
})
