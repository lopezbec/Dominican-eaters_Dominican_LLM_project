import assert from "node:assert/strict"
import test from "node:test"
import {createTestRenderer} from "@opentui/core/testing"
import {DominicanEatersTui} from "../../src/dominican_eaters/tui/opentui/app.ts"
const bootstrap={python:process.execPath,presets:[{id:"whisper-base",label:"Whisper Base",status:"current",reason:"Ready",runnable:true,execution:"inline" as const,precision:"fp16",devices:["cpu","cuda"],workerPython:""}]}
test("desktop and compact rendering",async()=>{const s=await createTestRenderer({width:110,height:40});new DominicanEatersTui(s.renderer,bootstrap);await s.renderOnce();let f=s.captureCharFrame();assert.match(f,/DOMINICAN EATERS/);assert.match(f,/dominican-eaters config validate/);assert.match(f,/config\/default.yaml/);s.resize(72,28);await s.renderOnce();f=s.captureCharFrame();assert.match(f,/RUN CONFIGURATION/);s.renderer.destroy()})
