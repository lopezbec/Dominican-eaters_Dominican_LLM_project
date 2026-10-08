export type Workflow = "config-validate"|"books-preflight"|"books-run"|"lyrics-preflight"|"lyrics-run"|"lyrics-download-audio"|"poems-preflight"|"poems-run"|"stt-preflight"|"stt-manifest-build"|"stt-benchmark"
export interface Preset { id:string; label:string; status:string; reason:string; runnable:boolean; execution:"inline"|"worker"; precision:string; devices:string[]; workerPython:string }
export interface Bootstrap { python:string; presets:Preset[] }
export const workflows:{name:string,value:Workflow}[] = [
 ["Validate configuration","config-validate"],["Preflight books manifest","books-preflight"],["Collect books","books-run"],
 ["Preflight lyrics manifest","lyrics-preflight"],["Collect lyrics","lyrics-run"],["Download collected lyrics audio","lyrics-download-audio"],
 ["Preflight poems manifest","poems-preflight"],["Collect poems","poems-run"],["Preflight speech-to-text manifest","stt-preflight"],
 ["Build speech-to-text manifest","stt-manifest-build"],["Run speech-to-text benchmark","stt-benchmark"],
].map(([name,value])=>({name,value:value as Workflow}))
export const sources:Record<Workflow,string>={
 "config-validate":"config/default.yaml","books-preflight":"data/manifests/books.json","books-run":"data/manifests/books.json",
 "lyrics-preflight":"data/manifests/lyrics.json","lyrics-run":"data/manifests/lyrics.json","lyrics-download-audio":"artifacts/lyrics/lyrics-collection.json",
 "poems-preflight":"data/manifests/poems.json","poems-run":"data/manifests/poems.json","stt-preflight":"data/manifests/stt.json",
 "stt-manifest-build":"data/audio","stt-benchmark":"data/manifests/stt.json"}
export const outputs:Partial<Record<Workflow,string>>={"books-run":"artifacts/books","lyrics-run":"artifacts/lyrics","lyrics-download-audio":"data/audio/lyrics","poems-run":"artifacts/poems","stt-manifest-build":"data/manifests/stt-all.json","stt-benchmark":"artifacts/stt-run"}
export interface FormState { workflow:Workflow; source:string; output:string; dataRoot:string; artifactsRoot:string; preset:string; device:string; precision:string; workerPython:string; warmupRuns:string; requestTimeout:string; shortAudioPolicy:string; minimumAudioSeconds:string; force:boolean; verifyHashes:boolean; timestamps:boolean }
export const initialState=():FormState=>({workflow:"config-validate",source:sources["config-validate"],output:"",dataRoot:"",artifactsRoot:"",preset:"whisper-base",device:"cuda",precision:"fp16",workerPython:"",warmupRuns:"1",requestTimeout:"300",shortAudioPolicy:"reject",minimumAudioSeconds:"0.1",force:false,verifyHashes:false,timestamps:false})
export const hasOutput=(w:Workflow)=>outputs[w]!==undefined
export const isStt=(w:Workflow)=>w==="stt-preflight"||w==="stt-benchmark"
export function changeWorkflow(s:FormState,w:Workflow){const oldSource=sources[s.workflow],oldOutput=outputs[s.workflow]??"";if(!s.source.trim()||s.source===oldSource)s.source=sources[w];if(!s.output.trim()||s.output===oldOutput)s.output=outputs[w]??"";s.workflow=w}
const need=(v:string,m:string)=>{v=v.trim();if(!v)throw Error(m);return v}
const positive=(v:string,l:string)=>{if(!Number.isFinite(Number(v))||Number(v)<=0)throw Error(`${l} must be a positive number.`);return v.trim()}
const quote=(v:string)=>/^[\w@%+=:,./-]+$/.test(v)?v:`'${v.replaceAll("'","'\\''")}'`
export function buildArgs(s:FormState,presets:Preset[]):string[]{
 const source=need(s.source,s.workflow==="config-validate"?"Select a configuration file.":"Select a manifest file.")
 if(s.workflow==="config-validate"){const a=["config","validate",source];if(s.dataRoot.trim())a.push("--data-root",s.dataRoot.trim());if(s.artifactsRoot.trim())a.push("--artifacts-root",s.artifactsRoot.trim());return a}
 const domain:Partial<Record<Workflow,string[]>>={"books-preflight":["collect","books","preflight"],"books-run":["collect","books","run"],"lyrics-preflight":["collect","lyrics","preflight"],"lyrics-run":["collect","lyrics","run"],"poems-preflight":["collect","poems","preflight"],"poems-run":["collect","poems","run"]}
 if(domain[s.workflow]){const a=[...domain[s.workflow]!,source];if(["books-run","lyrics-run","poems-run"].includes(s.workflow)){a.push("--output-dir",need(s.output,"Select an output directory."));if(s.force)a.push("--force")}return a}
 if(s.workflow==="lyrics-download-audio"){const a=["collect","lyrics","download-audio",source,"--output-dir",need(s.output,"Select an output directory.")];if(s.force)a.push("--force");return a}
 if(s.workflow==="stt-manifest-build")return["stt","manifest","build",source,"--output-file",need(s.output,"Select an output directory.")]
 const p=presets.find(x=>x.id===s.preset);if(!p)throw Error(`Unsupported preset: ${s.preset}`)
 if(s.workflow==="stt-benchmark"&&!p.runnable)throw Error(`Preset ${p.id} is ${p.status} and cannot be benchmarked: ${p.reason}`)
 if(p.execution==="worker"&&!s.workerPython.trim())throw Error("Select the isolated worker Python executable.")
 const a=["stt",s.workflow==="stt-preflight"?"preflight":"benchmark",source]
 if(s.workflow==="stt-preflight"&&s.dataRoot.trim())a.push("--dataset-root",s.dataRoot.trim());if(s.workflow==="stt-benchmark")a.push("--output-dir",need(s.output,"Select an output directory."))
 a.push("--preset",p.id,"--device",s.device,"--precision",s.precision)
 if(s.workflow==="stt-benchmark"){if(!/^\d+$/.test(s.warmupRuns.trim()))throw Error("Warmup runs must be a nonnegative integer.");a.push("--warmup-runs",s.warmupRuns.trim(),"--request-timeout",positive(s.requestTimeout,"Request timeout"),"--short-audio-policy",s.shortAudioPolicy,"--minimum-audio-seconds",positive(s.minimumAudioSeconds,"Minimum audio duration"))}
 if(s.workerPython.trim())a.push("--worker-python",s.workerPython.trim());if(s.verifyHashes)a.push("--verify-hashes");if(s.timestamps&&s.workflow==="stt-benchmark")a.push("--timestamps");return a
}
export function commandPreview(s:FormState,p:Preset[]){try{return["dominican-eaters",...buildArgs(s,p)].map(quote).join(" ")}catch(e){return`Complete the form: ${(e as Error).message}`}}
