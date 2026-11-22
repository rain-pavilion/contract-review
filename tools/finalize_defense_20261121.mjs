import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
const root='C:/Users/ROG/OneDrive/Desktop/LLM';
const skill='C:/Users/ROG/.codex/plugins/cache/openai-primary-runtime/presentations/26.1118.11504/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const [candidatePath,tableSlides]=process.argv.slice(2);
if(!candidatePath)throw new Error('Candidate path required');
const nativeTables=tableSlides?tableSlides.split(',').map(Number):[];
const output=root+'/outputs/智律契灵PPT_20261121';
await fs.mkdir(output,{recursive:true});
await fs.mkdir(root+'/tools/ppt_defense_20261121/build',{recursive:true});
const result=await finalizePresentation({
 workspaceDir:root,candidatePath,
 finalPath:output+'/智律契灵_答辩展示版_主讲与备答.pptx',
 explicitTotalSlideCount:26,
 requiredNativeTableOwnerSlides:nativeTables,
 requiredNativeChartOwnerSlides:[],
 pythonExecutable:'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1118-904-0_26.1118.904.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...nativeTables.flatMap(n=>['--require-native-table-slide',String(n)])],
 verifyArtifactToolImport:true,
 receiptPath:root+'/tools/ppt_defense_20261121/build/final_validation_complete.json',
});
console.log(JSON.stringify(result));
