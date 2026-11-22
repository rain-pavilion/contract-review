@echo off
setlocal
if not exist "%~dp0zhilv_qiling_v2\models\bert_mamba_entail\model.pt" (
  echo The trained BERT-Mamba checkpoint is missing. See docs for model preparation.
  exit /b 1
)
set "ZLQL_ENTAILMENT_MODEL_DIR=%~dp0zhilv_qiling_v2\models\bert_mamba_entail"
set "ZLQL_ENTAILMENT_BACKEND=bert"
set "ZLQL_ENTAILMENT_DEVICE=cpu"
set "ZLQL_MULTI_AGENT=1"
set "ZLQL_FREEZE_FEEDBACK_CYCLE=1"
set "ZLQL_LLM_FEEDBACK_WINDOW_POLICY=cover_all_context_v1"
set "ZLQL_LLM_INITIAL_SOURCE_PROTOCOL=initial-exact-source-selection-v1"
set "ZLQL_LLM_MAX_TOKENS=4096"
set "ZLQL_LLM_NATIVE_THINKING=off"
set "ZLQL_LLM_NATIVE_REASONING_FORMAT=deepseek"
echo Core mode: real LegalOne-8B and explicitly selected BERT-Mamba.
echo BERT-Mamba uses synthetic weak supervision, not human legal gold labels.
call "%~dp0start-model-web.cmd" %*
endlocal
