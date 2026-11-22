@echo off
setlocal
rem Isolated defense workspace. Existing historical databases and CPU services are retained.
set "ZLQL_STORE_PATH=%~dp0zhilv_qiling_v2\reports\defense_workspace_20261121.sqlite"
set "ZLQL_MODEL_AUDIT_PATH=%~dp0zhilv_qiling_v2\reports\defense_model_calls_20261121.sqlite"
set "ZLQL_ENTAILMENT_MODEL_DIR=%~dp0zhilv_qiling_v2\models\bert_mamba_entail"
set "ZLQL_ENTAILMENT_BACKEND=bert"
set "ZLQL_ENTAILMENT_DEVICE=cpu"
set "ZLQL_BERT_DEVICE=cpu"
set "ZLQL_MULTI_AGENT=1"
set "ZLQL_FREEZE_FEEDBACK_CYCLE=1"
set "ZLQL_LLM_FEEDBACK_WINDOW_POLICY=cover_all_context_v1"
set "ZLQL_LLM_INITIAL_SOURCE_PROTOCOL=initial-exact-source-selection-v1"
set "ZLQL_LLM_INITIAL_MAX_ATTEMPTS=1"
set "ZLQL_LLM_MAX_CALLS=2"
set "ZLQL_LLM_CHUNK_CHARS=6000"
set "ZLQL_LLM_CHUNK_OVERLAP=0"
set "ZLQL_LLM_CROSS_CLAUSE_PAIRS=0"
set "ZLQL_REVIEW_MAX_ROUNDS=2"
set "ZLQL_LLM_MAX_TOKENS=2048"
set "ZLQL_LLM_NATIVE_THINKING=off"
set "ZLQL_LLM_NATIVE_REASONING_FORMAT=deepseek"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "DEFENSE_LLAMA=%~dp0.model-runtime\llama.cpp\.cuda\b12375-win-cuda13.5-x64\runtime\llama-server.exe"
if not exist "%DEFENSE_LLAMA%" (
  echo Required local CUDA runtime is missing. Historical CPU launcher remains available.
  exit /b 1
)
echo Defense mode: LegalOne-8B GPU, BERT-Mamba CPU, bounded review, separate archive.
echo Pending items remain pending. Call limits are not a claim of complete review.
echo The verified model service stays resident and is reused on the next launch.
call "%~dp0start-web.cmd" --with-model --keep-model-running --port 8618 --model-port 8635 --model-backend llama.cpp --llama-server "%DEFENSE_LLAMA%" --model-gpu-layers 99 --model-cache-type q8_0 --model-context-tokens 16384 --cpu-threads 6 %*
endlocal
