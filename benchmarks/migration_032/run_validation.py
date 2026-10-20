import subprocess,sys,json,platform,importlib.metadata as md
from pathlib import Path
root=Path('benchmarks/migration_032');root.mkdir(parents=True,exist_ok=True)
groups={
'edge_windows': ['tests/edge','tests/windows','tests/config/test_talker_only_pipeline.py','tests/config/test_vllm_native_pipelines.py','tests/config/test_deploy_profiles.py','tests/config/test_import_guard.py','tests/engine/test_init_timeline.py','tests/engine/test_parallel_stage_init.py','tests/platforms/test_cpu_platform.py','tests/worker/test_cpu_ar_model_runner.py','tests/worker/test_forward_kwargs_filtering.py','--ignore=tests/edge/local/test_e2e_local_text.py','--ignore=tests/edge/local/test_e2e_serve_text.py'],
'predictor_cpu':['tests/model_executor/models/common/test_code_predictor_incremental.py'],
'speech_gpu':['tests/entrypoints/openai_api/test_serving_speech_stream.py','tests/entrypoints/openai_api/test_codec_stream_protocol.py','tests/model_executor/models/qwen3_tts/test_code_predictor_sampling.py','tests/model_executor/models/qwen3_tts/test_cached_code_predictor.py'],
}
report={'environment':{'platform':platform.platform(),'python':sys.version,'packages':{x:md.version(x) for x in ['vllm','vllm-omni','torch','transformers','pytest']}},'groups':[]}
for name,paths in groups.items():
 cmd=[sys.executable,'-m','pytest','-o','addopts=',*paths,'-q','--tb=short','--maxfail=5']
 with (root/(name+'.log')).open('w') as f:
  p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 text=(root/(name+'.log')).read_text();print(name,p.returncode,text.splitlines()[-1],flush=True)
 report['groups'].append({'name':name,'command':cmd,'exit_code':p.returncode,'summary':text.splitlines()[-1]})
 (root/'results.json').write_text(json.dumps(report,indent=2)+'\n')