"""Read-only diagnostic analysis of independently audited, closed Agent runs."""
import argparse,hashlib,json,math
from pathlib import Path

def analyze(run,audit_path):
 receipt_bytes=(run/'receipt.json').read_bytes();receipt=json.loads(receipt_bytes)
 audit_bytes=audit_path.read_bytes();audit=json.loads(audit_bytes)
 if receipt['phase']!='completed' or audit['status']!='passed' or audit['receipt_sha256']!=hashlib.sha256(receipt_bytes).hexdigest():
  raise ValueError('Completed run and matching independent audit required')
 def read(name):
  data=(run/name).read_bytes();metadata=receipt['raw_files'][name]
  if len(data)!=metadata['size_bytes'] or hashlib.sha256(data).hexdigest()!=metadata['sha256']:raise ValueError('Raw source changed: '+name)
  return json.loads(data)
 samples=read('profile-samples.json');calls=read('all-raw.json')['calls'];rows=[]
 for sample in samples:
  selected=calls[sample['call_start']:sample['call_end']]
  if len(selected)!=2 or sample['status']!='passed':raise ValueError('Incomplete task')
  phases=[]
  for call in selected:
   m=call['terminal_metrics']['backend_metrics'];t=m['runtime_telemetry'];im=t.get('image_observation')
   phases.append(dict(role='image_answer' if call['image_input'] is not None else 'planner',caller_wall_s=call['wall_s'],first_model_output_s=m['first_token_s'],native_prefill_ms=m['runtime_timings']['prompt_ms'],native_decode_ms=m['runtime_timings']['predicted_ms'],native_encoder_ms=im['encode_result']['native_encoder_ms'] if im else None,usage=m['usage'],decode_logical_read_bytes=t.get('logical_file_read_bytes'),logical_read_scope=t.get('logical_file_read_scope'),decode_expert_cache_hit_ratio=t.get('hit_rate')))
  if [p['role'] for p in phases]!=['planner','image_answer']:raise ValueError('Unexpected stage roles')
  rows.append(dict(index=sample['index'],phase=sample['phase'],length=sample['length'],whole_agent_task_s=sample['whole_agent_task_s_excludes_load'],first_visible_output_s=sample['first_visible_output_s'],calls=phases))
 summaries=[]
 def quantile(values,p):
  if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values):raise ValueError('Invalid timing')
  return sorted(values)[math.ceil(len(values)*p)-1]
 if receipt['profile_mode']=='full':
  for length in ('short','medium','long'):
   selected=[r for r in rows if r['phase']=='measured' and r['length']==length]
   if len(selected)<20:raise ValueError('Insufficient measured samples')
   groups={'whole_agent_task_s':[r['whole_agent_task_s'] for r in selected],'first_visible_output_s':[r['first_visible_output_s'] for r in selected]}
   for index,role in enumerate(('planner','image_answer')):
    for metric in ('caller_wall_s','native_prefill_ms','native_decode_ms'):
     groups[role+'_'+metric]=[r['calls'][index][metric] for r in selected]
   groups['native_encoder_ms']=[r['calls'][1]['native_encoder_ms'] for r in selected]
   summaries.append(dict(length=length,n=len(selected),metrics={k:dict(p50=quantile(v,.5),p95=quantile(v,.95)) for k,v in groups.items()}))
 return dict(schema='omni-strata-audited-agent-phase-analysis-v1',mode=receipt['profile_mode'],receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),audit_sha256=hashlib.sha256(audit_bytes).hexdigest(),raw_sources={n:receipt['raw_files'][n] for n in ('all-raw.json','profile-samples.json')},batch_size=1,concurrency=1,cold_load_and_verification_s=receipt['cold_load_and_verification_s'],samples=rows,measured_only_statistics=summaries,quantile_method='nearest rank',qualification_created=False,limits=['Phase timer percentiles cannot be summed to form whole-request latency.','Logical IO and cache hit ratios are decode-scoped; physical SSD attribution is unknown.','Cold load includes verification; this is not raw model load time.','Only the audited task family is covered; no paired speedup or broad Agent qualification is inferred.'])
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 result=analyze(args.run,args.audit)
 with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
 print(json.dumps(dict(mode=result['mode'],requests=len(result['samples']),measured_groups=len(result['measured_only_statistics']))))
