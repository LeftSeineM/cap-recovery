"""Prospective S1 call records. Offline observations are never physical successes."""
import json,hashlib,time,uuid
from pathlib import Path
VERSION='tower-s1-v0.9.2'
def source_hash(root):
 files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(root).glob('*.py'))}
 return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),files
class SkillInapplicable(RuntimeError):pass
class SkillCall:
 def __init__(self,out,skill,params,env=None,mode='physical',source_root=None):
  self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.env=env;self.start=self.steps();self.started=time.time();sha,files=source_hash(source_root or Path(__file__).parent)
  self.row=dict(call_id=str(uuid.uuid4()),skill=skill,skill_version=VERSION,source_sha256=sha,source_files=files,parameters=params,mode=mode,started_at=self.started,status='started',stage='applicability',actions_used=0,remaining_budget=self.budget(),actual_state=None,evidence=[])
  self.write('started')
 def steps(self):return int(self.env.take_action_cnt[0]) if self.env is not None else 0
 def budget(self):return int(self.env.step_lim)-self.steps() if self.env is not None else None
 def write(self,event):
  with (self.out/'skill_calls.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(self.row,event=event),ensure_ascii=False,default=lambda x:x.tolist())+'\n')
 def finish(self,status,stage,reason=None,actual_state=None,evidence=None):
  self.row.update(status=status,stage=stage,reason=reason,actual_state=actual_state,evidence=evidence if evidence is not None else self.row['evidence'],actions_used=self.steps()-self.start,remaining_budget=self.budget(),elapsed_seconds=time.time()-self.started,finished_at=time.time())
  self.write('finished');return self.row
 def invoke(self,operation):
  try:
   result=operation()
   if isinstance(result,dict) and 'actual_state' in result:
    self.row['postconditions']={k:v for k,v in result.items() if k!='actual_state'};actual=result['actual_state']
   else:actual=result
   return self.finish('completed','postconditions',actual_state=actual)
  except SkillInapplicable as e:return self.finish('inapplicable',self.row['stage'],str(e),actual_state=self.observe() if hasattr(self,'observe') else None)
  except Exception as e:
   # Native-success control exceptions must propagate after recording the true state.
   if self.env is not None and bool(self.env.end_flag[0]) and bool(self.env.success[0]):
    self.finish('native_success','native_terminal',str(e));raise
   return self.finish('execution_failed',self.row['stage'],repr(e),actual_state=self.observe() if hasattr(self,'observe') else None)
