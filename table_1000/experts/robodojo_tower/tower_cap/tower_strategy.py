"""S2 supplies roles/goals/supports; S1 computes and validates all executable paths."""
import json
from pathlib import Path
from .scene_adapter import build_runtime,structure_plan
from .skills import pick_place_block,place_board_bimanual

def run(env,emit,last,out):
 rt,entry=build_runtime(env,emit,last,out);plan=structure_plan(rt,entry);results=[]
 for i,item in enumerate(plan):
  if item['preserve']:
   rt.record(stage='preserve_completed_support',object=item['object'],actual=rt.snapshot());continue
  # An inactive hand must stay clear of placed supports during later physical actions.
  return_home=True
  if len(item['arms'])==2:result=place_board_bimanual(rt,item['object'],item['arms'],item['goal'],item['supports'],return_home=return_home)
  else:result=pick_place_block(rt,item['object'],item['arms'][0],item['goal'],item['supports'],allow_table=not item['supports'],goal_yaw_symmetries=([0.,90.,-90.,180.] if item['role'] in ('base_support','middle_support') else ([0.,180.] if item['role'] in ('roof','cap') else None)),return_home=return_home,task_upright_yaw_free=item['role'] in ('base_support','middle_support'))
  results.append(result);Path(out,'tower_plan_progress.json').write_text(json.dumps(dict(completed_atoms=results,next_index=i+1,whole_tower_complete=False),default=lambda x:x.tolist(),indent=2))
  if result['status']!='completed':return dict(status='S1_stopped',reason=result.get('reason'),stage=result['stage'],results=results,native_steps=int(env.take_action_cnt[0]))
 # No new acceptance rule: ask the unchanged task to advance its own final terminal.
 for _ in range(min(30,int(env.step_lim)-int(env.take_action_cnt[0]))):
  if env.end_flag[0]:break
  rt.motion.send('tower_native_final_observation')
 return dict(results=results,native_end_flag=bool(env.end_flag[0]),native_success=bool(env.success[0]))
