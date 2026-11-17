"""Finite candidate only; invoke after same-job three-target push. No reset, staging or resource work."""
from pathlib import Path
import json
from .json_safety import plain
from .bonus_pickup import run_bonus
from .bonus_handover import run_handover
from .place import run_place
from .grip_continuity import read_same_job_hold_grips
from .rightmost_target import runtime_target
def run(env,emit_action,out):
 out=Path(out);identity=json.loads((out/'identity.json').read_text(encoding='utf-8'))
 if not identity.get('case','').startswith('make_kong__'):raise ValueError('same-job s0_l21 identity required')
 read_same_job_hold_grips(env,out,identity['job'])
 def save(name,value):(out/name).write_text(json.dumps(plain(value),ensure_ascii=False,indent=2),encoding='utf-8')
 begin=int(env.take_action_cnt[0]);result={'status':'started','counts_as_success':False,'native_begin':begin,'native_budget':600}
 save('bonus80_entry_target_diagnostic.json',runtime_target(env))
 pickup=run_bonus(env,emit_action,out);result['pickup']=pickup
 if pickup.get('status')!='bonus_pickup_lift_observed' or pickup.get('pickup_reference_certified_by_lift') is not True:
  result['status']='pickup_not_certified';save('bonus80_result.json',result);return result
 transfer=run_handover(env,emit_action,out,pickup_subdir='mahjong_bonus_pickup');result['handover']=transfer
 if transfer.get('status')!='right_only_physical_carry_observed' or transfer.get('right_only_physical_carry_verified') is not True:
  result['status']='handover_not_certified';save('bonus80_result.json',result);return result
 folder=out/'mahjong_bonus_handover'
 ref=json.loads((folder/'receiver_reference_certified.json').read_text(encoding='utf-8'))
 physical=json.loads((folder/'physical_transfer_verification.json').read_text(encoding='utf-8'))
 if ref.get('status')!='right_only_physical_carry_observed' or ref.get('reset_allowed') is not False or physical.get('right_only_carry_verified') is not True or ref.get('verified_native_step')!=int(env.take_action_cnt[0]):
  result['status']='receiver_certificate_invalid';save('bonus80_result.json',result);return result
 # run_place re-reads runtime_target immediately before planning; entry diagnostic never binds the goal.
 placement=run_place(env,emit_action,out,ref,identity['job']);result['placement']=placement
 result.update(status='bonus80_sequence_returned',native_end=int(env.take_action_cnt[0]),complete_case_claim=False)
 save('bonus80_result.json',result);return result
