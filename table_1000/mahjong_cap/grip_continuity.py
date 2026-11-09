"""Read the last completed same-job native command; never infer hold from sensors."""
from pathlib import Path
import json,math,hashlib

def read_same_job_hold_grips(env, out, expected_job_id):
    out=Path(out)
    identity=json.loads((out/'identity.json').read_text(encoding='utf8'))
    if not expected_job_id or identity.get('job')!=expected_job_id:
        raise ValueError('same-job identity mismatch')
    argv=identity.get('argv',[])
    if '--out' not in argv or Path(argv[argv.index('--out')+1]).resolve()!=out.resolve():
        raise ValueError('identity output path is not this same job output')
    if env.num_envs!=1 or int(env.step_lim)!=600 or env.end_flag[0]:
        raise ValueError('native environment not continuable')
    log_path=out/'suffix_actions.jsonl'
    raw=log_path.read_bytes()
    lines=raw.decode('utf8').splitlines()
    if len(lines)<2 or not lines[-1] or not lines[-2]:
        raise ValueError('missing exact final submitted/completed pair')
    submitted,completed=[json.loads(s) for s in lines[-2:]]
    now=int(env.take_action_cnt[0])
    if submitted.get('event')!='submitted' or completed.get('event')!='completed':
        raise ValueError('last native action has not completed')
    if submitted.get('original_action_index')!=now-1 or completed.get('native_steps')!=now:
        raise ValueError('last command does not match current native boundary')
    if submitted.get('metadata')!=completed.get('metadata'):
        raise ValueError('submitted/completed metadata do not match')
    action=submitted.get('action',{})
    expected={'left_arm_joint_state','right_arm_joint_state','left_ee_joint_state','right_ee_joint_state'}
    if set(action)!=expected:
        raise ValueError('unexpected target command schema')
    for name,size in [('left_arm_joint_state',6),('right_arm_joint_state',6),('left_ee_joint_state',1),('right_ee_joint_state',1)]:
        values=action[name]
        if not isinstance(values,list) or len(values)!=size or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
            raise ValueError('invalid finite native command shape')
    left=float(action['left_ee_joint_state'][0]);right=float(action['right_ee_joint_state'][0])
    if not (0<=left<=1 and 0<=right<=1):
        raise ValueError('grip command outside native normalized bounds')
    return {'schema':'same_job_completed_grip_command_v1','job':expected_job_id,'native_boundary':now,'submitted_original_action_index':now-1,'left_hold_command':left,'right_hold_command':right,'source':'same_job_suffix_actions_final_submitted_completed_pair','suffix_log_sha256_at_entry':hashlib.sha256(raw).hexdigest(),'metadata':submitted['metadata'],'sensor_measurement_is_not_hold_command':True}
