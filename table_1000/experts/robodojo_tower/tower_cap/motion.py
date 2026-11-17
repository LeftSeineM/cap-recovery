"""S0: RoboDojo native action emitter, world poses in xyz,wxyz."""
from copy import deepcopy
import numpy as np
from .legal_joint_representative import nearest_legal, runtime_limits
from .rightmost_target import arr
from .style import count, interp, qdist

class Motion:
    def __init__(self, env, emit_action, last_command, arm_name="right_arm"):
        if env.num_envs != 1:
            raise ValueError("single live RoboDojo environment required")
        self.env, self.emit = env, emit_action
        self.rm = env.robot_manager
        self.arm = self.rm.get_robot_by_arm_name(arm_name)
        self.arm_key = self.rm.process_name(self.arm.arm_name)
        self.grip_key = self.rm.process_name(self.arm.gripper_name)
        self.cmd = deepcopy(last_command)
        self.limits = runtime_limits(self.rm, self.arm, 0)[1:]
        self.guard = lambda: None
        self.diagnose = getattr(emit_action, "diagnose", None)
        # Original policy prefix may contain noisy normalized gripper values.
        # Keep prefix exact; issue legal open/hold commands only in recovery.
        for robot in (self.rm.get_robot_by_arm_name("left_arm"),self.rm.get_robot_by_arm_name("right_arm")):
            key=self.rm.process_name(robot.gripper_name)
            original=np.asarray(self.cmd[key],dtype=float)
            if not np.isfinite(original).all():
                raise ValueError("nonfinite original gripper command")
            legal=np.clip(original,0.,1.)
            self.cmd[key]=legal.tolist()
            if self.diagnose is not None and not np.array_equal(original,legal):
                self.diagnose({"event":"recovery_gripper_command_normalized","key":key,"prefix_command":original.tolist(),"recovery_command":legal.tolist(),"prefix_unchanged":True})

    def tool(self):
        return arr(self.rm.get_real_endpose(self.arm,[0],is_relative=False)[0]).reshape(7)

    def send(self, stage):
        before=int(self.env.take_action_cnt[0])
        if self.env.end_flag[0] or before>=self.env.step_lim:
            raise RuntimeError("native terminal or budget reached")
        keep=self.emit(deepcopy(self.cmd),{"stage":stage})
        if int(self.env.take_action_cnt[0])!=before+1:
            raise RuntimeError("emitter must advance one action")
        self.guard()
        if not keep or self.env.end_flag[0]:
            if self.env.end_flag[0] and self.env.success[0]:
                raise OfficialSuccess()
            raise RuntimeError("native emitter stopped")

    def move(self, end, stage, spacing=.004, degrees=3):
        # Import from the NEW package when integrating this method.
        from .style import count, interp, qdist
        from .rightmost_target import arr
        from .legal_joint_representative import nearest_legal
        max_depth = 7
        def log(data):
            # Owner should install append-only JSONL diagnostic callback.
            callback = getattr(self, 'diagnose', None)
            if callback is not None:
                callback(dict(stage=stage, native_step=int(self.env.take_action_cnt[0]), **data))
        def advance(desired, depth):
            current = arr(self.rm.get_joint(self.arm, [0])[0]).reshape(-1)
            actual = self.tool()
            from .multisolution import continuous_world_ik
            try:
                picked=continuous_world_ik(self.rm,self.arm,desired,current,self.limits,lambda row:log(dict(event='multi_candidate',**row)))
                ik={'status':'Success','joint_value':picked}
            except Exception as ex:
                log(dict(event='multi_failure',error=repr(ex)))
                ik={'status':'Fail'}
            q = None
            reason = 'IK failed'
            info = {}
            if ik.get('status') == 'Success':
                raw = arr(ik['joint_value']).reshape(-1)
                try:
                    q, info = nearest_legal(raw, current, *self.limits)
                    # Keep original safety boundary, including finite legality checks.
                    if np.max(np.abs(q-current)) > .20:
                        reason = 'joint discontinuity'
                        q = None
                except ValueError as exc:
                    reason = 'joint legality: ' + str(exc)
                log(dict(event='ik_candidate', depth=depth, current=current.tolist(),
                         desired=desired.tolist(), raw=raw.tolist(), detail=info,
                         accepted=q is not None, reason=None if q is not None else reason))
            else:
                log(dict(event='ik_rejected', depth=depth, current=current.tolist(),
                         desired=desired.tolist(), status=str(ik.get('status'))))
            if q is None:
                translation = np.linalg.norm(desired[:3]-actual[:3])
                rotation = qdist(actual[3:], desired[3:])
                if depth >= max_depth or (translation <= .00025 and rotation <= .15):
                    raise RuntimeError(reason + ': ' + stage + ' (bounded subdivision exhausted)')
                # No rejected candidate is ever emitted. Move midpoint through physics,
                # then solve endpoint again from the NEW actual state, never old q.
                midpoint = next(interp(actual, desired, 2))
                advance(midpoint, depth+1)
                advance(desired, depth+1)
                return
            self.cmd[self.arm_key] = q.tolist()
            self.send(stage)
            measured = self.tool()
            if np.linalg.norm(measured[:3]-desired[:3]) > .015 or qdist(measured[3:], desired[3:]) > 15:
                # Keep original post-action tracking guard. No automatic retry here.
                raise RuntimeError('tracking error: ' + stage)
        start = self.tool()
        for desired in interp(start, end, count(start, end, spacing, degrees)):
            advance(desired, 0)
    

class OfficialSuccess(Exception):
    """Native terminal success was observed while executing an action."""
