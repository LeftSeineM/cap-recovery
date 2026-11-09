"""S0: RoboDojo native action emitter, world poses in xyz,wxyz."""
from copy import deepcopy
import numpy as np
from .legal_joint_representative import nearest_legal, runtime_limits
from .rightmost_target import arr
from .style import count, interp, qdist

class Motion:
    def __init__(self, env, emit_action, last_command):
        if env.num_envs != 1:
            raise ValueError("single live RoboDojo environment required")
        self.env, self.emit = env, emit_action
        self.rm = env.robot_manager
        self.arm = self.rm.get_robot_by_arm_name("right_arm")
        self.arm_key = self.rm.process_name(self.arm.arm_name)
        self.grip_key = self.rm.process_name(self.arm.gripper_name)
        self.cmd = deepcopy(last_command)
        self.limits = runtime_limits(self.rm, self.arm, 0)[1:]
        self.guard = lambda: None

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
        start=self.tool()
        for desired in interp(start,end,count(start,end,spacing,degrees)):
            ik=self.rm.solve_ik(target_pose=desired.tolist(),env_idx=0,robot=self.arm,trans="world")
            if ik.get("status")!="Success":
                raise RuntimeError("IK failed: "+stage)
            current=arr(self.rm.get_joint(self.arm,[0])[0])
            q,_=nearest_legal(arr(ik["joint_value"]).reshape(-1),current,*self.limits)
            if np.max(np.abs(q-current))>.20:
                raise RuntimeError("joint discontinuity: "+stage)
            self.cmd[self.arm_key]=q.tolist()
            self.send(stage)
            actual=self.tool()
            if np.linalg.norm(actual[:3]-desired[:3])>.015 or qdist(actual[3:],desired[3:])>15:
                raise RuntimeError("tracking error: "+stage)

class OfficialSuccess(Exception):
    """Native terminal success was observed while executing an action."""
