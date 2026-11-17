"""Tower task adapter v0.2; reused S0, new tower S2/S1. No state writes."""
import numpy as np
from .rightmost_target import arr

def read_pose(env,label):
 lm=env.scene_manager.layout_manager;inst=lm.get_instance_name(env_idx=0,label=label)
 s=lm.get_scene_object(env_idx=0,inst_name=inst).get_current_dynamic_state()
 return np.r_[arr(s.position).reshape(3),arr(s.orientation).reshape(4)]

def expert(env,emit,last,job_output):
 from .tower_strategy import run
 return run(env,emit,last,job_output)
