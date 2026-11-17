"""Restore recorded velocities through actual rigid-body APIs after render sync.
No recorder/cache mutation. No physics advance. Compare actual dynamic getters.
"""
import numpy as np
import torch
import objstate

def restore_and_verify(scene_manager,data,k):
    index=objstate.build_index(scene_manager,0)
    lookup={key:obj for key,_i,obj in list(index['rigid'])+list(index['dynamic'])}
    rows=[]
    for i,key in enumerate(data['rigid_names']):
        obj=lookup[key]
        if objstate._type_of(obj)!='rigid':continue
        lin=np.asarray(data['rigid_linvel'][k,i],dtype=np.float32)
        ang=np.asarray(data['rigid_angvel'][k,i],dtype=np.float32)
        before=obj.get_current_dynamic_state()
        before_lin=objstate._np(before.linear_velocity).reshape(3)
        before_ang=objstate._np(before.angular_velocity).reshape(3)
        obj.set_linear_velocity(torch.tensor(lin,dtype=torch.float32))
        obj.set_angular_velocity(torch.tensor(ang,dtype=torch.float32))
        state=obj.get_current_dynamic_state()
        actual_lin=objstate._np(state.linear_velocity).reshape(3)
        actual_ang=objstate._np(state.angular_velocity).reshape(3)
        errlin=float(np.max(np.abs(actual_lin-lin)))
        errang=float(np.max(np.abs(actual_ang-ang)))
        rows.append({'key':key,'source_linear':lin.tolist(),'source_angular':ang.tolist(),'before_linear_error':float(np.max(np.abs(before_lin-lin))),'before_angular_error':float(np.max(np.abs(before_ang-ang))),'actual_linear':actual_lin.tolist(),'actual_angular':actual_ang.tolist(),'linear_error':errlin,'angular_error':errang})
        if not np.isfinite(actual_lin).all() or not np.isfinite(actual_ang).all() or max(errlin,errang)>1e-6:
            raise RuntimeError('Actual source velocity readback mismatch: '+key)
    return {'rows':rows,'actual_rigid_API_readback_verified':True,'max_error':max(max(r['linear_error'],r['angular_error']) for r in rows),'physics_steps_added':0,'recorded_cache_overwrite':False}
