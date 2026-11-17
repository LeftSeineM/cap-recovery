"""Expert-development diagnostic: finite physical hold test, never whole-tower credit."""
import numpy as np
from .preflight_model import Model,transform
from .planner import plan_acquire,bind_held,rigid_move,shape_profile
from .skills import pick_place_block,release_on_support,clear_and_park

def verify_parallel_face_lift(rt,object_id,arm):
 params=dict(object=object_id,arms=[arm],goal=rt.read_object(object_id).tolist(),validation_scope='actual_hold_lift_only',grasp_family='parallel_prism_end_faces',mode='expert_development')
 def run(call):
  if rt.held is not None:raise RuntimeError('diagnostic requires empty hands after inherited support placement')
  initial=np.asarray(rt.read_object(object_id)).copy();cert=plan_acquire(rt.snapshot(),rt.calibration,rt.scene,rt.table,params,rt.sha,rt.template);rt.execute(cert,call);rt.capture(object_id,[arm]);start=float(rt.read_object(object_id)[2]);rt.record(stage='diagnostic_before_physical_lift',object=object_id,actual=rt.snapshot())
  request=dict(params,immutable_references={a:h.tolist() for a,h in rt.references.items()});model=Model(rt.snapshot(),rt.calibration,rt.scene,rt.table,context=rt.context);bind_held(model,object_id,[arm],request['immutable_references']);lift=model.objects[object_id].copy();lift[2,3]+=.03;rigid_move(model,lift,'physical_hold_lift',.004,3);liftcert=model.certificate(request,rt.sha);liftcert.update(object=object_id,arms=[arm]);rt.execute(liftcert,call);rt.guard();rise=float(rt.read_object(object_id)[2]-start)
  if rise<.018:raise RuntimeError('diagnostic actual roof rise below existing18mm condition')
  rt.record(stage='diagnostic_physical_lift_verified',object=object_id,rise_m=rise,actual=rt.snapshot())
  # Return to the same measured table support through a fresh checked path; no tower transport is attempted.
  model=Model(rt.snapshot(),rt.calibration,rt.scene,rt.table,context=rt.context);bind_held(model,object_id,[arm],request['immutable_references']);rigid_move(model,transform(initial),'diagnostic_support_lower',.004,3);lower=model.certificate(request,rt.sha);lower.update(object=object_id,arms=[arm]);rt.execute(lower,call)
  for row in [release_on_support(rt,object_id,[],1,True),clear_and_park(rt,[arm],protect=[object_id])]:
   if row['status']!='completed':raise RuntimeError('diagnostic cleanup '+str(row.get('reason')))
  return dict(actual_state=rt.snapshot(),physical_lift_m=rise,scope='finite real lift and return; no full tower or generalization claim')
 return rt.call('verify_parallel_face_lift',params,run)
