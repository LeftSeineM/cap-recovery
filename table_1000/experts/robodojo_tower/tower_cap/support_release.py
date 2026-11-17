"""Shared support/release observations for single and paired object skills.
Geometric support and sampled stability are diagnostic; native success stays authoritative.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from .entry_state import vertices_world
from scipy.spatial.transform import Rotation as R

def support_report(object_pose,object_vertices,supports,required_count=1,table_height=None,gap_limits=(-.001,.003)):
 v=vertices_world(object_vertices,object_pose);bottom=float(v[:,2].min());center=np.mean(v[:,:2],axis=0);details=[];contacts=[]
 if len({s.get('id',i) for i,s in enumerate(supports)})!=len(supports):raise ValueError('support identities must be distinct')
 for s in supports:
  w=vertices_world(s['vertices'],s['pose']);gap=bottom-float(w[:,2].max());h=ConvexHull(w[:,:2]);inside=bool(np.all(h.equations[:,:2]@center+h.equations[:,2]<=1e-6))
  # A board spans separate support footprints: each actual support must overlap its projection.
  hv=ConvexHull(v[:,:2]);hw=ConvexHull(w[:,:2]);eq=np.vstack([hv.equations,hw.equations]);r=linprog([0,0],A_ub=eq[:,:2],b_ub=-eq[:,2],bounds=[(None,None)]*2,method='highs');overlap=bool(r.success)
  details.append(dict(object=s.get('id'),vertical_gap_m=gap,projection_overlap=overlap,center_inside=inside,geometric_contact=bool(overlap and gap_limits[0]<=gap<=gap_limits[1])))
 # Clip actual overlap polygons, then check the projected centroid inside their combined support region.
  if details[-1]['geometric_contact']:
   polygon=v[ConvexHull(v[:,:2]).vertices,:2].tolist()
   for equation in hw.equations:
    clipped=[]
    for i,a in enumerate(polygon):
     b=polygon[(i+1)%len(polygon)];a=np.asarray(a);b=np.asarray(b);da=float(equation[:2]@a+equation[2]);db=float(equation[:2]@b+equation[2])
     if da<=1e-9:clipped.append(a.tolist())
     if (da<=1e-9)!=(db<=1e-9):clipped.append((a+(b-a)*da/(da-db)).tolist())
    polygon=clipped
    if not polygon:break
   contacts.extend(polygon)
 center_supported=False
 if len(contacts)>=3:
  try:
   h=ConvexHull(np.asarray(contacts));center_supported=bool(np.all(h.equations[:,:2]@center+h.equations[:,2]<=1e-6))
  except Exception:pass
 count=sum(x['geometric_contact'] for x in details);table_gap=bottom-table_height if table_height is not None else None
 table=bool(table_gap is not None and gap_limits[0]<=table_gap<=gap_limits[1])
 return dict(supported=bool((count>=required_count and center_supported) or (table and required_count==1)),details=details,projected_center_supported=center_supported,table_gap_m=table_gap,required_count=required_count,scope='convex projection and height; not force proof')

def stability_report(samples,position_limit=.002,angle_limit_deg=2.):
 if len(samples)<3:return dict(stable=False,reason='three post-release samples required')
 p=np.asarray(samples);q=R.from_quat(np.c_[p[:,4:],p[:,3]]);dp=float(np.max(np.linalg.norm(p[:,:3]-p[0,:3],axis=1)));dq=float(np.rad2deg((q*q[0].inv()).magnitude()).max())
 return dict(stable=bool(dp<=position_limit and dq<=angle_limit_deg),position_change_m=dp,angle_change_deg=dq,scope='sampled stability; not whole-task acceptance')

def release_on_support(motion,read_object,object_vertices,supports,closed_commands,table_height=None,required_count=1,observe_actions=3,record=lambda **x:None):
 before=support_report(read_object(),object_vertices,supports(),required_count,table_height);record(stage='before_release',report=before)
 if not before['supported']:raise RuntimeError('actual support not verified; no release')
 motion.guard=lambda:None
 arms=motion.arms if hasattr(motion,'arms') else {motion.arm.arm_name:motion}
 for i in range(1,7):
  for name,arm in arms.items():motion.cmd[arm.grip_key]=[closed_commands[name]+(1.-closed_commands[name])*i/6]
  motion.send('s1_supported_release')
 samples=[]
 for _ in range(observe_actions):motion.send('s1_release_observation');samples.append(read_object().copy())
 after=support_report(samples[-1],object_vertices,supports(),required_count,table_height);stable=stability_report(samples);record(stage='after_release',support=after,stability=stable)
 return dict(support_before=before,support_after=after,stability=stable,postconditions_met=bool(after['supported'] and stable['stable']),samples=[x.tolist() for x in samples])
