"""State/geometry entry inference; no case IDs, switch steps, or object-name rules.
Contact neighborhoods and relative-motion evidence nominate grasps, never force proof.
A candidate must pass the unchanged immutable-reference physical lift in S1.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation as R

def transform(p):
 p=np.asarray(p);m=np.eye(4);m[:3,3]=p[:3];m[:3,:3]=R.from_quat(np.r_[p[4:],p[3]]).as_matrix();return m

def vertices_world(vertices,p):
 t=transform(p);return np.asarray(vertices)@t[:3,:3].T+t[:3,3]

def convex_margin(v,w):
 eq=np.vstack([ConvexHull(v).equations,ConvexHull(w).equations])
 r=linprog([0,0,0,-1],A_ub=np.c_[eq[:,:3],np.ones(len(eq))],b_ub=-eq[:,3],bounds=[(None,None)]*4,method='highs')
 if not r.success:raise ValueError('contact geometry LP failed')
 return float(r.x[3])

def classify_entry(history,grip_commands,scene,gripper,contact_neighborhood=.0015,max_relative_translation=.005,max_relative_angle_deg=5.):
 if not history:raise ValueError('actual state history required')
 last=history[-1];arms=sorted(last['tools']);matches={a:[] for a in arms};reports=[]
 for a in arms:
  grip=float(grip_commands[a]);tool=last['tools'][a]
  if not 0<=grip<=1:raise ValueError('normalized observed gripper command required')
  if grip>=.7:continue
  fingers=[]
  for key,sign in [('link7',1.),('link8',-1.)]:
   v=np.array(gripper['parts'][key]['hull_vertices']);v[:,1]+=sign*(-.01+.054*grip);fingers.append(vertices_world(v,tool))
  for name,spec in scene.items():
   obj=last['objects'][name]['pose'];world=vertices_world(spec['vertices'],obj)
   margins=[convex_margin(v,world) for v in fingers]
   refs=[np.linalg.inv(transform(s['tools'][a]))@transform(s['objects'][name]['pose']) for s in history]
   h=refs[-1];dp=max(float(np.linalg.norm(x[:3,3]-h[:3,3])) for x in refs);dq=max(float(np.rad2deg(R.from_matrix(x[:3,:3]@h[:3,:3].T).magnitude())) for x in refs)
   contact=all(x>=-contact_neighborhood for x in margins);relative=dp<=max_relative_translation and dq<=max_relative_angle_deg
   report=dict(arm=a,object=name,finger_margins_m=margins,relative_translation_m=dp,relative_angle_deg=dq,contact_candidate=contact,relative_motion_consistent=relative)
   reports.append(report)
   if contact and relative:matches[a].append(name)
 ambiguous={a:v for a,v in matches.items() if len(v)>1}
 associations={a:v[0] for a,v in matches.items() if len(v)==1}
 if ambiguous:mode='ambiguous'
 elif len(associations)==2 and len(set(associations.values()))==1:mode='paired_held_candidate'
 elif len(associations)==1:mode='single_held_candidate'
 elif len(associations)==0 and all(float(grip_commands[a])>=.7 for a in arms):mode='empty'
 else:mode='unresolved_closed_gripper'
 return dict(mode=mode,arms_by_object={n:[a for a,k in associations.items() if k==n] for n in set(associations.values())},ambiguous=ambiguous,reports=reports,physical_hold_verified=False,scope='actual state and calibrated collision geometry; not contact-force or physical-lift proof',thresholds=dict(contact_neighborhood_m=contact_neighborhood,relative_translation_m=max_relative_translation,relative_angle_deg=max_relative_angle_deg))
