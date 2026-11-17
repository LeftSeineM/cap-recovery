"""Geometry-driven S1 plans. Every plan is freshly checked against actual state."""
import copy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.spatial import ConvexHull
from .preflight_model import Model, PreflightFailure, transform, pose, digest, world, overlap
from .skill_contract import SkillInapplicable

def bind_held(model,object_id,arms,references=None):
 if object_id not in model.objects or not arms or len(set(arms))!=len(arms):raise SkillInapplicable('unknown object or invalid holding hands')
 if any(a not in model.tools or not 0<=model.grips[a]<.7 for a in arms):raise SkillInapplicable('closed actual holding hands required')
 model.held=object_id
 model.refs={a:np.asarray(references[a],dtype=float).reshape(4,4).copy() if references is not None else np.linalg.inv(model.tools[a])@model.objects[object_id] for a in arms}
 for a,h in model.refs.items():
  actual=np.linalg.inv(model.tools[a])@model.objects[object_id]
  if np.linalg.norm(actual[:3,3]-h[:3,3])>.012 or np.rad2deg(R.from_matrix(actual[:3,:3]@h[:3,:3].T).magnitude())>12:raise SkillInapplicable('immutable held reference already outside limits; no recapture')

def segment_count(distance,spacing):
 # A few ulps above an exact interval must not create an extra segment.
 return max(1,int(np.ceil(float(distance)/float(spacing)-1e-9)))

def rigid_move(model,object_goal,stage,spacing=None,degrees=None):
 if spacing is None:spacing=.008 if len(model.refs)==2 else .012
 if degrees is None:degrees=3 if len(model.refs)==2 else 8
 anchor=next(iter(model.refs));start=model.tools[anchor]@model.refs[anchor];goal=transform(object_goal) if np.asarray(object_goal).shape==(7,) else np.asarray(object_goal)
 n=max(segment_count(np.linalg.norm(goal[:3,3]-start[:3,3]),spacing),segment_count(R.from_matrix(start[:3,:3].T@goal[:3,:3]).magnitude(),np.deg2rad(degrees)))
 for a,h in model.refs.items():n=max(n,segment_count(np.linalg.norm((goal@np.linalg.inv(h))[:3,3]-model.tools[a][:3,3]),spacing))
 def advance(obj,depth=0):
  before=len(model.steps);checkpoint=model.fork()
  try:model.advance({a:obj@np.linalg.inv(h) for a,h in model.refs.items()},stage,model.held,True,depth=7)
  except PreflightFailure as e:
   # Geometry conflicts are not made safe by finer interpolation. Only failed IK is subdivided.
   if 'reach/discontinuity' not in str(e) or depth>=7:raise
   model.__dict__=checkpoint.__dict__
   now=model.tools[anchor]@model.refs[anchor];mid=np.eye(4);mid[:3,3]=(now[:3,3]+obj[:3,3])/2;mid[:3,:3]=Slerp([0,1],R.from_matrix([now[:3,:3],obj[:3,:3]]))(.5).as_matrix()
   advance(mid,depth+1);advance(obj,depth+1)
 for t in np.linspace(0,1,n+1)[1:]:
  obj=np.eye(4);obj[:3,3]=(1-t)*start[:3,3]+t*goal[:3,3];obj[:3,:3]=Slerp([0,1],R.from_matrix([start[:3,:3],goal[:3,:3]]))(t).as_matrix();advance(obj)

def choose_supported_placement(goal,vertices,supports,pickup_sweeps,required_count=2,clearance_m=.002,tracking_reserve_m=.0015):
 # S1 geometric applicability: preserve support and leave future pickup sweeps clear.
 # CPU goal selection only; the ordinary S1 acquire/carry/release pipeline still executes real actions.
 from .support_release import support_report
 original=np.asarray(goal,dtype=float);v=world(vertices,transform(original));axes=[np.array([1.,0.]),np.array([0.,1.])];blockers=[]
 planning_clearance=clearance_m+tracking_reserve_m
 corners=np.array([[x,y,z] for x in (-planning_clearance,planning_clearance) for y in (-planning_clearance,planning_clearance) for z in (-planning_clearance,planning_clearance)])
 def inflated(w):return (w[:,None,:]+corners[None,:,:]).reshape(-1,3)
 for item in pickup_sweeps:
  w=np.asarray(item['vertices'],dtype=float)
  if overlap(v,inflated(w))>1e-6:
   blockers.append(item)
   for n in ConvexHull(w).equations[:,:3]:
    if np.linalg.norm(n[:2])>.1:axes.append(n[:2]/np.linalg.norm(n[:2]))
 candidates=[(0.,0.)];unique={}
 for axis in axes:
  for sign in (-1.,1.):unique[tuple(np.round(sign*axis,6))]=sign*axis
 for d in unique.values():
  if not blockers:continue
  distance=max(float(np.max(np.asarray(item['vertices'])[:,:2]@d)-np.min(v[:,:2]@d)+planning_clearance) for item in blockers)
  if 0.<distance<=.04:candidates.append(tuple((d*distance).tolist()))
 candidates=sorted(set(candidates),key=lambda d:(float(np.linalg.norm(d)),d[0],d[1]));rejected=[]
 for dx,dy in candidates:
  g=original.copy();g[:2]+=[dx,dy];report=support_report(g,vertices,supports,required_count)
  if not report['supported']:rejected.append(dict(offset_m=[dx,dy],reason='required support/centroid not established'));continue
  vv=world(vertices,transform(g));conflicts=[item['object'] for item in pickup_sweeps if overlap(vv,inflated(np.asarray(item['vertices'])))>1e-6]
  if conflicts:rejected.append(dict(offset_m=[dx,dy],reason='future pickup clearance',objects=conflicts));continue
  return g.tolist(),dict(nominal_goal=original.tolist(),selected_goal=g.tolist(),offset_m=[dx,dy],support=report,clearance_m=clearance_m,tracking_reserve_m=tracking_reserve_m,planning_clearance_m=planning_clearance,blocking_objects=[item['object'] for item in blockers],rejected=rejected,scope='hypothetical supported target geometry; runtime fresh preflight and physical support/lift checks required')
 raise SkillInapplicable('no supported bridge placement clearing future pickups: '+str(rejected))

def geometric_escape_offsets(model,lift_m=.03,clearance_m=.002):
 # Derive finite horizontal exits from the actual vertical swept hull and peer geometry.
 # No case, asset-number or entry-side routing. World/peer facet axes only propose; full S0 checks decide.
 obj=model.held;start=world(model.scene[obj]['vertices'],model.objects[obj]);swept=np.vstack([start,start+np.array([0.,0.,lift_m])]);blockers=[];axes=[np.array([1.,0.]),np.array([0.,1.])]
 for name,p in model.objects.items():
  if name==obj:continue
  peer=world(model.scene[name]['vertices'],p)
  if overlap(swept,peer)<=1e-6:continue
  blockers.append(peer)
  for normal in ConvexHull(peer).equations[:,:3]:
   xy=normal[:2];norm=np.linalg.norm(xy)
   if norm>.1:axes.append(xy/norm)
 if not blockers:return [(0.,0.)]
 unique={}
 for axis in axes:
  for sign in (-1.,1.):
   d=axis*sign;key=tuple(np.round(d,6));unique[key]=d
 candidates=[]
 for d in unique.values():
  # Candidate separates the vertical sweep from each blocking peer along this direction.
  distance=max(float(np.max(peer[:,:2]@d)-np.min(swept[:,:2]@d)+clearance_m) for peer in blockers)
  if 0.<distance<=.08:candidates.append(tuple((d*distance).tolist()))
 candidates.sort(key=lambda d:(float(np.linalg.norm(d)),d[0],d[1]))
 return [(0.,0.)]+candidates

def plan_carry(state,calibration,scene,table_height,request,source_sha256):
 if request.get('goal_yaw_symmetries'):return plan_symmetric_transport(state,calibration,scene,table_height,request,source_sha256,True)
 object_id=request['object'];arms=request['arms'];goal=transform(request['goal']);root=Model(state,calibration,scene,table_height);root.support_ids=tuple(request.get('protect',request.get('supports',[])));bind_held(root,object_id,arms,request.get('immutable_references'));start=root.objects[object_id].copy()
 # Small finite routes are geometry candidates, not case identities. Every candidate includes all collision checks.
 failures=[];directions=geometric_escape_offsets(root)
 for dx,dy in directions:
  m=root.fork()
  try:
   escaped=start.copy();escaped[0,3]+=dx;escaped[1,3]+=dy
   if dx or dy:rigid_move(m,escaped,'held_clearance_escape',.001,3)
   lift=escaped.copy();lift[2,3]+=.03;rigid_move(m,lift,'physical_hold_lift',.004,3)
   high=lift.copy();high[2,3]=max(high[2,3],goal[2,3]+.06);rigid_move(m,high,'held_vertical_clearance')
   above=goal.copy();above[2,3]=high[2,3];rigid_move(m,above,'held_transport')
   rigid_move(m,goal,'support_lower',.0015 if len(arms)==2 else .004,3 if len(arms)==2 else 4)
   cert=m.certificate(request,source_sha256);cert.update(object=object_id,arms=arms,immutable_references={a:h.tolist() for a,h in m.refs.items()},initial_object_pose=pose(start).tolist(),physical_lift_required_m=.018,route_offset_m=[dx,dy],rejected_routes=failures);return cert
  except PreflightFailure as e:failures.append(dict(offset=[dx,dy],reason=str(e)))
 raise SkillInapplicable('no checked held path: '+str(failures))

def plan_clear(state,calibration,scene,table_height,request,source_sha256):
 # Intermediate semantic withdrawal ends at a checked local staging pose; final withdrawal retains the old home route.
 # Geometry, original3mm/4deg clearance spacing and2mm endpoint object buffer are unchanged.
 if not request.get('return_home',True):
  staged_failures=[];arms=request['arms']
  # Released protected parts need separation before the wrist rotates home.
  # The6mm local route passed nominal geometry but native withdrawal disturbed
  # a released support. Require18mm for this semantic stage, retaining every
  # original sample/check and the fresh physical-state home replan.
  clearances=(.018,.03,.045) if request.get('protect') else (.006,.012,.018,.03,.045)
  for clearance in clearances:
   m=Model(state,calibration,scene,table_height);high={a:m.tools[a].copy() for a in arms}
   try:
    for a in high:high[a][2,3]+=clearance
    m.move(high,'released_clearance',.003,4)
    m.clearance_margin=.002;m.stage='empty_home'
    previous={a:np.asarray(state['qs'][a]) for a in m.q}
    for prior in m.steps[:-1]:
     if prior['kind']=='move':previous.update({a:np.asarray(q) for a,q in prior['q'].items()})
    m.check({a:(previous[a]+m.q[a])*.5 for a in m.q},m.grips);m.check(m.q,m.grips)
    m.steps[-1]['stage']='empty_home'
    cert=m.certificate(request,source_sha256);cert.update(clearance_m=clearance,route='checked_local_staging',collision_clearance_m=.002,return_home=False,staging_tools={a:pose(m.tools[a]).tolist() for a in arms},rejected_clear_routes=staged_failures);return cert
   except PreflightFailure as e:staged_failures.append(dict(clearance_m=clearance,reason=str(e)))
  if request.get('protect'):
   raise SkillInapplicable('no checked protected withdrawal with minimum18mm vertical separation: '+str(staged_failures))
  # No unchecked staging fallback: retain the established complete home path.
 failures=[];arms=request['arms'];home=request['home'];margin=.002
 # Contact can occur before convex overlap. Keep a conservative2mm object buffer while rotating to home.
 routes=[('local_raise',h) for h in (0.,.006,.012,.018,.03,.045)]+[('raised_fallback',h) for h in (.015,.03,.045,.065)]
 for family,clearance in routes:
  m=Model(state,calibration,scene,table_height);high={a:m.tools[a].copy() for a in arms}
  try:
   if clearance:
    for a in high:high[a][2,3]=high[a][2,3]+clearance if family=='local_raise' else max(high[a][2,3]+clearance,float(home[a][2])+min(clearance,.04))
    m.move(high,'released_clearance',.003 if family=='local_raise' else .006,4)
   m.clearance_margin=margin;m.move({a:transform(home[a]) for a in arms},'empty_home',.024,15)
   cert=m.certificate(request,source_sha256);cert.update(clearance_m=clearance,route=family,collision_clearance_m=margin,rejected_clear_routes=failures);return cert
  except PreflightFailure as e:failures.append(dict(route=family,clearance_m=clearance,reason=str(e)))
 raise SkillInapplicable('no checked withdrawal route with physical clearance buffer: '+str(failures))

def shape_profile(vertices):
 # A triangular extrusion is identified from its vertices, independent of scene labels.
 v=np.unique(np.asarray(vertices,dtype=float),axis=0)
 if len(v)==6:
  for axis in range(3):
   levels=np.unique(np.round(v[:,axis],8))
   if len(levels)!=2:continue
   groups=[v[np.isclose(v[:,axis],z,atol=1e-7)] for z in levels]
   if any(len(g)!=3 for g in groups):continue
   plane=[i for i in range(3) if i!=axis];a=groups[0][:,plane];b=groups[1][:,plane]
   a=a[np.lexsort(a.T)];b=b[np.lexsort(b.T)]
   if np.allclose(a,b,atol=1e-7) and np.linalg.matrix_rank(a[1:]-a[0],tol=1e-8)==2:return 'triangular_prism'
 return 'other'

def grasp_candidates(object_pose,vertices,arms,template):
 # Preserve all nominal candidates first; additional vertical regions follow actual shape height.
 p=transform(object_pose);v=np.asarray(vertices);extent=np.ptp(v,axis=0);long=int(np.argmax(extent));paired=len(arms)==2
 # A geometry-matched prism template fitted from a verified stable source hold.
 # It only proposes rigid grasp endpoints; all normal approach/closure/lift and
 # immutable-reference checks still decide whether physical execution is valid.
 prism_template=Path(__file__).with_name('prism_grasp_template.json')
 if not paired and shape_profile(v)=='triangular_prism' and prism_template.exists():
  import json
  fitted=json.loads(prism_template.read_text());reference=np.asarray(fitted['object_vertices'])
  if v.shape==reference.shape and np.allclose(v,reference,rtol=0.,atol=1e-8):
   calibrated_H=np.asarray(fitted['H_object_in_tool']);mirrors=(0.,180.) if arms[0].startswith('left') else (180.,0.)
   for mirror in mirrors:
    for height in fitted.get('contact_heights_m',[0.]):
     h=calibrated_H.copy();h[:3,3]-=h[:3,:3]@np.array([0.,float(height),0.])
     off=np.eye(4);off[:3,:3]=R.from_euler('y',mirror,degrees=True).as_matrix();contact=p@off@np.linalg.inv(h)
     yield dict(contacts={arms[0]:contact},regions={arms[0]:[0.,float(height),0.]},yaw=mirror,grasp_height_m=float(height),grasp_family='geometry_matched_source_prism',prism_clamp_command=float(fitted.get('clamp_command',.4)),calibration_sha256=digest(fitted))

 if paired and extent[long]<2.5*np.median(np.delete(extent,long)):raise SkillInapplicable('paired-board grasp requires elongated geometry')
 x=p[:3,0].copy();x[2]=0
 if np.linalg.norm(x)<1e-5:raise SkillInapplicable('unsupported upright grasp frame')
 x/=np.linalg.norm(x);z=np.array([0.,0.,1.]);frame=p.copy();frame[:3,:3]=np.column_stack([x,np.cross(z,x),z]);H=np.asarray(template)
 height=float(np.ptp(world(v,p),axis=0)[2])
 rises=[0.] if paired else [0.]+sorted(set(min(.006,height*f) for f in (.1,.2,.3)))
 yaws=(90.,-90.,0.,180.) if not paired and shape_profile(v)=='triangular_prism' else (0.,90.,-90.,180.)
 candidates=[]
 for fraction in ((.78,.62,.46,.90) if paired else (None,)):
  for rise in rises:
   for yaw in yaws:
    contacts={};regions={}
    for i,a in enumerate(arms):
     off=np.eye(4);off[:3,:3]=R.from_euler('z',yaw if not paired else (-90 if i==0 else 90),degrees=True).as_matrix()
     if paired:
      off[long,3]=(1 if i else -1)*extent[long]*.5*fraction;off[2,3]=min(.006,extent[2]*.3)
     else:off[2,3]=rise
     contacts[a]=frame@off@np.linalg.inv(H);regions[a]=off[:3,3].tolist()
    closing=next(iter(contacts.values()))[:3,1];span=float(np.ptp(world(v,p)@closing))
    candidates.append(dict(contacts=contacts,regions=regions,yaw=yaw,grasp_height_m=rise,closing_span_m=span))
    if paired:break
 # A thin panel is grasped across its shortest horizontal extent. A broad
 # nominal intersection is not evidence of a clamp: native capture can miss.
 # Keep the same contact/closure/lift checks and all vertical grasp regions.
 world_extent=np.ptp(world(v,p),axis=0)
 if not paired and shape_profile(v)=='other' and world_extent[2]<=.35*min(world_extent[:2]):
  narrow=min(c['closing_span_m'] for c in candidates)
  candidates=[c for c in candidates if c['closing_span_m']<=narrow+1e-6]
 for candidate in candidates:yield candidate

def select_contact_closure(model,object_id,arms):
 # Calibrated actuator carry domain comes from earlier physical paired-board/cube grasp evidence.
 # Geometry contact is necessary, not a force-closure or friction guarantee. Never branch on object identity.
 profile=shape_profile(model.scene[object_id]['vertices']);max_command=.4 if profile=='triangular_prism' else .34
 target=world(model.scene[object_id]['vertices'],model.objects[object_id]);closed={};evidence={}
 for arm in arms:
  samples=[]
  # Same actuator model: two fingers each travel54mm per normalized command.
  # Physical v075/v077 rectangular-span61.8mm grasps at.34 establish the reference preload.
  tool=model.tools[arm];axis=tool[:3,1];iv=np.linalg.inv(tool);local=world(target,iv);center=(iv@np.r_[model.objects[object_id][:3,3],1.])[:3];lo=-1e9;hi=1e9
  for nx,ny,nz,offset in ConvexHull(local).equations:
   rhs=-(nx*center[0]+nz*center[2]+offset)
   if abs(ny)<1e-9:
    if rhs < -1e-8:raise PreflightFailure('object-origin closing section outside convex shape')
   elif ny>0:hi=min(hi,rhs/ny)
   else:lo=max(lo,rhs/ny)
  if not np.isfinite([lo,hi]).all() or lo>=hi or max(abs(lo),abs(hi))>1:raise PreflightFailure('invalid geometric closing section')
  span=float(hi-lo);reference_span=.0618;reference_command=.34
  derived=float(np.clip(reference_command+(span-reference_span)/.108,.06,max_command))
  commands=(.4,.38,.36,.34,.30,.26,.22,.18,.14,.10,.06) if max_command==.4 else (.34,.30,.26,.22,.18,.14,.10,.06)
  commands=tuple(dict.fromkeys([round(derived,8)]+[x for x in commands if x<=derived]))
  for command in commands:
   nominal=model.fork();nominal.measured_gripper_joints={} # Closing contact prediction uses nominal endpoint, never swept-envelope intersection.
   parts=nominal.parts(arm,nominal.q[arm],command);margins={finger:float(overlap(parts[finger],target)) for finger in ('link7','link8')};samples.append(dict(command=command,contact_margins_m=margins))
   if min(margins.values())>=.0005:
    closed[arm]=command;break
  if arm not in closed:raise PreflightFailure('carry clamp domain lacks bilateral finger contact: '+str(samples))
  evidence[arm]=dict(selected_command=closed[arm],shape_profile=profile,geometric_span_m=span,span_reference_m=reference_span,reference_command=reference_command,actuator_two_finger_travel_m=.108,derived_command=derived,calibrated_max_carry_command=max_command,minimum_bilateral_overlap_m=.0005,samples=samples,scope='exact convex geometric contact in calibrated actuator domain; no force proof')
 return closed,evidence

def approach_contact(model,contacts,object_id):
 # The initial pregrasp remains45mm above the contact pose. Use the existing
 # empty-hand step only for a fully checked, non-contact approach to12mm.
 # Keep the final contact descent at the original3mm/3degree spacing.
 checkpoint=model.fork();near={a:p.copy() for a,p in contacts.items()}
 for p in near.values():p[2,3]+=.012
 try:
  model.move(near,'free_approach',.018,12)
  model.move(contacts,'contact',.003,3,target=object_id,contact=True)
  return dict(mode='checked_free_then_fine',free_approach_contact_exemption=False,free_spacing_m=.018,final_contact_spacing_m=.003,final_fine_distance_m=.012)
 except PreflightFailure as ex:
  model.__dict__=checkpoint.__dict__
  model.move(contacts,'contact',.003,3,target=object_id,contact=True)
  return dict(mode='original_fine_approach_fallback',rejected_free_approach=str(ex),final_contact_spacing_m=.003)


def plan_acquire(state,calibration,scene,table_height,request,source_sha256,template):
 try:return _plan_acquire_from_entry(state,calibration,scene,table_height,request,source_sha256,template)
 except SkillInapplicable as direct_error:
  if 'home' not in request:raise
  retry=copy.deepcopy(request);retry['empty_transit_via_home']=True
  try:
   cert=_plan_acquire_from_entry(state,calibration,scene,table_height,retry,source_sha256,template);cert['direct_entry_rejected']=str(direct_error);return cert
  except SkillInapplicable as home_error:raise SkillInapplicable('direct and checked-home acquisition unavailable: '+str(direct_error)+'; via_home: '+str(home_error))

def _plan_acquire_from_entry(state,calibration,scene,table_height,request,source_sha256,template):
 request=copy.deepcopy(request);request['grasp_template_sha256']=digest(np.asarray(template).tolist());object_id=request['object'];arms=request['arms'];failures=[]
 if any(not .7<=state['grips'][a]<=1. for a in arms):raise SkillInapplicable('acquire requires actual empty/open entry')
 for grasp in grasp_candidates(state['objects'][object_id]['pose'],scene[object_id]['vertices'],arms,template):
  if request.get('grasp_family')=='parallel_prism_end_faces':
   axis=prism_extrusion_axis(scene[object_id]['vertices'])
   if axis is None:raise SkillInapplicable('parallel end-face grasp requires triangular extrusion geometry')
   object_frame=transform(state['objects'][object_id]['pose'])
   if any(abs(float((object_frame[:3,:3].T@p[:3,1])[axis]))<.98 for p in grasp['contacts'].values()):continue
  for opening in (1.,.85,.7):
   m=Model(state,calibration,scene,table_height)
   try:
    if request.get('empty_transit_via_home'):
     transit=plan_clear(state,calibration,scene,table_height,dict(arms=list(state['tools']),home=request['home'],return_home=True),source_sha256)
     m.clearance_margin=.002
     for step in transit['path']:
      if step['kind']!='move':raise SkillInapplicable('empty-home transit unexpectedly changes grip')
      m.stage=step['stage'];target_q={a:np.asarray(step['q'].get(a,q)) for a,q in m.q.items()}
      for fraction in (.5,1.):m.check({a:m.q[a]*(1-fraction)+target_q[a]*fraction for a in m.q},m.grips)
      m.q.update(target_q);m.tools.update({a:transform(p) for a,p in step['tools'].items()});m.steps.append(copy.deepcopy(step))
    contacts=grasp['contacts'];pre={a:p.copy() for a,p in contacts.items()}
    for a in pre:pre[a][2,3]+=.045
    high={a:m.tools[a].copy() for a in arms}
    for a in high:high[a][2,3]=max(high[a][2,3],pre[a][2,3])
    m.move(high,'empty_raise',.018,12);opening_steps,opening_evidence=pregrasp_opening_steps(m,arms,opening);m.close({a:opening for a in arms},object_id,opening_steps)
    m.move(pre,'pregrasp',.018,12);approach_evidence=approach_contact(m,contacts,object_id)
    if request.get('grasp_family')=='parallel_prism_end_faces':
     prism_clamp=float(grasp.get('prism_clamp_command',.4))
     if prism_clamp not in (.3,.34,.4):raise SkillInapplicable('prism clamp target outside admitted finite set')
     close={a:prism_clamp for a in arms};closure_evidence={a:dict(selected_command=prism_clamp,scope='bounded prism closing target; no force certificate; fresh actual capture/lift/immutable hold required') for a in arms}
    else:close,closure_evidence=select_contact_closure(m,object_id,arms)
    m.close(close,object_id,15 if request.get('grasp_family')=='parallel_prism_end_faces' else 10)
    # Future contact response is unknown: do not label original measured apertures as acquisition-end measurements.
    preview_state=dict(geometry_observation='hypothetical_acquisition_endpoint_nominal_aperture',qs={a:q.tolist() for a,q in m.q.items()},tools={a:pose(p).tolist() for a,p in m.tools.items()},objects={k:dict(pose=pose(p).tolist()) for k,p in m.objects.items()},grips=m.grips.copy())
    try:
     if request.get('validation_scope')=='actual_hold_lift_only':
      probe=Model(preview_state,calibration,scene,table_height);bind_held(probe,object_id,arms);lift=probe.objects[object_id].copy();lift[2,3]+=.03;rigid_move(probe,lift,'physical_hold_lift',.004,3);preview=probe.certificate(request,source_sha256)
     elif request.get('feedback_transport_single') or request.get('observed_object_feedback'):
      from .feedback_transport import plan_observed_transport
      preview=plan_observed_transport(preview_state,calibration,scene,table_height,request,source_sha256,include_lift=True)['certificate']
     else:preview=plan_carry(preview_state,calibration,scene,table_height,request,source_sha256)
    except SkillInapplicable as e:
     failures.append(dict(yaw=grasp['yaw'],grasp_height_m=grasp['grasp_height_m'],stage='joint_acquire_carry_preflight',reason=str(e)))
     break
    cert=m.certificate(request,source_sha256);cert['suffix_preflight']=dict(actions=preview['actions'],path_sha256=preview['path_sha256'],scope='ideal acquisition end state; fresh actual replan required after physical capture');cert.update(collision_clearance_m=.002 if request.get('empty_transit_via_home') else 0.,empty_transit='checked_home' if request.get('empty_transit_via_home') else 'direct',object=object_id,arms=arms,free_approach_evidence=approach_evidence,grasp_regions=grasp['regions'],grasp_yaw_deg=grasp['yaw'],grasp_height_m=grasp['grasp_height_m'],grasp_family=grasp.get('grasp_family'),grasp_calibration_sha256=grasp.get('calibration_sha256'),approach_opening=opening,opening_preparation=opening_evidence,closed_commands=close,closure_evidence=closure_evidence,rejected_candidates=failures);return cert
   except PreflightFailure as e:failures.append(dict(yaw=grasp['yaw'],grasp_height_m=grasp['grasp_height_m'],opening=opening,reason=str(e)))
 raise SkillInapplicable('no geometry-checked grasp: '+str(failures))


def prism_extrusion_axis(vertices):
 v=np.unique(np.asarray(vertices,dtype=float),axis=0)
 if len(v)!=6:return None
 for axis in range(3):
  levels=np.unique(np.round(v[:,axis],8))
  if len(levels)!=2:continue
  groups=[v[np.isclose(v[:,axis],z,atol=1e-7)] for z in levels]
  if any(len(g)!=3 for g in groups):continue
  plane=[i for i in range(3) if i!=axis];a=groups[0][:,plane];b=groups[1][:,plane];a=a[np.lexsort(a.T)];b=b[np.lexsort(b.T)]
  if np.allclose(a,b,atol=1e-7) and np.linalg.matrix_rank(a[1:]-a[0],tol=1e-8)==2:return axis
 return None

def exact_goal_variants(goal,vertices,yaws):
 original=transform(goal);v=np.asarray(vertices,dtype=float);reference=world(v,original);out=[]
 for yaw in yaws:
  g=original.copy();g[:3,:3]=R.from_euler('z',float(yaw),degrees=True).as_matrix()@original[:3,:3]
  candidate=world(v,g);dist=np.linalg.norm(reference[:,None,:]-candidate[None,:,:],axis=2);error=float(max(dist.min(0).max(),dist.min(1).max()))
  if error<=1e-7:out.append((float(yaw),g,error))
 if not out:raise SkillInapplicable('no exact geometry-equivalent goal allowed')
 return out

def task_goal_variants(request,scene):
 # Match the existing S2/legacy upright-support semantics in feedback carry.
 # Changed cuboid footprints are checked geometrically, never called symmetry.
 object_id=request['object'];yaws=request.get('goal_yaw_symmetries',[0.])
 if not request.get('task_upright_yaw_free'):return exact_goal_variants(request['goal'],scene[object_id]['vertices'],yaws)
 if scene[object_id]['category']!='2':raise SkillInapplicable('upright yaw freedom is only bound to support-cuboid semantics')
 original=transform(request['goal']);v=np.asarray(scene[object_id]['vertices']);reference=world(v,original);variants=[]
 if not np.allclose(original[:3,2],[0,0,1],atol=1e-7):raise SkillInapplicable('upright original goal required')
 for yaw in yaws:
  if abs(float(yaw)/90-round(float(yaw)/90))>1e-9:raise SkillInapplicable('only bounded quarter-turn yaw candidates')
  g=original.copy();g[:3,:3]=R.from_euler('z',float(yaw),degrees=True).as_matrix()@original[:3,:3];candidate=world(v,g);dist=np.linalg.norm(reference[:,None,:]-candidate[None,:,:],axis=2);error=float(max(dist.min(0).max(),dist.min(1).max()));variants.append((float(yaw),g,error))
 return variants


def plan_symmetric_transport(state,calibration,scene,table_height,request,source_sha256,include_lift=False):
 object_id=request['object'];arms=request['arms'];root=Model(state,calibration,scene,table_height);root.support_ids=tuple(request.get('protect',request.get('supports',[])));bind_held(root,object_id,arms,request.get('immutable_references'));failures=[]
 anchor=arms[0]
 if request.get('task_upright_yaw_free'):
  # S2 permits quarter-turn goal yaw only for upright support cuboids.
  # Footprint changes are modeled, never treated as exact mesh symmetry.
  if scene[object_id]['category']!='2':raise SkillInapplicable('upright yaw freedom is only bound to support-cuboid semantics')
  original=transform(request['goal']);v=np.asarray(scene[object_id]['vertices']);reference=world(v,original);variants=[]
  if not np.allclose(original[:3,2],[0,0,1],atol=1e-7):raise SkillInapplicable('upright original goal required')
  for yaw in request['goal_yaw_symmetries']:
   if abs(float(yaw)/90-round(float(yaw)/90))>1e-9:raise SkillInapplicable('only bounded quarter-turn yaw candidates')
   g=original.copy();g[:3,:3]=R.from_euler('z',float(yaw),degrees=True).as_matrix()@original[:3,:3];candidate=world(v,g);dist=np.linalg.norm(reference[:,None,:]-candidate[None,:,:],axis=2);error=float(max(dist.min(0).max(),dist.min(1).max()));variants.append((float(yaw),g,error))
 else:variants=exact_goal_variants(request['goal'],scene[object_id]['vertices'],request['goal_yaw_symmetries'])
 # Prefer less wrist rotation; every selected route still passes all original geometry checks.
 preferred=transform(request['home'][anchor]) if request.get('task_upright_yaw_free') and request.get('home') else root.tools[anchor]
 variants.sort(key=lambda row:float(R.from_matrix(preferred[:3,:3].T@(row[1]@np.linalg.inv(root.refs[anchor]))[:3,:3]).magnitude()))
 for yaw,goal,error in variants:
  for clearance in request.get('carry_clearances_m',[0.,.003,.006,.009,.012,.015,.03,.045,.06]):
   m=root.fork()
   try:
    if include_lift:
     lift=m.objects[object_id].copy();lift[2,3]+=.03;rigid_move(m,lift,'physical_hold_lift',.004,3)
    now=m.tools[anchor]@m.refs[anchor];high=now.copy();high[2,3]=max(now[2,3],goal[2,3]+float(clearance));rigid_move(m,high,'held_vertical_clearance');above=goal.copy();above[2,3]=high[2,3];rigid_move(m,above,'held_transport');rigid_move(m,goal,'support_lower',.004,4)
    cert=m.certificate(request,source_sha256);cert.update(object=object_id,arms=arms,selected_goal=pose(goal).tolist(),goal_yaw_semantics='task-permitted upright quarter turn' if request.get('task_upright_yaw_free') else 'exact mesh equivalence',goal_yaw_symmetry_deg=yaw,goal_geometry_error_m=error,carry_clearance_m=float(clearance),immutable_references={a:h.tolist() for a,h in m.refs.items()},rejected_routes=failures);return cert
   except PreflightFailure as e:failures.append(dict(yaw=yaw,clearance_m=clearance,reason=str(e)))
 raise SkillInapplicable('no checked exact-equivalent held path: '+str(failures))


def choose_table_group_placement(items,scene,pickup_sweeps,table_height,clearance_m=.002,max_translation_m=.25,tracking_reserve_m=.0015,placement_envelopes=None,extra_group_meshes=None,future_pickups=None):
 # Bounded separating translations revisit newly encountered obstacles, including
 # full-open-hand destination envelopes. Only uncommitted support groups move.
 import heapq
 from .support_release import support_report
 pad=clearance_m+tracking_reserve_m
 corners=np.array([[x,y,z] for x in (-pad,pad) for y in (-pad,pad) for z in (-pad,pad)])
 peers=[(p['object'],(np.asarray(p['vertices'])[:,None,:]+corners).reshape(-1,3)) for p in pickup_sweeps]
 base_meshes=[world(v,transform(item['goal'])) for item in items for v in [scene[item['object']]['vertices']]+list((placement_envelopes or {}).get(item['object'],[]))]
 physical_meshes=[world(scene[item['object']]['vertices'],transform(item['goal'])) for item in items]+list(extra_group_meshes or [])
 base_meshes+=list(extra_group_meshes or [])
 pending=[(0.,0.,0.)];seen=set();rejected=[];all_blockers=set();sequence=0
 while pending and len(seen)<96:
  _,dx,dy=heapq.heappop(pending);key=(round(dx,6),round(dy,6))
  if key in seen:continue
  seen.add(key);delta=np.array([dx,dy,0.]);blocking=[]
  for mesh in base_meshes:
   v=mesh+delta
   for name,w in peers:
    if overlap(v,w)>1e-6:blocking.append((v,w,name));all_blockers.add(name)
  reservations={}
  for pickup in future_pickups or []:
   alternatives=[]
   for index,meshes in enumerate(pickup['alternatives']):
    conflicts=[]
    for mesh in physical_meshes:
     for hand in meshes:
      w=(np.asarray(hand)[:,None,:]+corners).reshape(-1,3);v=mesh+delta
      if overlap(v,w)>1e-6:conflicts.append((v,w,pickup['object']))
    alternatives.append((len(conflicts),index,conflicts))
   count,index,conflicts=min(alternatives,key=lambda row:(row[0],row[1]))
   reservations[pickup['object']]=index
   if count:blocking.extend(conflicts);all_blockers.add(pickup['object'])
  goals=[];supported=True
  for item in items:
   g=np.asarray(item['goal'],dtype=float).copy();g[:2]+=[dx,dy];goals.append(g.tolist())
   supported=supported and support_report(g,scene[item['object']]['vertices'],[],1,table_height)['supported']
  if not blocking and supported:
   return goals,dict(offset_m=[dx,dy],blocking_objects=sorted(all_blockers),clearance_m=clearance_m,tracking_reserve_m=tracking_reserve_m,planning_clearance_m=pad,rejected=rejected,evaluated_candidates=len(seen),candidate_limit=96,full_open_hand_geometry_included=bool(placement_envelopes),future_pickup_reservations=reservations,scope='CPU geometry candidate, not physical support or success proof')
  rejected.append(dict(offset_m=[dx,dy],blocking_objects=sorted({name for _,_,name in blocking}),table_supported=bool(supported)))
  axes=[np.array([1.,0.]),np.array([0.,1.])]
  for _,w,_ in blocking:
   for normal in ConvexHull(w).equations[:,:3]:
    if np.linalg.norm(normal[:2])>.1:axes.append(normal[:2]/np.linalg.norm(normal[:2]))
  directions={tuple(np.round(axis*sign,6)):axis*sign for axis in axes for sign in (-1.,1.)}
  for direction in directions.values():
   if not blocking:continue
   distance=max(float(np.max(w[:,:2]@direction)-np.min(v[:,:2]@direction)+1e-5) for v,w,_ in blocking)
   candidate=np.array([dx,dy])+direction*distance;norm=float(np.linalg.norm(candidate));ck=tuple(np.round(candidate,6))
   if distance>0 and norm<=max_translation_m and ck not in seen:
    sequence+=1;heapq.heappush(pending,(sequence,float(candidate[0]),float(candidate[1])))
 raise SkillInapplicable('no checked group destination within finite candidate/radius bounds: '+str(rejected))


def pregrasp_opening_steps(model,arms,opening):
 # Retain one native hold/observation when the EMPTY gripper is already at the requested opening.
 # A changed command or unsettled measured finger keeps the original three-step preparation.
 nominal=-.01+.054*opening;reports={};ready=True
 for a in arms:
  same=abs(float(model.grips[a])-opening)<=1e-9;measured=model.measured_gripper_joints.get(a)
  settled=measured is None or (len(measured)==2 and np.isfinite(measured).all() and max(abs(float(q)-nominal) for q in measured)<=.0005)
  reports[a]=dict(command_matches=same,measured_joints=measured,nominal_joint=nominal,settled=bool(settled),observation='actual' if measured is not None else 'CPU hypothetical nominal')
  ready=ready and same and settled
 return (1 if ready else 3),dict(steps=1 if ready else 3,arms=reports,scope='empty pregrasp idempotent opening only; grasp closure and release waits unchanged')
