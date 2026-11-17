"""Scene semantic bindings live here; S1 consumes geometry, never instance ordinals."""
import json,re
from pathlib import Path
import numpy as np
from .entry_state import vertices_world
from .skills import Runtime
from . import read_pose

def build_runtime(env,emit,last,out):
 root=Path(__file__).parent;assets=json.loads((root/'geometry_assets.json').read_text());binding=json.loads(Path(out,'reset_asset_binding.json').read_text());scene={}
 for label,instance in binding['actual_instances'].items():
  match=re.fullmatch(r'block_(\d+)_(\d+)',instance)
  if not match:raise RuntimeError('unrecognized scene asset identity')
  cat=match.group(1);scene[label]=dict(vertices=assets['geometry']['exact'][cat]['vertices'],category=cat,instance=instance)
 rows=[json.loads(line) for line in Path(out,'continuous_states.jsonl').read_text().splitlines()];reset=rows[0]
 # Table surface inferred from original reset bottoms using exact collision vertices.
 bottoms=[float(vertices_world(scene[k]['vertices'],v['pose'])[:,2].min()) for k,v in reset['objects'].items()];table=float(np.median(bottoms))
 if max(abs(x-table) for x in bottoms)>.004:raise RuntimeError('reset table plane cannot be established from all scene objects')
 home=reset['tools'];template=json.loads((root/'inherited_grasp_template.json').read_text())['initial_original_boundary_H'];rt=Runtime(env,emit,last,out,scene,assets,table,lambda k:read_pose(env,k),home,template)
 # Preserve exact final five prefix actions, including original-boundary duplicate snapshot.
 rt.reset_snapshot=reset
 history=rows[-6:];entry=rt.adopt_entry(history)
 rt.record(stage='scene_binding',scene=scene,table_height=table,geometry_sha256=__import__('hashlib').sha256((root/'geometry_assets.json').read_bytes()).hexdigest(),entry=entry)
 return rt,entry

def structure_plan(rt,entry):
 # Bind tower roles to asset categories once. The four cube roles are assigned from actual completed supports/held entry.
 scene=rt.scene;objects={k:np.asarray(rt.read_object(k)) for k in scene};boxes=[k for k,s in scene.items() if s['category']=='2'];boards=sorted([k for k,s in scene.items() if s['category'] in ('0','4')],key=lambda k:np.ptp(np.asarray(scene[k]['vertices'])[:,0]),reverse=True)
 caps=[k for k,s in scene.items() if s['category']=='1'];roofs=[k for k,s in scene.items() if s['category']=='3']
 if len(boxes)!=4 or len(boards)!=2 or len(caps)!=1 or len(roofs)!=1:raise RuntimeError('unsupported tower role inventory')
 from .support_release import support_report
 held=rt.held;board=boards[0];dims={k:np.ptp(np.asarray(s['vertices']),axis=0) for k,s in scene.items()};cube_h=float(dims[boxes[0]][2]);board_h=float(dims[board][2])
 # Original policy places the base cubes away from untouched initial bridge footprint. Detect them via table support and closest shared front row.
 table_boxes=[k for k in boxes if k!=held and support_report(objects[k],scene[k]['vertices'],[],1,rt.table)['supported']]
 front=[k for k in table_boxes if abs(objects[k][1]-objects[board][1])>cube_h*1.5 and abs(objects[k][0])<dims[board][0]*.5 and np.linalg.norm(objects[k][:3]-np.asarray(rt.reset_snapshot['objects'][k]['pose'])[:3])>.02]
 if len(front)>=2:
  pairs=[(a,b) for i,a in enumerate(front) for b in front[i+1:] if abs(objects[a][1]-objects[b][1])<cube_h*.4 and dims[board][0]*.4<abs(objects[a][0]-objects[b][0])<dims[board][0]*.9]
 else:pairs=[]
 if len(pairs)>1:raise RuntimeError('existing base role ambiguity')
 if pairs:
  base=sorted(pairs[0],key=lambda k:objects[k][0]);center=np.mean([objects[k][:2] for k in base],axis=0);preserve=set(base)
 elif front and held in boxes:
  # Inherited second base block: preserve first grounded cube and derive its counterpart within the measured bridge span.
  fixed=min(front,key=lambda k:abs(objects[k][1]-objects[held][1]));sep=dims[board][0]*.8;sign=1 if objects[held][0]>objects[fixed][0] else -1;base=[fixed,held] if sign>0 else [held,fixed];center=np.array([objects[fixed][0]+sign*sep/2,objects[fixed][1]]);preserve={fixed}
 else:
  # Early entry: role choice follows the actual held cube; remaining boxes sorted by spatial side/distance.
  left=[k for k in boxes if objects[k][0]<objects[board][0]];right=[k for k in boxes if objects[k][0]>objects[board][0]]
  if held is None and len(left)==2 and len(right)==2:
   # Clear the future paired-board pickup neighborhood before building the base.
   # Native task permits either block from each spatial side; preserve late/held entries.
   base=[min(side,key=lambda k:float(np.linalg.norm(objects[k][:2]-objects[board][:2]))) for side in (left,right)]
  else:
   first=held if held in boxes else min(boxes,key=lambda k:objects[k][0]);opposite=max([k for k in boxes if k!=first],key=lambda k:objects[k][0]);base=[first,opposite]
  center=np.array([0.,-.08]);preserve=set()
 remaining=sorted([k for k in boxes if k not in base],key=lambda k:objects[k][0]);ident=[1.,0.,0.,0.];sep=dims[board][0]*.8;midsep=dims[boards[1]][0]*.74;plan=[]
 def add(k,xy,z,q,supports,role,bimanual=False):
  goal=[float(xy[0]),float(xy[1]),float(z),*q];arms=['left_arm','right_arm'] if bimanual else [next(iter(rt.references)) if k==held and len(rt.references)==1 else ('left_arm' if objects[k][0]<0 else 'right_arm')]
  plan.append(dict(object=k,goal=goal,supports=supports,role=role,arms=arms,preserve=k in preserve))
 for k,sign in zip(base,(-1,1)):add(k,center+[sign*sep/2,0],rt.table+cube_h/2,ident,[],'base_support')
 # Existing established supports stay fixed. Only an uncommitted base pair can be jointly shifted.
 reserved_pickup_sweeps=[]
 if not preserve:
  from .planner import choose_table_group_placement,grasp_candidates,world,transform
  obstacles=[]
  for k in scene:
   if k in base:continue
   vv=vertices_world(scene[k]['vertices'],objects[k]);obstacles.append(dict(object=k,vertices=np.vstack([vv,vv+np.array([0.,0.,.03])]).tolist()))
  # Nominal full-open hand geometry is a planning hypothesis; fresh measured
  # grasp/carry/release guards still decide every executed action.
  hand_local=[];inverse_template=np.linalg.inv(np.asarray(rt.template))
  for part,spec in rt.calibration['geometry']['gripper']['parts'].items():
   v=np.asarray(spec['hull_vertices']).copy()
   if part in ('link7','link8'):v[:,1]+=(1. if part=='link7' else -1.)*.044
   hand_local.append((v@inverse_template[:3,:3].T+inverse_template[:3,3]).tolist())
  envelopes={item['object']:hand_local for item in plan}
  from .future_pickup import future_pickups
  targets=[dict(object=k,arms=['left_arm' if objects[k][0]<0 else 'right_arm']) for k in remaining]
  future,future_evidence=future_pickups(rt.snapshot(),rt.calibration,scene,rt.table,targets,rt.template,relocated=base+[board])
  bridge_pose=[float(center[0]),float(center[1]),rt.table+cube_h+board_h/2,*ident]
  bridge_mesh=world(scene[board]['vertices'],transform(bridge_pose))
  goals,group_evidence=choose_table_group_placement(plan,scene,obstacles,rt.table,placement_envelopes=envelopes,extra_group_meshes=[bridge_mesh],future_pickups=future)
  group_evidence['future_pickup_endpoint_evidence']=future_evidence
  for pickup in future:
   index=group_evidence['future_pickup_reservations'][pickup['object']]
   for mesh in pickup['alternatives'][index]:reserved_pickup_sweeps.append(dict(object=pickup['object'],vertices=mesh))
  group_evidence['full_open_hand_geometry_included']=True
  for item,goal in zip(plan,goals):item['goal']=goal;item['group_placement_clearance']=group_evidence
  center=np.mean([np.asarray(item['goal'])[:2] for item in plan],axis=0)
 z=rt.table+cube_h+board_h/2
 # Use planned base support poses and the actual future-source pickup geometry; no instance-specific routing.
 from .planner import choose_supported_placement
 sweeps=list(reserved_pickup_sweeps)
 for k in scene:
  if k in base or k==board:continue
  vv=vertices_world(scene[k]['vertices'],objects[k]);sweeps.append(dict(object=k,vertices=np.vstack([vv,vv+np.array([0.,0.,.03])]).tolist()))
 supports=[dict(id=x['object'],pose=x['goal'],vertices=scene[x['object']]['vertices']) for x in plan if x['object'] in base]
 bridge_goal,placement_evidence=choose_supported_placement([float(center[0]),float(center[1]),z,*ident],scene[board]['vertices'],supports,sweeps,2)
 center=np.asarray(bridge_goal[:2]);add(board,center,z,ident,base,'base_bridge',True);plan[-1]['placement_clearance']=placement_evidence
 middle_z=z+board_h/2+cube_h/2
 for k,sign in zip(remaining,(-1,1)):add(k,center+[sign*midsep/2,0],middle_z,ident,[board],'middle_support')
 upper_z=middle_z+cube_h/2+dims[boards[1]][2]/2;add(boards[1],center,upper_z,ident,remaining,'upper_bridge',True)
 cap=caps[0];cap_z=upper_z+dims[boards[1]][2]/2+dims[cap][2]/2;add(cap,center,cap_z,[0.,0.,0.,1.],[boards[1]],'cap')
 roof=roofs[0];rv=np.asarray(scene[roof]['vertices']);roof_q=[2**-.5,2**-.5,0.,0.];from .entry_state import transform
 roof_local_bottom=float(vertices_world(rv,[0,0,0,*roof_q])[:,2].min());roof_z=cap_z+dims[cap][2]/2-roof_local_bottom;add(roof,center,roof_z,roof_q,[cap],'roof')
 if held is not None:
  # Do not park/regrasp a late-entry board while pretending it is an empty entry.
  held_index=next(i for i,x in enumerate(plan) if x['object']==held)
  if held_index>0 and any(not x['preserve'] for x in plan[:held_index]):raise RuntimeError('held late-entry object prerequisites not established from current geometry')
 Path(rt.out,'tower_dependency_plan.json').write_text(json.dumps(dict(plan=plan,source='actual-state geometry and task semantic role graph; no case/switch routing',table_height=rt.table),indent=2))
 return plan
