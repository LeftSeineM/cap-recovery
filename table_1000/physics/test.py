"""Run asset physics TESTS on one or more backends and summarize each asset."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import traceback

from table_1000.common.batch import add_arguments, job_count


def load_tests(path):
    spec = importlib.util.spec_from_file_location('table1000_asset_tests',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TESTS


class PhysicalFailure(Exception):
    """An asset test's physical assertion failed."""


def run_test(test, context):
    try:
        test(context)
    except AssertionError as error:
        raise PhysicalFailure(str(error)) from error


def ground_clearance(model, rows, instance_names, ground_z):
    """Evaluate collision hull vertices against the plane from recorded poses."""
    import numpy as np
    from table_1000.physics.simulation import rotation
    worlds, owners, points, spheres = {}, {}, {}, {}
    for node in model['nodes']:
        name,parent=node['name'],node['parent']
        worlds[name]=worlds.get(parent,np.eye(4)) @ np.array(node['matrix'])
        owners[name]=name if node['rigid'] else owners.get(parent)
        if node.get('role')=='collision':
            body=owners[name]
            transform=np.linalg.inv(worlds[body]) @ worlds[name]
            if node.get('shape')=='SPHERE':
                spheres.setdefault(body,[]).append((transform[:3,3],node['radius']))
            else:
                vertices=np.c_[node['vertices'],np.ones(len(node['vertices']))]
                points.setdefault(body,[]).extend((vertices @ transform.T)[:,:3])
    points={body:np.array(vertices) for body,vertices in points.items()}
    minimum=[]
    for row in rows:
        values=[]
        for instance in instance_names:
            for body,vertices in points.items():
                prefix=f'{instance}.{body}.'
                quaternion=[row[prefix+'quaternion.'+axis] for axis in 'wxyz']
                values.append(float((vertices @ rotation(quaternion).T)[:,2].min()+row[prefix+'position.z']-ground_z))
            for body,entries in spheres.items():
                prefix=f'{instance}.{body}.'
                quaternion=[row[prefix+'quaternion.'+axis] for axis in 'wxyz']
                r=rotation(quaternion)
                values.extend(float((r@center)[2]+row[prefix+'position.z']-radius-ground_z) for center,radius in entries)
        minimum.append(min(values))
    index=int(np.argmin(minimum))
    return {'minimum_collision_clearance_m':minimum[index],
            'maximum_geometry_penetration_m':max(0.,-minimum[index]),'time_s':rows[index]['time'],
            'sampling_interval_s':rows[1]['time']-rows[0]['time'],
            'method':'collision hull vertices and analytic sphere extents transformed by recorded rigid poses; geometric overlap excludes contact margins'}


def benchmark(asset,name,test,app,physics="cpu",num_envs=1,utilization=None):
    from table_1000.physics.testing import TestContext
    # Warm a disposable scene: restoring poses alone cannot restore the
    # contact solver's internal state. Every measured repeat starts fresh.
    warm = TestContext(asset, name=name, app=app, physics=physics, num_envs=num_envs)
    start = time.perf_counter()
    try:
        run_test(test, warm)
        assert warm.session.time >= 1., 'RTF warmup requires at least one simulated second'
        warmup = time.perf_counter()-start
    finally:
        warm.close()
    rounds=[]
    for _ in range(3):
        seconds=wall=prepare=0.
        finals=[]
        while seconds < 10-1e-8:
            start=time.perf_counter()
            ctx=TestContext(asset,name=name,app=app,physics=physics,num_envs=num_envs)
            ctx.utilization=utilization
            construction=time.perf_counter()-start
            try:
                run_test(test, ctx)
                seconds+=ctx.session.time;wall+=time.perf_counter()-ctx.experiment_started
                if utilization: utilization.stop()
                prepare+=construction+ctx.preparation_seconds
                finals.append(ctx.rows[-1])
            finally:
                ctx.close()
        rounds.append({'simulated_seconds':seconds,'wall_seconds':wall,'RTF':seconds/wall,
                       'independent_repeats':len(finals),'preparation_reset_seconds':prepare,
                       'final_states':finals})
    median=statistics.median(r['RTF'] for r in rounds)
    return {'warmup':{'wall_seconds':warmup,'method':'one complete experiment in a disposable scene before measured rounds'},
            'rounds':rounds,'median_RTF':median,'environments':num_envs,'median_environment_seconds_per_second':median*num_envs,
            'utilization':utilization.summary() if utilization else None,
            'includes':'physical steps, official behavior, test actions, state reading and in-memory sampling; no camera or rendering'}


def conditions(ctx):
    import carb
    import numpy as np
    from pxr import PhysxSchema,UsdPhysics
    from omni.physx.bindings import _physx
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.version import get_version
    from threadpoolctl import threadpool_info
    from table_1000.sim.isaaclab.fixtures import LINK_MASS, LINK_INERTIA
    settings=carb.settings.get_settings()
    engine={}
    for instance in ctx.session.instances.values():
        shapes={}
        for name in instance.body_names:
            # A combined rigid view pads to the largest body's shape count.
            # One-body views report only that body's actual collision shapes.
            tensor=SimulationManager.get_physics_sim_view().create_rigid_body_view(instance.paths[name])
            shapes[name]={'count':tensor.max_shapes,
                'materials_static_dynamic_restitution':np.unique(np.asarray(tensor.get_material_properties().cpu()).reshape(-1,3),axis=0).tolist(),
                'contact_offsets_m':np.unique(np.asarray(tensor.get_contact_offsets().cpu())).tolist(),
                'rest_offsets_m':np.unique(np.asarray(tensor.get_rest_offsets().cpu())).tolist()}
        engine[instance.name]={'mass_kg':instance.mass.tolist(),'center_of_mass_m':instance.local_com.tolist(),
              'inertia_kg_m2':instance.local_inertia.tolist(),
              'collision_shapes':shapes,
              'rigid_bodies':{name:{'position_iterations':PhysxSchema.PhysxRigidBodyAPI(ctx.session.stage.GetPrimAtPath(instance.paths[name])).GetSolverPositionIterationCountAttr().Get(),
                                   'velocity_iterations':PhysxSchema.PhysxRigidBodyAPI(ctx.session.stage.GetPrimAtPath(instance.paths[name])).GetSolverVelocityIterationCountAttr().Get(),
                                   'ccd':PhysxSchema.PhysxRigidBodyAPI(ctx.session.stage.GetPrimAtPath(instance.paths[name])).GetEnableCCDAttr().Get()}
                              for name in instance.body_names}}
        if instance.articulation is not None:
            engine[instance.name]['articulation_dof_parameters'] = {
                'friction_coefficients':instance.articulation.get_dof_friction_coefficients().tolist(),
                'friction_properties_static_dynamic_viscous':instance.articulation.get_dof_friction_properties().tolist(),
                'stiffnesses':instance.articulation.get_dof_stiffnesses().tolist(),
                'dampings':instance.articulation.get_dof_dampings().tolist()}
    cpu=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    scene=next(p for p in ctx.session.stage.Traverse() if p.IsA(UsdPhysics.Scene))
    scene_api=PhysxSchema.PhysxSceneAPI(scene)
    scene_ccd=scene_api.GetEnableCCDAttr().Get()
    return {'engine':'Isaac Lab / Isaac Sim / '+ctx.physics.upper()+' PhysX','isaac_version':get_version(),'dt':ctx.dt,'solver':scene_api.GetSolverTypeAttr().Get(),'gpu_dynamics':scene_api.GetEnableGPUDynamicsAttr().Get(),'device':ctx.world.device,'scene_CCD':scene_ccd,
            'solver_configuration':ctx.world.cfg.physx.to_dict(),
            'fixtures':[({'body':f.body.path,'kind':'fixed articulation root','locked':True}
                         if not hasattr(f, 'stiffness') else {'body':f.body.path,'kind':'six-axis tensor articulation','locked':f.locked,
                         'virtual_link_mass_kg':LINK_MASS,'virtual_link_inertia_kg_m2':LINK_INERTIA,
                         'virtual_links':6,'stiffness':f.stiffness.tolist(),'damping':f.damping.tolist()})
                        for f in ctx.fixtures],
            'instances':len(ctx.session.instances),'gravity':ctx.gravity,'ground':getattr(ctx,'ground_config',None),
            'physics_values_from_engine':engine,'cpu':cpu,
            'load_average':list(os.getloadavg()),'environment_threads':{key:os.getenv(key) for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']},
            'worker_controls':{'carb_tasking':settings.get('/plugins/carb.tasking.plugin/threadCount'),
                 'tbb':settings.get('/plugins/omni.tbb.globalcontrol/maxThreadCount'),
                 'physx':settings.get(_physx.SETTING_NUM_THREADS)},'native_math_threadpools':threadpool_info(),
            'render_settings':{path:settings.get(path) for path in ['/rtx/rendermode','/persistent/rtx/modes/rt2/enabled',
              '/rtx/rtpt/maxBounces','/rtx/rtpt/maxSpecularAndTransmissionBounces','/rtx/post/motionblur/enabled',
                '/rtx/directLighting/sampledLighting/enabled','/rtx/directLighting/sampledLighting/samplesPerPixel',
                '/rtx/directLighting/sampledLighting/denoisingTechnique','/rtx/directLighting/domeLight/sampleCount',
                '/rtx/directLighting/domeLight/denoisingTechnique','/rtx/shadows/sampleCount','/rtx/shadows/denoiser/enable',
                '/rtx/reflections/sampledLighting/samplesPerPixel','/rtx/reflections/denoiser/enabled','/rtx/post/taa/samples']},
            'camera':ctx.camera_config,
            'render_passes_per_capture':2 if ctx.camera_instance else 0,
            'lighting':{'dome_intensity':ctx.session.stage.GetPrimAtPath('/World/Light').GetAttribute('inputs:intensity').Get(),
                        'sun_intensity':ctx.session.stage.GetPrimAtPath('/World/Sun').GetAttribute('inputs:intensity').Get(),
                        'sun_rotation_xyz_degrees':list(ctx.session.stage.GetPrimAtPath('/World/Sun').GetAttribute('xformOp:rotateXYZ').Get())}
                       if ctx.camera_instance else None,
            'standard_dt':ctx.dt==.004}


def run_video(asset,name,test,output,app,startup,rtf,gpu,physics="cpu",num_envs=1,utilization=None):
    from table_1000.physics.testing import TestContext
    from table_1000.physics.plots import plots
    directory=output/name
    started=time.perf_counter()
    ctx=TestContext(asset,name=name,render=True,app=app,physics=physics,num_envs=num_envs)
    ctx.utilization=utilization
    construction=time.perf_counter()-started
    try:
        run_test(test, ctx)
        replica_deviation=max(instance.view.replica_deviation() for instance in ctx.session.instances.values())
        frames=directory/'frames';frames.mkdir()
        for i,image in enumerate(ctx.images):image.save(frames/f'{i:05}.jpg',quality=95)
        columns=list(ctx.rows[0])
        with (directory/'trace.csv').open('w') as stream:
            writer=csv.DictWriter(stream,columns);writer.writeheader();writer.writerows(ctx.rows)
        fps=ctx.camera_config['fps']
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(fps),'-i',str(frames/'%05d.jpg'),
                        '-c:v','libx264','-pix_fmt','yuv420p','-crf','20',str(output/(name+'.mp4'))],check=True)
        wall=time.perf_counter()-ctx.video_started
        video_utilization=utilization.summary() if utilization else None
        plots(ctx.rows,ctx.plot_columns,directory,ctx.units)
        # Fresh independent input executions, not subtraction of render costs.
        errors=[max(abs(float(row[key])-float(ctx.rows[-1][key])) for key in row.keys() & ctx.rows[-1].keys() if key!='time')
                for round in rtf['rounds'] for row in round['final_states']] if rtf else []
        if rtf is None:
            # One pass: physics steps and sampling of this run, camera capture excluded.
            rtf={'method':'physics steps and sampling within the video run, camera capture excluded; use --benchmark for the standard measurement',
                 'simulated_seconds':ctx.session.time,'wall_seconds':ctx.integrating_seconds,
                 'RTF':ctx.session.time/ctx.integrating_seconds,'environments':num_envs,
                 'environment_seconds_per_second':num_envs*ctx.session.time/ctx.integrating_seconds}
        result={'test':name,'execution_status':'completed','asset':str(asset),
            'conditions':{**conditions(ctx),'gpu':gpu},
            'environments':{'count':num_envs,'recorded_and_rendered':'environment 0; replicas receive identical inputs and are hidden',
                'spacing_m':float(ctx.session.offsets[1][1]) if num_envs>1 else None,
                'final_replica_max_position_deviation_m':replica_deviation},'columns':ctx.units,'plots':ctx.plot_columns,
            'steps':ctx.session.steps,'frames':len(ctx.images),'sampling':'state after integration at t; action columns are the force applied over [t-dt,t); initial t=0 has zero actions',
            'physics_callbacks':'actions/state priority -100; official BehaviorScript priority 0 when present; batched force flush priority 100; one external physical step; actual solver and iterations in conditions',
            'performance':{'no_render':rtf,'video_end_to_end':{'simulated_seconds':ctx.session.time,'wall_seconds':wall,'RTF':ctx.session.time/wall,
                     'environments':num_envs,'environment_seconds_per_second':num_envs*ctx.session.time/wall,
                     'utilization':video_utilization},
                 'preparation':{'process_startup_seconds':startup,'construction_seconds':construction,'initialization_render_warmup_reset_seconds':ctx.preparation_seconds},
                 'independent_no_render_vs_video_final_max_scalar_error':max(errors) if errors else None},
            'reaction_forces':'contact reaction forces not sampled; grip columns record the force each grip applies'}
        result['metrics'] = ctx.metrics
        if getattr(ctx, 'experiment_details', None):
            result['experiment'] = ctx.experiment_details
        if ctx.grips:
            result['grips'] = ctx.grip_details()
        if getattr(ctx,'ground_config',None):
            result['ground_contact']=ground_clearance(json.loads((asset.parent/'model.json').read_text()),
                ctx.rows,ctx.session.instances,ctx.ground_config['z'])
        print('PHYSICS_TEST_COMPLETED',name,flush=True)
        return result
    finally:
        ctx.close()


def clear(output,name):
    """Remove one test's previous results, so a rerun never reports them."""
    if (output/name).exists():shutil.rmtree(output/name)
    (output/(name+'.mp4')).unlink(missing_ok=True)


def run_case(asset,name,test,output,app,startup,gpu,physics="cpu",num_envs=1,utilization=None,standard=False):
    directory=output/name
    clear(output,name)
    directory.mkdir(parents=True)
    try:
        rtf=benchmark(asset,name,test,app,physics=physics,num_envs=num_envs,utilization=utilization) if standard else None
        return run_video(asset,name,test,output,app,startup,rtf,gpu,physics=physics,num_envs=num_envs,utilization=utilization)
    except PhysicalFailure:
        return {'test':name,'execution_status':'failed','error':traceback.format_exc()}


# Backend name -> PhysX device. Each backend writes physics_test/<name>/.
BACKENDS={'isaac-cpu':'cpu','isaac-gpu':'gpu'}
ROOT=Path(__file__).resolve().parents[2]


def significant(value,digits=4):
    return float(f'{value:.{digits}g}')


def summary(directory,names,error=None):
    """Collect each backend's result.json into one human-facing digest."""
    asset=str(directory.relative_to(directory.parents[1]))
    if error:
        return {'asset':asset,'error':error,'backends':{},'tests':{}}
    backends,tests={},{}
    for backend in BACKENDS:
        output=directory/'physics_test'/backend
        for name in names:
            path=output/name/'result.json'
            if not path.exists():continue
            result=json.loads(path.read_text())
            test=tests.setdefault(name,{'status':{},'metrics':{},'rtf':{}})
            status=test['status'][backend]=result['execution_status']
            entry=backends.setdefault(backend,{'completed':0,'failed':[]})
            if status!='completed':
                entry['failed'].append(name)
                test.setdefault('error',{})[backend]=result['error'].strip().splitlines()[-1]
                continue
            entry['completed']+=1
            entry.setdefault('engine',f"Isaac Sim {result['conditions']['isaac_version'][0]} {BACKENDS[backend].upper()} PhysX")
            entry.setdefault('environments',result['environments']['count'])
            for key,metric in result.get('metrics',{}).items():
                test['metrics'].setdefault(key,{'label':metric['label'],'unit':metric['unit']})[backend]=significant(metric['value'])
            performance=result['performance']
            physics=performance['no_render']
            test['rtf'][backend]={'physics':significant(physics.get('median_RTF',physics.get('RTF')),3),
                                  'video':significant(performance['video_end_to_end']['RTF'],3)}
    ordered={name:tests[name] for name in names if name in tests}
    return {'asset':asset,'backends':backends,'tests':ordered}


def write_summary(directory,names,error=None,source=None):
    path=directory/'physics_test'/'summary.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    report=summary(directory,names,error)
    if source is not None:report.update(source.record(path.parent))
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return path


def entry(directory):
    return directory/('object.usda' if (directory/'object.usda').exists() else 'object.usdz')



def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('assets/objects'),help='built assets root')
    parser.add_argument('--assets',nargs='+',help='asset directories relative to --source; default all with physics_test.py')
    parser.add_argument('--physics',nargs='+',choices=tuple(BACKENDS),required=True,help='physics backends to run')
    parser.add_argument('--tests',nargs='+',help='only these tests')
    parser.add_argument('--scripts',type=Path,help='read physics_test.py from this root (e.g. asset_sources/objects) instead of the build snapshot')
    parser.add_argument('--num-envs',type=int,help='parallel copies of each experiment; default 1 on CPU, 4 on GPU')
    parser.add_argument('--benchmark',action='store_true',help='standard RTF measurement: warmup and three timed rounds before the video run')
    parser.add_argument('--plots-only',action='store_true')
    add_arguments(parser)
    args=parser.parse_args(argv)
    if args.num_envs is not None and args.num_envs<1:parser.error('--num-envs must be positive')
    source=args.source.resolve()
    references=([Path(reference) for reference in args.assets] if args.assets is not None
                else sorted(path.parent.relative_to(source) for path in source.rglob('physics_test.py')))
    scripts={reference:(args.scripts.resolve()/reference if args.scripts else source/reference)/'physics_test.py'
             for reference in references}
    if not args.scripts and not args.plots_only:
        for reference,script in scripts.items():
            edited=ROOT/'asset_sources'/'objects'/reference/'physics_test.py'
            if edited.is_file() and script.is_file() and edited.read_bytes()!=script.read_bytes():
                print(f'WARNING {reference}: {script} differs from {edited}; rebuild with build_physics.py '
                      'or test the edited one with --scripts asset_sources/objects',flush=True)
    names,broken={},{}
    for reference,script in scripts.items():
        try:
            names[reference]=list(load_tests(script))
        except Exception:
            # A broken definition fails only its own asset.
            traceback.print_exc()
            broken[reference]=traceback.format_exc().strip().splitlines()[-1]
            continue
        unknown=set(args.tests or ())-set(names[reference])
        if unknown:parser.error(f'{reference}: unknown tests {sorted(unknown)}')
    if args.plots_only:
        from table_1000.physics.plots import plot_trace
        for reference in names:
            for backend in args.physics:
                output=source/reference/'physics_test'/backend
                for name in args.tests or names[reference]:
                    result=json.loads((output/name/'result.json').read_text())
                    plot_trace(output/name/'trace.csv',result['plots'],result['columns'])
        if broken:
            raise SystemExit('failed: '+', '.join(f'{reference} (physics_test.py)' for reference in broken))
        return
    job_count(parser,args)
    from table_1000.common.batch_runtime import run as run_program
    selected={}
    units=[]
    for reference,names_for_asset in names.items():
        for backend in args.physics:
            selected[reference,backend]=args.tests or names_for_asset
            units.extend((reference,backend,name) for name in selected[reference,backend])
    scripts_root=args.scripts.resolve() if args.scripts else source
    if not args.dry_run:
        for reference,backend,name in units:
            clear(source/reference/'physics_test'/backend,name)
    missing=run_program('table_1000.physics.batch_program:program', vars(args) | dict(
        units=units,source=source,scripts=str(scripts_root),
        scripts_relative=str(scripts_root.relative_to(ROOT)) if scripts_root.is_relative_to(ROOT)
            and scripts_root.relative_to(ROOT).parts[0] not in ('assets','data','runs','outputs') else None),
        root=ROOT,output=ROOT/'runs/physics-tests'/str(time.time_ns()))
    if args.dry_run:
        return
    for unit in missing:
        reference,backend,name=unit[:3]
        path=source/reference/'physics_test'/backend/name/'result.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        error=json.loads(path.read_text())['error'] if path.exists() else f'not run: {backend} worker exited before result.json'
        path.write_text(json.dumps({'test':name,'execution_status':'failed','error':error},indent=2)+'\n')
    failed=[]
    for (reference,backend),tests in selected.items():
        output=source/reference/'physics_test'/backend
        for name in tests:
            if not (output/name/'result.json').exists():
                (output/name).mkdir(parents=True,exist_ok=True)
                (output/name/'result.json').write_text(json.dumps({'test':name,'execution_status':'failed',
                    'error':f'not run: {backend} worker exited before result.json'},indent=2)+'\n')
        if any(json.loads((output/name/'result.json').read_text())['execution_status']!='completed'
               for name in tests):
            failed.append(f'{reference} {backend}')
    for reference in references:
        print('SUMMARY',write_summary(source/reference,names.get(reference,[]),broken.get(reference),source=missing),flush=True)
    failed+=[f'{reference} (physics_test.py)' for reference in broken]
    if failed:
        raise SystemExit('failed: '+', '.join(failed))


if __name__ == '__main__':
    main()
