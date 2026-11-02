"""Replay two recorded pen/A5 edge-contact microstates on a static table.

Clip 12 is the exact state preceding the first bad contact. Clip 13 uses the
recorded state plus an explicit 10 µm upward pen displacement: cold isolation
of its unperturbed state is quiet. Both previously produced a downward contact
normal with 5–7 mm fictitious penetration and a >2 m/s pen launch in 4–12 ms.
The fixture contains only poses/velocities; geometry comes from built assets.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets')
    parser.add_argument('--case', choices=('clip12', 'clip13'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixture = json.loads((Path(__file__).with_name('fixtures') / 'pen_cover_edge.json').read_text())
    case = next(row for row in fixture['cases'] if row['name'] == args.case)
    from table_1000.common.batch import worker_gpu
    from table_1000.sim.isaaclab.app import start_app
    app = start_app(gpu=worker_gpu(args.gpu), workers=4, device='cpu', cameras=False)
    failed = True
    try:
        import carb
        import numpy as np
        import torch
        import isaaclab.sim as sim
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics
        from table_1000.assets import usd_entry
        from table_1000.sim.isaaclab.scene import object_cfgs, object_structure, spawner_classes
        dt = fixture['dt_s']
        world = sim.SimulationContext(sim.SimulationCfg(
            device='cpu', dt=dt, use_fabric=False, render_interval=10**9,
            physx=sim.PhysxCfg(solver_type=1, enable_ccd=True)))
        cfg = InteractiveSceneCfg(num_envs=1, env_spacing=5., replicate_physics=False)
        TableCfg, _ = spawner_classes()
        cfg.table = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Table', spawn=TableCfg(table=fixture['table']))
        for i, (name, reference) in enumerate(fixture['objects'].items()):
            structure = object_structure(str(usd_entry(args.assets / reference)))
            for key, value in object_cfgs(name, structure, i).items():
                setattr(cfg, key, value)
        scene = InteractiveScene(cfg)
        for prim in sim.get_current_stage().Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        world.reset()
        scene.reset()
        for name, entry in case['state'].items():
            scene[name].write_root_pose_to_sim(torch.tensor([entry['root_pose']], dtype=torch.float32))
            scene[name].write_root_velocity_to_sim(torch.tensor([entry['root_velocity']], dtype=torch.float32))
        world.forward()
        scene.update(dt)
        carb.settings.get_settings().set_bool('/physics/disableContactProcessing', False)
        contacts, rows = [], []
        clock = [0]

        def on_contact(headers, data):
            for header in headers:
                paths = [str(PhysicsSchemaTools.intToSdfPath(path))
                         for path in (header.collider0, header.collider1)]
                if not (any('Object_pen_black/' in p for p in paths)
                        and any('Object_notebook_a5/' in p for p in paths)):
                    continue
                for point in data[header.contact_data_offset:header.contact_data_offset + header.num_contact_data]:
                    # PhysX normals point from collider1 toward collider0.
                    sign = 1. if 'Object_pen_black/' in paths[0] else -1.
                    contacts.append({'step': clock[0], 'separation_m': float(point.separation),
                                     'normal_toward_pen': [sign * float(point.normal[k]) for k in range(3)],
                                     'colliders': paths})

        subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
        cpu, wall = time.process_time(), time.perf_counter()
        pen = scene['pen_black__pen']
        for step in range(round(fixture['duration_s'] / dt) + 1):
            velocity = pen.data.root_com_vel_w[0].cpu().numpy()
            rows.append({'step': step, 'speed_m_s': float(np.linalg.norm(velocity[:3])),
                         'angular_speed_rad_s': float(np.linalg.norm(velocity[3:])),
                         'pose': pen.data.root_link_pose_w[0].cpu().numpy().tolist()})
            if step < round(fixture['duration_s'] / dt):
                clock[0] = step + 1
                scene.write_data_to_sim()
                world.step(render=False)
                scene.update(dt)
        wrong_deep = [c for c in contacts if c['separation_m'] < -.001 and c['normal_toward_pen'][2] < -.5]
        maximum_speed = max(row['speed_m_s'] for row in rows)
        maximum_angular = max(row['angular_speed_rad_s'] for row in rows)
        result = dict(case=args.case, assets=str(args.assets.resolve()), dt_s=dt,
                      source=case, contacts=contacts, rows=rows, reverse_deep_contacts=len(wrong_deep),
                      maximum_speed_m_s=maximum_speed, maximum_angular_rad_s=maximum_angular,
                      cpu_s=time.process_time()-cpu, wall_s=time.perf_counter()-wall)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        assert not wrong_deep, f'reversed deep cover contact: {wrong_deep[0]}'
        assert maximum_speed < .10, f'pen launched at {maximum_speed:g} m/s'
        assert maximum_angular < 15., f'pen spun at {maximum_angular:g} rad/s'
        subscription = None
        failed = False
        print(f'PEN_COVER_EDGE_PASS {args.case}: {maximum_speed:g} m/s, {maximum_angular:g} rad/s', flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        app.close(exit_code=int(failed))


if __name__ == '__main__':
    main()
