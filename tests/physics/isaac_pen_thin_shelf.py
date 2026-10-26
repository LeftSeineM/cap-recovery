"""Drop capped pens onto a 3.5 mm shelf at the expert's 4 ms physics step.

The 0.8 mm spring balls must stay above the shelf. Discrete collision can
leave a ball underneath while its cap stays above, clamping the shelf with a
large persistent load. Test several heights and rolls without changing masses,
drives, contact offsets, or the solver timestep.
"""

import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from table_1000.common.batch import worker_gpu
    from table_1000.sim.isaaclab.app import start_app
    app = start_app(gpu=worker_gpu(args.gpu), cameras=False)
    ctx = None
    failed = True
    try:
        import numpy as np
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade
        from scipy.spatial.transform import Rotation
        from table_1000.physics.testing import TestContext
        from table_1000.scenes.geometry import AssetGeometry
        ctx = TestContext(args.asset, name='thin_shelf', app=app, physics='cpu')
        ctx.simulation(dt=.004)
        stage = ctx.session.stage
        shelf = UsdGeom.Cube.Define(stage, ctx.session.environment(0) + '/Shelf')
        shelf.CreateSizeAttr(1.)
        xf = UsdGeom.Xformable(shelf)
        xf.AddTranslateOp().Set(Gf.Vec3d(0., 0., -.0035 / 2))
        xf.AddScaleOp().Set(Gf.Vec3f(4., 4., .0035))
        UsdPhysics.CollisionAPI.Apply(shelf.GetPrim())
        collision = PhysxSchema.PhysxCollisionAPI.Apply(shelf.GetPrim())
        collision.CreateContactOffsetAttr(.0001)
        collision.CreateRestOffsetAttr(0.)
        material = UsdShade.Material.Define(stage, '/World/ShelfMaterial')
        props = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        props.CreateStaticFrictionAttr(.30)
        props.CreateDynamicFrictionAttr(.22)
        props.CreateRestitutionAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(shelf.GetPrim()).Bind(material, materialPurpose='physics')
        geometry = AssetGeometry(args.asset.parent / 'model.json')
        pens, inputs = [], []
        for index, (height, roll) in enumerate((h, r) for h in (.06, .09, .12) for r in (0., 20., 40., 60.)):
            pen = ctx.asset if not index else ctx.spawn(args.asset, name=f'pen{index}')
            rotation = Rotation.from_euler('x', roll, degrees=True)
            q = rotation.as_quat()[[3, 0, 1, 2]]
            position = ((index % 4 - 1.5) * .3, (index // 4 - 1) * .3,
                        height - geometry.lowest(rotation.as_matrix()))
            ctx.initial(pen, position=position, rotation=q)
            pens.append(pen)
            inputs.append({'name': pen.name, 'height_m': height, 'roll_deg': roll})
        ctx.observe('lowest_ball_center_z_m', lambda: min(
            pen.body(f'cap_ball{i}').state['position'][2] for pen in pens for i in (1, 2, 3)), unit='m')
        ctx.run(2.)
        minimum = min(row['lowest_ball_center_z_m'] for row in ctx.rows)
        final = {pen.name: [float(pen.body(f'cap_ball{i}').state['position'][2]) for i in (1, 2, 3)]
                 for pen in pens}
        result = {'asset': str(args.asset.resolve()), 'physics_dt_s': .004, 'shelf_thickness_m': .0035,
                  'ball_radius_m': .0008, 'inputs': inputs, 'minimum_ball_center_z_m': minimum,
                  'final_ball_center_z_m': final, 'rows': ctx.rows}
        (args.output / 'thin-shelf.json').write_text(json.dumps(result, indent=2) + '\n')
        assert np.isfinite(minimum) and minimum >= -.0001, f'ball crossed shelf: z={minimum:g} m'
        failed = False
        print('PEN_THIN_SHELF_CHECK_COMPLETED', flush=True)
    except Exception:
        (args.output / 'thin-shelf.error').write_text(traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        if ctx is not None:
            ctx.close()
        app.close(exit_code=int(failed))


if __name__ == '__main__':
    main()
