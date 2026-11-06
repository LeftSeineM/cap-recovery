"""Evaluate isolated assets, recording every seed for later rendering.

Formal: --config configs/asset_grasp_<scene>.py (full clean checkout only).
Development: --dev [--name LABEL] (--config CONFIG [--items PATH ...] | --items PATH ... --robot ROBOT)
[--seeds 0 1] [--video].
Replay: render --result RESULT/objects/CATEGORY/ID --seed 0 --output VIDEO.mp4.
Compare: compare --baseline BASELINE --run RUN --output OUTPUT.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
TRIALS = 20
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from table_1000.common.batch import add_arguments, job_count
    from table_1000.experts.evals.asset_grasp.runner import run
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path,
                        help='test profile: scene asset list, exclusions, robot and trials (configs/asset_grasp_*.py)')
    parser.add_argument('--items', nargs='+', help='asset paths, e.g. objects/soldering_iron/000000; '
                        'with --config a subset of its assets, without it any built asset')
    parser.add_argument('--robot', help='robot (overrides the config; required without --config)')
    parser.add_argument('--physics', choices=['isaac-cpu', 'isaac-gpu'], default='isaac-cpu')
    parser.add_argument('--trials', type=int, help=f'random placements per asset (default config TRIALS, else {TRIALS})')
    parser.add_argument('--seeds', nargs='+', help='explicit seeds or a:b ranges instead of --trials')
    parser.add_argument('--center', nargs=2, type=float, help='world XY; default WORK + (0.05, 0) m')
    parser.add_argument('--region-size', type=float, default=.10, help='square side length in metres')
    parser.add_argument('--diagnostic-shake', action='store_true', help='append the optional vigorous wrist diagnostic')
    parser.add_argument('--collect-preferences', action='store_true',
                        help='reset failed holds and try remaining parts to collect extra grasp preference data')
    parser.add_argument('--video', action='store_true', help='record every requested seed live (requires --dev)')
    parser.add_argument('--seeds-per-worker', type=int,
                        help='maximum seeds of one asset per process, for balancing long candidate searches')
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets')
    parser.add_argument('--dev', action='store_true', help='development run under runs/grasp/<run_name>')
    parser.add_argument('--name', help='development run label')
    parser.add_argument('--output', type=Path, help='batch directory; --dry-run may be followed by a real run here')
    add_arguments(parser)
    args = parser.parse_args(argv)
    args.assets = args.assets.resolve()
    if args.config:
        args.config = args.config.resolve()
    job_count(parser, args)
    run(args, ROOT, parser.error)


def render_main(argv):
    from table_1000.common.batch import add_arguments, job_count
    from table_1000.experts.evals.asset_grasp.runner import render_run
    parser = argparse.ArgumentParser(description='Render one recorded asset/seed without running physics')
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets')
    add_arguments(parser)
    args = parser.parse_args(argv)
    job_count(parser, args)
    render_run(args, ROOT)


if __name__ == '__main__':
    if sys.argv[1:2] == ['compare']:
        from table_1000.experts.evals.asset_grasp.comparison import main as compare_main
        compare_main(sys.argv[2:])
    elif sys.argv[1:2] == ['render']:
        render_main(sys.argv[2:])
    else:
        main()
