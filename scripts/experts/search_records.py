"""Statistics, portable cases, independent solve replay and offline search visualization."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from table_1000.common.serialization import dumps


def main(argv=None):
    from table_1000.experts.audit.records import statistics, export_case, media_batch
    from table_1000.experts.audit.view import export_view
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    stats = commands.add_parser('stats')
    stats.add_argument('batches', type=Path, nargs='+')
    stats.add_argument('--output', type=Path)
    view = commands.add_parser('view')
    view.add_argument('batch', type=Path)
    view.add_argument('--seed', type=int, required=True)
    view.add_argument('--repeat', type=int, help='one execution; default includes every repeat')
    view.add_argument('--output', type=Path, required=True)
    view.add_argument('--media', action='store_true', help='export a normal batch for render.py')
    case = commands.add_parser('case')
    case.add_argument('batches', type=Path, nargs='+', help='all attempts of this step, in order')
    case.add_argument('--seed', type=int, required=True)
    case.add_argument('--repeat', type=int, default=0)
    case.add_argument('--failure', type=int, help='raw planning event ID; default exports a b step-root case')
    case.add_argument('--output', type=Path, required=True)
    replay = commands.add_parser('replay')
    replay.add_argument('batch', type=Path)
    replay.add_argument('--seed', type=int, required=True)
    replay.add_argument('--repeat', type=int, default=0)
    replay.add_argument('--failure', type=int, required=True)
    replay.add_argument('--output', type=Path, required=True)
    replay.add_argument('--run', action='store_true', help='execute the printed run_expert command (submit this tool through gq)')
    thumbs = commands.add_parser('thumbnails')
    thumbs.add_argument('view', type=Path)
    segments = commands.add_parser('segments')
    segments.add_argument('batch', type=Path)
    segments.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'stats':
        value = statistics(args.batches)
        text = dumps(value, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + '\n')
        print(text)
    elif args.command == 'view':
        value = export_view(args.batch, args.seed, args.output, media=args.media, repeat=args.repeat)
        print(args.output / 'index.html', len(value['nodes']), 'states', len(value['edges']), 'actions')
    elif args.command == 'case':
        export_case(args.batches, args.seed, args.output, args.failure, repeat=args.repeat)
        print(args.output / 'summary.json')
    elif args.command == 'replay':
        case_dir = args.output / 'case'
        result = export_case([args.batch], args.seed, case_dir, args.failure, repeat=args.repeat)
        from table_1000.common.batch import worker_gpu
        command = [sys.executable, str(ROOT / 'scripts/scenes/run_expert.py'), result['scene'], '--dev',
                   '--robot', result['robot'], '--expert', result['expert'], '--physics', result['physics'],
                   '--from', str(case_dir.resolve()), '--seeds', str(args.seed), '--repeats', '1',
                   '--skill-attempts', '1', '--output', str((args.output / 'runs').resolve()), '--jobs', '1', '--retries', '0',
                   '--gpus', str(worker_gpu(None))]
        print(shlex.join(command), flush=True)
        if args.run:
            subprocess.run(command, check=True)
    elif args.command == 'thumbnails':
        directory = args.view / 'media/stills'
        directory.mkdir(parents=True, exist_ok=True)
        graph = json.loads((args.view / 'graph.json').read_text())
        for node in graph['nodes']:
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(args.view / f"media/videos/{node['media']}/merge.mp4"),
                            '-frames:v', '1', str(directory / f"{node['media']}.jpg")], check=True)
    elif args.command == 'segments':
        value = statistics([args.batch])
        for index, clip in enumerate(value['clips']):
            source = Path(clip['batch'])
            report = clip['report']
            media_batch(source, report, [dict(start=clip['control_start'], end=clip['control_end'])], args.output / str(index))
        print(len(value['clips']), 'render.py batches:', args.output)


if __name__ == '__main__':
    main()
