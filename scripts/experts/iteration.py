"""Expert iteration: the website, or its steps by hand.

site [--port 60011] [--python PY]: serve runs/expert (see table_1000/experts/audit/site.py); lines
    are indexed on demand and media rendered from the page.
index LINE: (re)build LINE/.iteration/index.json, LINE = runs/expert/<scene>/<robot>/<NNNN[-legacy-MMDD]>.
finish-media DIR: publish a render batch's previews into its line's .iteration/media.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from table_1000.experts.audit import iteration, site
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('site', add_help=False)
    index = commands.add_parser('index')
    index.add_argument('line', type=Path)
    index.add_argument('--seed', type=int, default=0)
    finish = commands.add_parser('finish-media')
    finish.add_argument('batch', type=Path)
    finish.add_argument('--ready-only', action='store_true', help='publish already completed clips while rendering continues')
    finish.add_argument('--overwrite', action='store_true', help='atomically refresh previously published previews')
    finish.add_argument('--kinds', nargs='+', choices=('still', 'clip'), default=('still', 'clip'))
    args, rest = parser.parse_known_args(argv)
    if args.command == 'site':
        return site.main(rest)
    if rest:
        parser.error(f'unrecognized arguments: {" ".join(rest)}')
    if args.command == 'index':
        value = iteration.line_index(args.line, seed=args.seed, progress=lambda done, total, batch: print(f'{done + 1}/{total} {batch}', flush=True))
        print(args.line/iteration.CACHE/'index.json', len(value['steps']), 'steps')
    else:
        print(iteration.finish_media(args.batch, ready_only=args.ready_only, overwrite=args.overwrite, kinds=args.kinds), 'media files published')


if __name__ == '__main__':
    main()
