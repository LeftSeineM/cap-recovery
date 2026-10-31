"""Output and provenance of isolated asset grasp tests (no simulator required)."""

from pathlib import Path
import shutil

from table_1000.common.batch import run_name


def prepare(root, *, dev, label, checkout):
    if not dev and (checkout is None or checkout.endswith('+dirty')):
        raise ValueError('正式运行要求干净的 Git 提交；未提交的开发运行请加 --dev')
    if label and (Path(label).name != label or label in {'.', '..'}):
        raise ValueError('--name 必须是单个目录名')
    name = run_name(checkout, label)
    output = root / ('runs/grasp' if dev else 'outputs/asset-grasp') / name
    output.mkdir(parents=True, exist_ok=False)
    return name, output


def require_mode(*, dev, asset, seeds, video, label, trials, configured_trials):
    if not dev and (asset is not None or seeds is not None or video or label or trials != configured_trials):
        raise ValueError('部分资产、seed、--video、--name 或更改试验次数需要 --dev；正式运行须跑完整配置')


def acceptance_path(assets, asset, robot, backend):
    return assets / asset / 'grasp_test' / robot / backend


def publish(source, assets, asset, robot, backend):
    """Replace the entire current result, only after the staged result is complete."""
    target = acceptance_path(assets, asset, robot, backend)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(source), target)
    return target


def video_seed(reports):
    """First failing seed, otherwise the smallest seed (zero in a full run)."""
    ordered = sorted(reports, key=lambda report: report['seed'])
    return next((report['seed'] for report in ordered if report['status'] != 'success'), ordered[0]['seed'])
