"""Build USDZ physical assets in the Isaac Python environment."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from table_1000.physics.build import main
from table_1000.common.diagnostics import dump_stacks_every
if __name__ == '__main__':
    dump_stacks_every()
    main()
