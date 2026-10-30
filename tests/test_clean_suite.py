import sys
import os
import subprocess
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
M35_EXE = str(BASE_DIR / 'solvers/m3_5_v2' / ('m3_5_v2.exe' if os.name == 'nt' else 'm3_5_v2'))
KISSAT_F52_EXE = str(BASE_DIR / 'solvers/neuro_cdcl_f52' / ('neuro_cdcl.exe' if os.name == 'nt' else 'neuro_cdcl'))
KISSAT_STOCK_EXE = str(BASE_DIR / 'solvers/kissat_stock' / ('kissat.exe' if os.name == 'nt' else 'kissat'))
CADICAL_STOCK_EXE = str(BASE_DIR / 'solvers/cadical_stock' / ('cadical.exe' if os.name == 'nt' else 'cadical'))

DATA_DIR = BASE_DIR / 'data'
EXT_REPOS_DIR = BASE_DIR / 'external_repos'

def test_m35_solver_binary_exists():
    assert os.path.exists(M35_EXE), f'M3.5 v2 solver binary {M35_EXE} must exist'

def test_m35_sat_solving_correctness():
    cnfs = list((DATA_DIR / 'rlaf/coloring_400').glob('*.cnf'))
    assert len(cnfs) > 0, 'RLAF Coloring 400 benchmarks must exist'
    
    p = subprocess.run([M35_EXE, str(cnfs[0])], capture_output=True, text=True, timeout=10)
    assert p.returncode in [0, 10, 20], 'Solver must exit with standard code'
    assert 'Decisions:' in p.stdout, 'Solver output must include decision telemetry'
    assert 'SATISFIABLE' in p.stdout, 'Instance must solve to SATISFIABLE'

def test_neuro_cdcl_kissat_binary_exists():
    assert os.path.exists(KISSAT_F52_EXE), f'NeuroCDCL Kissat binary {KISSAT_F52_EXE} must exist'

def test_neuro_cdcl_kissat_solving_correctness():
    cnfs = list((DATA_DIR / 'rlaf/coloring_400').glob('*.cnf'))
    p = subprocess.run([KISSAT_F52_EXE, str(cnfs[0])], capture_output=True, text=True, timeout=10)
    assert p.returncode in [0, 10, 20], 'Kissat must exit with SAT code 10'
    assert 's SATISFIABLE' in p.stdout, 'Instance must solve to SATISFIABLE'

def test_kissat_stock_binary_exists():
    assert os.path.exists(KISSAT_STOCK_EXE), f'Kissat Stock binary {KISSAT_STOCK_EXE} must exist'
    p = subprocess.run([KISSAT_STOCK_EXE, '--version'], capture_output=True, text=True, timeout=5)
    assert p.returncode == 0, 'Kissat stock must return exit code 0 on --version'
    assert '4.0.1' in p.stdout, 'Kissat version must be 4.0.1'

def test_cadical_stock_binary_exists():
    assert os.path.exists(CADICAL_STOCK_EXE), f'CaDiCaL Stock binary {CADICAL_STOCK_EXE} must exist'
    p = subprocess.run([CADICAL_STOCK_EXE, '--version'], capture_output=True, text=True, timeout=5)
    assert p.returncode == 0, 'CaDiCaL stock must return exit code 0 on --version'
    assert '2.1.3' in p.stdout, 'CaDiCaL version must be 2.1.3'

def test_weights_file_exists():
    w_m35 = BASE_DIR / 'models/weights/m3_5_v2/structural_mlp_weights.bin'
    assert w_m35.exists(), 'M3.5 V2 Structural MLP weights must exist'
    assert w_m35.stat().st_size == 648, 'Structural MLP weights must be exactly 648 bytes'

    w_f52 = BASE_DIR / 'models/weights/folder52_kissat/structural_mlp_weights.h'
    assert w_f52.exists(), 'Folder 52 Kissat C99 weights header must exist'

    w_teacher = BASE_DIR / 'models/weights/folder52_kissat/best_causal_teacher.pt'
    assert w_teacher.exists(), 'Folder 52 Causal Teacher model must exist'

    inv = BASE_DIR / 'models/WEIGHTS_INVENTORY.md'
    assert inv.exists(), 'Weights inventory document must exist'

def test_benchmarks_present():
    rlaf_cnfs = list((DATA_DIR / 'rlaf').rglob('*.cnf'))
    imitsat_cnfs = list((DATA_DIR / 'imitsat').rglob('*.cnf'))
    neuroback_cnfs = list((DATA_DIR / 'neuroback').rglob('*.cnf'))
    assert len(rlaf_cnfs) >= 60, f'Expected >=60 RLAF CNFs, got {len(rlaf_cnfs)}'
    assert len(imitsat_cnfs) >= 200, f'Expected >=200 ImitSAT CNFs, got {len(imitsat_cnfs)}'
    assert len(neuroback_cnfs) >= 500, f'Expected >=500 NeuroBack CNFs, got {len(neuroback_cnfs)}'

def test_audit_artifacts_present():
    pdf = BASE_DIR / 'docs/papers_published/DAC203_DEFINITIVE_ML4SAT_OFFICIAL_AUDIT.pdf'
    assert pdf.exists(), 'Official Master PDF audit report must exist'
    assert pdf.stat().st_size > 10000, 'Master PDF must be non-empty'

def test_external_repositories_present():
    for repo in ['rlaf', 'imitsat', 'neuroback']:
        rdir = EXT_REPOS_DIR / repo
        assert rdir.exists(), f'External repo {repo} must exist at {rdir}'
        assert len(list(rdir.glob('*'))) > 0, f'External repo {repo} must not be empty'

def test_industrial_manifest_and_script_dry_run():
    manifest = BASE_DIR / 'data/industrial_400/UNSEEN_400_MANIFEST.csv'
    assert manifest.exists(), 'Industrial 400 manifest must exist'
    assert manifest.stat().st_size > 50000, 'Manifest must contain 400 indexed instances'

    script = BASE_DIR / 'scripts/run_industrial_400_tournament.py'
    assert script.exists(), 'Tournament runner script must exist'

    p = subprocess.run([sys.executable, str(script), '--dry-run'], capture_output=True, text=True, timeout=10)
    assert p.returncode == 0, f'Tournament script dry-run failed: {p.stderr}'
    assert '[DRY RUN COMPLETE]' in p.stdout, 'Tournament dry-run must finish successfully'
