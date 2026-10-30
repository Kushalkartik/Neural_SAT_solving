import os
import subprocess
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
M35_EXE = str(BASE_DIR / 'solvers/m3_5_v2' / ('m3_5_v2.exe' if os.name == 'nt' else 'm3_5_v2'))
KISSAT_EXE = str(BASE_DIR / 'solvers/neuro_cdcl_f52' / ('neuro_cdcl.exe' if os.name == 'nt' else 'neuro_cdcl'))
DATA_DIR = BASE_DIR / 'data'

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
    assert os.path.exists(KISSAT_EXE), f'NeuroCDCL Kissat binary {KISSAT_EXE} must exist'

def test_neuro_cdcl_kissat_solving_correctness():
    cnfs = list((DATA_DIR / 'rlaf/coloring_400').glob('*.cnf'))
    p = subprocess.run([KISSAT_EXE, str(cnfs[0])], capture_output=True, text=True, timeout=10)
    assert p.returncode in [0, 10, 20], 'Kissat must exit with SAT code 10'
    assert 's SATISFIABLE' in p.stdout, 'Instance must solve to SATISFIABLE'

def test_weights_file_exists():
    w = BASE_DIR / 'models/weights/structural_mlp_weights.bin'
    assert w.exists(), 'Structural MLP weights binary must exist'
    assert w.stat().st_size == 648, 'Structural MLP weights binary must be exactly 648 bytes'

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
