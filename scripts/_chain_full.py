"""Full chain: reviewer -> overnight -> bonus_fill (10-hour fill)"""
import subprocess, sys, os, time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PYTHON = r'D:\Miniconda3\python.exe'
BASE = r'C:\Users\高帅东\Desktop\evo_causal\scripts'
RESULTS = r'C:\Users\高帅东\Desktop\evo_causal\results'

scripts = [
    ('run_reviewer_experiments.py', 'reviewer_chain_log.txt'),
    ('run_overnight_massive.py', 'overnight_chain_log.txt'),
    ('run_bonus_fill.py', 'bonus_chain_log.txt'),
]

for script, log_name in scripts:
    path = os.path.join(BASE, script)
    log_path = os.path.join(RESULTS, log_name)
    print(f"\n{'='*60}")
    print(f"START: {script}  [{time.strftime('%H:%M:%S')}]")
    print(f"{'='*60}")
    sys.stdout.flush()

    with open(log_path, 'w', encoding='utf-8') as f:
        result = subprocess.run(
            [PYTHON, path],
            stdout=f, stderr=subprocess.STDOUT, cwd=BASE,
        )
    code = result.returncode
    if code != 0:
        print(f"WARNING: {script} exit {code}")
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [l.rstrip() for l in f if l.strip()]
        for l in lines[-5:]:
            print(f"  {l}")
    else:
        print(f"DONE: {script}  [{time.strftime('%H:%M:%S')}]")

print(f"\nALL DONE at {time.strftime('%H:%M:%S')}")
