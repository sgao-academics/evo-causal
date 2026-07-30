"""
run_all.py — One-command replication of "The Phylogenetic Wall" paper.

Stages:
  1. Verify PIC implementation (5 diagnostic tests)
  2. Verify results checkpoint (master_all.json)
  3. Generate all 3 figures (fig1–fig3)
  4. Validate outputs

Usage:  python run_all.py
Output: figures/*.pdf  (3 figures)

Requirements: numpy, matplotlib, scipy  (pip install numpy matplotlib scipy)
Full re-run of experiments requires: pip install causalscale
and takes ~10 hours on GPU.
"""
import sys, os, json, time, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
RES_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def stage(n, title):
    print(f"\n{'='*60}\n  STAGE {n}: {title}\n{'='*60}")


def run_script(relpath, desc):
    script = os.path.join(ROOT, relpath)
    print(f"\n  [{relpath}]")
    try:
        r = subprocess.run([sys.executable, script], cwd=ROOT,
                           capture_output=True, text=True, timeout=120)
        ok = r.returncode == 0
        if ok:
            print(f"  PASS — {len(r.stdout)} bytes output")
        else:
            print(f"  FAIL (code {r.returncode})")
            if r.stderr:
                print(f"  {r.stderr[:300]}")
        return ok
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main():
    t0 = time.time()
    print("=" * 60)
    print("  EVO-CAUSAL REPLICATION")
    print("  Paper: The Phylogenetic Wall")
    print(f"  Python: {sys.version.split()[0]}")
    print("=" * 60)

    # Stage 1
    stage(1, "PIC Implementation Verification")
    pic_ok = run_script("scripts/felsenstein_pic.py", "5 diagnostic tests")
    if not pic_ok:
        print("\n  FATAL: PIC verification failed. Aborting.")
        sys.exit(1)

    # Stage 2
    stage(2, "Results Checkpoint Verification")
    ckpt = os.path.join(RES_DIR, "master_all.json")
    if not os.path.exists(ckpt):
        print(f"  MISSING: {ckpt}")
        print("  Run 'python scripts/run_master_all.py' first (~10h on GPU)")
        sys.exit(1)
    with open(ckpt, "r", encoding="utf-8") as f:
        data = json.load(f)
    sections = sum(1 for s in ["p1","p2","p3","p4"] if s in data)
    sample = data.get("p2", {}).get("d=30_K=0.02", {})
    pic_f1 = sample.get("pic_f1", 0)
    print(f"  Checkpoint: {os.path.getsize(ckpt)/1024:.0f} KB, sections={sections}/4 present")
    print(f"  Sanity PIC(d=30,α=0.02): F1={pic_f1} (expected ~0.305)")
    if abs(pic_f1 - 0.305) > 0.02:
        print("  WARNING: unexpected F1 value")

    # Stage 3
    stage(3, "Generate Figures")
    figs = [
        ("scripts/figures/gen_fig1_phylogenetic_wall.py", "Fig1"),
        ("scripts/figures/gen_fig2_ct_mechanism.py", "Fig2"),
        ("scripts/figures/gen_fig3_method_landscape.py", "Fig3"),
    ]
    all_ok = True
    for path, label in figs:
        ok = run_script(path, label)
        all_ok = all_ok and ok

    # Stage 4
    stage(4, "Output Validation")
    for fn, expected_min in [("fig1_phylogenetic_wall.pdf", 50),
                              ("fig2_ct_mechanism.pdf", 30),
                              ("fig3_method_landscape.pdf", 30)]:
        fpath = os.path.join(FIG_DIR, fn)
        if not os.path.exists(fpath):
            print(f"  MISSING: {fn}")
            all_ok = False
        else:
            kb = os.path.getsize(fpath) / 1024
            status = "OK" if kb > expected_min else "TOO SMALL"
            print(f"  {fn}: {kb:.0f} KB [{status}]")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    if all_ok:
        print(f"  REPLICATION COMPLETE in {elapsed:.0f}s. 3 figures ready.")
    else:
        print(f"  WARNING: some outputs missing. Check above.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
