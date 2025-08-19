# export_tb_json.py
import os, json, argparse
from tensorboard.backend.event_processing import event_accumulator

def find_run_dirs(logdir: str):
    run_dirs = []
    for dirpath, _, filenames in os.walk(logdir):
        if any(f.startswith("events.out.tfevents.") for f in filenames):
            run_dirs.append(dirpath)
    return sorted(set(run_dirs))

def export_scalars(logdir: str):
    data = {}
    for run_dir in find_run_dirs(logdir):
        run_name = os.path.relpath(run_dir, logdir) or "."
        ea = event_accumulator.EventAccumulator(
            run_dir,
            size_guidance={
                event_accumulator.SCALARS: 0,  # load all scalar points
            },
        )
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        run_payload = {}
        for tag in tags:
            events = ea.Scalars(tag)
            series = [{"wall_time": ev.wall_time, "step": int(ev.step), "value": float(ev.value)}
                      for ev in events]
            run_payload[tag] = series
        data[run_name] = run_payload
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, help="Path to TensorBoard log directory (e.g., ./hashgremlin_logs)")
    ap.add_argument("--out", default="tensorboard_export.json", help="Output JSON file")
    args = ap.parse_args()

    data = export_scalars(args.logdir)
    # Pretty-print but keep file size reasonable
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out} with {len(data)} run(s).")

if __name__ == "__main__":
    main()

