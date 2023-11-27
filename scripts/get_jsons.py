from pathlib import Path
from subprocess import run

output_paths = []

for path in Path(".").rglob("output.pkl"):
    output_paths.append(path)

for path in output_paths:
    json_path = path.parent / "output.json"
    if json_path.exists():
        continue
    print(f"Converting {path} to {json_path}")
    run(
        [
            "python3",
            "scripts/pkl_to_json.py",
            "--pkl_file",
            str(path),
            "--json_file",
            str(json_path),
        ]
    )
