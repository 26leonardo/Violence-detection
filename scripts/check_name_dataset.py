import tarfile
from pathlib import Path

archive = Path("RWF-2000.tar.gz")

with tarfile.open(archive) as tar:
    files = [m for m in tar.getmembers() if m.isfile()]

    longest = sorted(
        files,
        key=lambda m: len(Path(m.name).name.encode("utf-8")),
        reverse=True,
    )

    print(f"Files: {len(files)}")
    print()

    for member in longest[:10]:
        name = Path(member.name).name
        size = len(name.encode("utf-8"))
        print(f"{size:4d} bytes  {name[:120]}")