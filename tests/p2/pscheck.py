import pathlib, subprocess, sys
r = subprocess.run(
    ["wmic", "process", "where", 'Name="python.exe"',
     "get", "ProcessId,KernelModeTime,UserModeTime,CommandLine", "/format:list"],
    capture_output=True, text=True,
)
pathlib.Path(r"d:\taichi\tests\p2\ps3.txt").write_text(r.stdout, encoding="utf-8")
print(r.stdout[-2000:])
