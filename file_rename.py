from pathlib import Path

folder = Path(r"D:\CV5570\CV5570_3D_Reconstruction_Assignment\Reconstruction Workspace\images")

files = sorted(
    [f for f in folder.iterdir() if f.is_file()],
    key=lambda f: f.name.lower()
)

for i, file in enumerate(files, start=1):
    temp_path = folder / f"__temp_rename_{i}__{file.suffix}"
    file.rename(temp_path)

for i in range(1, len(files) + 1):
    temp_path = folder / f"__temp_rename_{i}__{files[i - 1].suffix}"
    final_path = folder / f"{i}{files[i - 1].suffix}"
    temp_path.rename(final_path)

print(f"Successfully renamed {len(files)} files.")