import zipfile
from google.colab import files
REPORT_ZIP = WORK / "gemma_colab_pilot_V3_8_reports.zip"
with zipfile.ZipFile(REPORT_ZIP, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".log", ".py", ".in", ".lock"):
            bundle.write(path, path.name)
    for directory in ("profile", "profile_tools", "tokenizer"):
        for path in sorted((OUTPUT / directory).rglob("*")):
            if path.is_file():
                bundle.write(path, str(path.relative_to(OUTPUT)))
shutil.copyfile(REPORT_ZIP, OUTPUT / REPORT_ZIP.name)
files.download(str(REPORT_ZIP))
print("FP32 weights:", OUTPUT / "weights.npz")
print("Tokenizer files:", OUTPUT / "tokenizer")
print("Finished. Select Runtime > Disconnect and delete runtime when done.")
