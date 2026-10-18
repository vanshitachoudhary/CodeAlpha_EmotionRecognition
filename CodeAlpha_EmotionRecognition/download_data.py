"""RAVDESS (speech) ko Zenodo se download karta hai. TESS ke liye manual steps print karta hai (Kaggle login chahiye).
Run:  python download_data.py"""
import sys
import urllib.request
import zipfile
from pathlib import Path

RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
TESS_HELP = """
TESS (Toronto Emotional Speech Set) - manual download:
  1. Kaggle pe dataset "Toronto emotional speech set (TESS)" search karo (author: ejlok1) aur download karo
     (ya University of Toronto Dataverse: https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF)
  2. Zip ko extract karke folder ko  data/tess/  mein rakh do (andar OAF_* aur YAF_* folders).
"""


def _progress(blocks, size, total):
    if total > 0:
        print(f"\r  {min(blocks * size, total) / 1e6:6.1f} / {total / 1e6:.1f} MB", end="")


def main(root="data"):
    dest = Path(root) / "ravdess"
    dest.mkdir(parents=True, exist_ok=True)
    if len(list(dest.rglob("*.wav"))) >= 1400:
        print("RAVDESS already present.")
    else:
        zp = dest / "ravdess_speech.zip"
        print(f"Downloading RAVDESS speech (~200 MB) from Zenodo ...")
        try:
            urllib.request.urlretrieve(RAVDESS_URL, zp, _progress)
            print("\nExtracting ...")
            with zipfile.ZipFile(zp) as z:
                z.extractall(dest)
            zp.unlink()
            print(f"RAVDESS ready: {len(list(dest.rglob('*.wav')))} wav files in {dest}")
        except Exception as e:
            print(f"\nAutomatic download failed ({type(e).__name__}: {e}).\n"
                  "Manual: open https://zenodo.org/records/1188976 , download 'Audio_Speech_Actors_01-24.zip', "
                  f"and extract it into {dest}/ (folders Actor_01 ... Actor_24).")
    tess = Path(root) / "tess"
    n = len(list(tess.rglob("*.wav"))) if tess.exists() else 0
    print(f"\nTESS wav files found: {n}" + ("" if n else TESS_HELP))
    print("Next:  python train.py --data data")


if __name__ == "__main__":
    main(*sys.argv[1:2])
