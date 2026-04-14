import sys
import pandas as pd 
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))



def main():
    df = pd.read_csv(f"{OUTPUT_DIR}/plda_distribution.csv")
    print(df.sample(10))
    return None


if __name__ == ("__main__"):
    main()
