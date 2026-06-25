import pandas as pd 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
t7_path = "/Volumes/T7/projs/jab_speeches"

aus_df = pd.read_parquet(f"{t7_path}/Czechia_Data_Translated_Opus_Metadata.parquet")
test_df = pd.read_csv(f"{BASE_DIR}/data/parlam/ParlaMint-CZ_extracted.csv")
def main():
    print(aus_df.info())
    print(test_df.info())
    pass

if __name__ == "__main__":
    main()
