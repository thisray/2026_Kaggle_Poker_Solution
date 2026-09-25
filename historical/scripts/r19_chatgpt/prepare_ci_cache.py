"""Read two prior R3 tables; no raw competition download or duplicate packaging."""
import os,sys
from pathlib import Path
import pandas as pd
bundle=Path(os.environ['R3_BUNDLE']);out=Path(os.environ.get('R19_WORK','./r19_work'));out.mkdir(parents=True,exist_ok=True)
for name in ['t5_dev_seq','t4_wrong_vs_hit']:
    path=bundle/'artifacts/tables'/f'{name}.parquet'
    try: d=pd.read_parquet(path)
    except ImportError:
        sys.path.insert(0,str(bundle/'scripts/round15_reference'))
        from read_flat_parquet import read_flat
        d=read_flat(path)
    d.to_pickle(out/f'{name}.pkl');print(name,d.shape,flush=True)
