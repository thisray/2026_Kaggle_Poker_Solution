"""Round-8 eval export with the score features required by the combo residual."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True); p.add_argument('--np-dir', required=True)
    p.add_argument('--local-index', required=True); p.add_argument('--nn'); p.add_argument('--out', required=True)
    a = p.parse_args()
    c = pd.read_parquet(a.input).rename(columns={'sl': 'slot'})
    nn = pd.read_parquet(a.nn).rename(columns={'sl': 'slot'})
    c = c.merge(nn[['slot', 'h', 'lg_nn_cal']], on=['slot', 'h'], how='left', validate='one_to_one')
    assert c.lg_nn_cal.notna().all()
    c['u_rr'] = c.u0 + 0.5 * c.lin
    c['nn_contrib'] = 0.1 * c.lg_nn_cal
    c['u_r5b'] = c.u_rr + c.nn_contrib
    c['lin_contrib'] = 0.5 * c.lin
    c['sc_r5'] = c['sc']; c['t1_score'] = c['t1']; c['s1_stage1'] = c['s1']
    c['gen_logit'] = lg(c.s1.values)
    c['gen_rank_pct'] = c.gen_rank_all
    c['rank_u_r5b'] = c.groupby('slot')['u_r5b'].rank(ascending=False, method='first')
    r = Path(a.np_dir)
    hidx = pd.read_parquet(r / 'hand_index.parquet').set_index('hi')
    pidx = pd.read_parquet(r / 'player_index.parquet').set_index('pi')
    loc = pd.read_parquet(a.local_index); members = loc.set_index(['pool', 'local']).player_gi
    pool = c.slot.to_numpy() // 900; al = (c.slot.to_numpy() % 900) // 30; bl = c.slot.to_numpy() % 30
    ai = members.reindex(pd.MultiIndex.from_arrays([pool, al])).to_numpy()
    bi = members.reindex(pd.MultiIndex.from_arrays([pool, bl])).to_numpy()
    assert not (pd.isna(ai).any() or pd.isna(bi).any())
    out = pd.DataFrame({'slot': c.slot, 'pool': pool,
        'pair_player_lo': pidx.player_id.reindex(ai).to_numpy(),
        'pair_player_hi': pidx.player_id.reindex(bi).to_numpy(),
        'hand_id': hidx.hand_id.reindex(c.h).to_numpy(), 'u_r5b': c.u_r5b})
    keep = ['sc_r5', 'u0', 'u_rr', 'lin_contrib', 'nn_contrib', 't1_score', 's1_stage1',
            'gen_logit', 'gen_rank_pct', 'rank_u_r5b']
    for k in keep:
        out[k] = c[k].to_numpy()
    out.to_csv(a.out, index=False)
    print('exported', len(out), 'rows,', out.slot.nunique(), 'pairs, cols', out.shape[1])

if __name__ == '__main__':
    main()
