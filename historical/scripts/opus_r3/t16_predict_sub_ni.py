"""ni-color: predict policy_v2 on the partner-card feature rows exported by t15 (6 threads; the host is shared)."""
import numpy as np, lightgbm as lgb, time, sys
t0 = time.time()
b = lgb.Booster(model_file="policy_v2.txt"); print(f"[{time.time()-t0:6.1f}s] model loaded", flush=True)
for nm in sys.argv[1:]:
    X = np.load(f"sub_X_{nm}.npy", mmap_mode="r"); n = len(X); out = np.zeros((n, 4), np.float32); B = 250_000
    for i in range(0, n, B):
        out[i:i + B] = b.predict(np.asarray(X[i:i + B]), num_threads=6)
        if i == 0 or (i // B) % 4 == 0: print(f"[{time.time()-t0:6.1f}s] {nm} {i + B}/{n}", flush=True)
    np.save(f"q_sub_{nm}.npy", out); print(f"[{time.time()-t0:6.1f}s] {nm} saved", flush=True)
