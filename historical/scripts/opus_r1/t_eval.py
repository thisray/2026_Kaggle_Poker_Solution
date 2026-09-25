import numpy as np, itertools, time
from pk_eval import eval_cards, B
rc=np.zeros(13,np.int64); sc=np.zeros(4,np.int64); sm=np.zeros(4,np.int64)
def c(s): return "23456789TJQKA".index(s[0])*4+"cdhs".index(s[1])
def ev(s):
    arr=np.array([c(x) for x in s.split()],np.int64); return eval_cards(arr,len(arr),rc,sc,sm)
tests = [("As Ks Qs Js Ts 2d 3c",8),("Ah 2d 3c 4s 5h 9d Kc",4),("Ah Ad Ac As 2d 3c 4h",7),("Kh Kd Kc 2s 2d 3c 4h",6),("Kh Kd Kc 2s 2d 2c 4h",6),
 ("2h 7h 9h Jh Kh Ad Ac",5),("2h 3d 4c 5s 6h Ad Ac",4),("9h 9d 9c 2s 5d Jc 4h",3),("9h 9d 5c 5s 2d Jc 4h",2),("9h 9d 5c 7s 2d Jc 4h",1),("9h Td 5c 7s 2d Jc 4h",0),
 ("Ah 2h 3h 4h 5h Kd Kc",8)]
for s, cat in tests:
    v = ev(s); assert v//B == cat, (s, v//B, cat)
assert ev("Ah Ad Kc Ks 2d 3c 4h") > ev("Ah Ad Qc Qs Kd 3c 4h")
assert ev("Ah Ad Kc Ks Qd 3c 4h") > ev("Ah Ad Kc Ks Jd 3c 4h")
assert ev("6h 2d 3c 4s 5h 9d Kc") > ev("Ah 2d 3c 4s 5h 9d Kc")
assert ev("Th Td Tc 9s 9h 9d Kc") > ev("9h 9d 9c Ts Th 2d Kc")
assert ev("Th Td Tc 9s 9h 8d 8c") == ev("Th Td Tc 9s 9h 2d 3c")
# brute-force check vs naive 5-card best-of-7 using a simple reference
import random
random.seed(1)
def ref5(cards):
    rs=sorted([x>>2 for x in cards],reverse=True); ss=[x&3 for x in cards]
    fl=len(set(ss))==1
    u=sorted(set(rs),reverse=True)
    st=-1
    if len(u)==5 and u[0]-u[4]==4: st=u[0]
    if set(u)=={12,0,1,2,3}: st=3
    from collections import Counter
    cnt=Counter(rs); grp=sorted(cnt.items(), key=lambda kv:(-kv[1],-kv[0]))
    if fl and st>=0: return (8,st)
    if grp[0][1]==4: return (7,grp[0][0],grp[1][0])
    if grp[0][1]==3 and grp[1][1]==2: return (6,grp[0][0],grp[1][0])
    if fl: return (5,)+tuple(rs)
    if st>=0: return (4,st)
    if grp[0][1]==3: return (3,grp[0][0])+tuple(r for r in rs if r!=grp[0][0])
    if grp[0][1]==2 and grp[1][1]==2: return (2,grp[0][0],grp[1][0],grp[2][0])
    if grp[0][1]==2: return (1,grp[0][0])+tuple(r for r in rs if r!=grp[0][0])
    return (0,)+tuple(rs)
def ref7(cards): return max(ref5(list(cmb)) for cmb in itertools.combinations(cards,5))
bad=0
for it in range(20000):
    x=random.sample(range(52),7); y=random.sample(range(52),7)
    a=eval_cards(np.array(x,np.int64),7,rc,sc,sm); b=eval_cards(np.array(y,np.int64),7,rc,sc,sm)
    ra=ref7(x); rb=ref7(y)
    if (a>b)!=(ra>rb) or (a==b)!=(ra==rb): bad+=1
print("mismatch", bad)
t=time.time(); arr=np.array(random.sample(range(52),7),np.int64)
for _ in range(200000): eval_cards(arr,7,rc,sc,sm)
print("py-loop 200k", time.time()-t)
