"""Out-of-sample CDI evaluation: fit CDI params on a TRAIN path (the Rust search),
then roll those FIXED params out on many DISJOINT TEST paths (faithful Python port
of env.step_state / epoch_cost and capped_dual_index_action) and report mean+/-std.
Compare to the exact-expectation bounded-DP optimum -> honest CDI gap-to-optimum."""
import invman_rust as ir
import numpy as np

CR=100.0; H=5.0; CAP=12

def dvals(kind): return (list(range(0,5)),0,4) if kind=="U04" else (list(range(0,9)),0,8)

def fit_cdi(lr,ce,b,kind,seed,hz=8000,tub=28):
    vals,low,high=dvals(kind); md=float(np.mean(vals))
    state=[round((lr+1)*md)]+[0]*(lr-1)
    d=np.random.default_rng(seed).choice(vals,size=hz).astype(int).tolist()
    cdi,_=ir.dual_sourcing_capped_dual_index_search_from_demands(
        state=state,demands=d,regular_max_order_size=CAP,expedited_max_order_size=CAP,
        regular_order_cost=CR,expedited_order_cost=ce,holding_cost=H,shortage_cost=b,
        warm_up_periods_ratio=0.2,target_upper_bound=tub,top_k=1)
    return cdi[0],cdi[1],cdi[2]  # s_e, s_r, cap_r

def cdi_action(red, s_e, s_r, cap_r):
    eip=red[0]; rip=sum(red)
    exp=min(max(s_e-max(eip,0),0), CAP)
    des=max(s_r-(max(rip,0)+exp),0)
    return min(des,cap_r,CAP), exp

def rollout(params,lr,ce,b,kind,seed,hz=20000,warm=0.2):
    vals,low,high=dvals(kind); md=float(np.mean(vals))
    red=[round((lr+1)*md)]+[0]*(lr-1)
    d=np.random.default_rng(seed).choice(vals,size=hz)
    s_e,s_r,cap_r=params
    warmn=int(hz*warm); tot=0.0; cnt=0
    for t in range(hz):
        dem=int(d[t])
        qr,qe=cdi_action(red,s_e,s_r,cap_r)
        end=red[0]+qe-dem
        c=CR*qr+ce*qe+H*max(end,0)+b*max(-end,0)
        # step
        if len(red)==1:
            red=[red[0]+qe-dem+qr]
        else:
            ns=[end+red[1]]+list(red[2:])+[qr]
            red=ns
        if t>=warmn:
            tot+=c; cnt+=1
    return tot/cnt

def dp(lr,ce,b,kind,lo,hi):
    _,low,high=dvals(kind)
    r=ir.dual_sourcing_bounded_average_cost_optimal_summary(
        regular_lead_time=lr,regular_order_cost=CR,expedited_order_cost=ce,holding_cost=H,
        shortage_cost=b,regular_max_order_size=CAP,expedited_max_order_size=CAP,
        demand_low=low,demand_high=high,inventory_lower=lo,inventory_upper=hi,
        tolerance=1e-8,max_iterations=400)
    return r["average_cost"]

def assess(lr,ce,b,kind,box,train_seed=123,test_seeds=(1000,1001,1002,1003,1004,1005,1006,1007)):
    params=fit_cdi(lr,ce,b,kind,train_seed)
    costs=[rollout(params,lr,ce,b,kind,sd) for sd in test_seeds]
    costs=np.array(costs)
    dopt=dp(lr,ce,b,kind,*box)
    gap=100.0*(costs.mean()/dopt-1.0)
    print(f"cell lr={lr} ce={ce} b={b} {kind}: CDI params(s_e,s_r,cap_r)={params}")
    print(f"  OOS CDI cost = {costs.mean():.3f} +/- {costs.std():.3f}  (n={len(costs)} disjoint test paths, hz=20000)")
    print(f"  exact DP opt = {dopt:.3f}  box={box}")
    print(f"  CDI gap-to-OPTIMUM = {gap:+.4f}%   (#paths above DP: {int((costs>dopt).sum())}/{len(costs)})")
    return costs.mean(),costs.std(),dopt,gap,params

if __name__=="__main__":
    print("=== Calibration: Gijs l_r=2 ce=110 b=495 U04 (published CDI gap 0.03-0.11%) ===")
    assess(2,110.0,495.0,'U04',(-24,48))
    print("=== HARDEST cell: l_r=2 ce=110 b=50 U08 ===")
    assess(2,110.0,50.0,'U08',(-40,72))
