"""Leaner l_r=2 sweep: drop b=5 (DP-unreachable: no penalty floor, box never
plateaus), cap ladder at 3 boxes, looser plateau tol (1e-4 relative ~ 0.02 cost
on ~220). Heuristic horizon 4000."""
import invman_rust as ir
import numpy as np
import sys, time

CR=100.0; H=5.0; CAP=12

def path(seed,hz,vals):
    return np.random.default_rng(seed).choice(vals,size=hz).astype(int).tolist()

def dvals(kind):
    return (list(range(0,5)),0,4) if kind=="U04" else (list(range(0,9)),0,8)

def heur(lr,ce,b,kind,seed=123,hz=4000,tub=28):
    vals,low,high=dvals(kind); md=float(np.mean(vals))
    state=[round((lr+1)*md)]+[0]*(lr-1)
    d=path(seed,hz,vals)
    c=dict(state=state,demands=d,regular_max_order_size=CAP,expedited_max_order_size=CAP,
           regular_order_cost=CR,expedited_order_cost=ce,holding_cost=H,shortage_cost=b,
           warm_up_periods_ratio=0.2,target_upper_bound=tub,top_k=1)
    si,_=ir.dual_sourcing_single_index_search_from_demands(**c)
    di,_=ir.dual_sourcing_dual_index_search_from_demands(**c)
    cdi,_=ir.dual_sourcing_capped_dual_index_search_from_demands(**c)
    tbs,_=ir.dual_sourcing_tailored_base_surge_search_from_demands(**c)
    return dict(SI=si[-1],DI=di[-1],CDI=cdi[-1],TBS=tbs[-1])

def dp(lr,ce,b,low,high,lo,hi,mi=400):
    r=ir.dual_sourcing_bounded_average_cost_optimal_summary(
        regular_lead_time=lr,regular_order_cost=CR,expedited_order_cost=ce,
        holding_cost=H,shortage_cost=b,regular_max_order_size=CAP,expedited_max_order_size=CAP,
        demand_low=low,demand_high=high,inventory_lower=lo,inventory_upper=hi,
        tolerance=1e-8,max_iterations=mi)
    return r["average_cost"],r["iterations"]

def vdp(lr,ce,b,kind,tol=1e-4):
    vals,low,high=dvals(kind)
    ladder=[(-12,24),(-24,48),(-40,72)] if kind=="U04" else [(-24,48),(-40,72),(-64,108)]
    tr=[];prev=None
    for (lo,hi) in ladder:
        t0=time.time();val,it=dp(lr,ce,b,low,high,lo,hi);dt=time.time()-t0
        tr.append((lo,hi,round(val,4),f"it={it},{dt:.1f}s"))
        if prev is not None and abs(val-prev)/max(abs(prev),1e-9)<=tol:
            return val,(lo,hi),True,tr
        prev=val
    return prev,ladder[-1],False,tr

def cell(lr,ce,b,kind):
    h=heur(lr,ce,b,kind);d,box,v,tr=vdp(lr,ce,b,kind)
    g=None if d is None else 100.0*(h["CDI"]/d-1.0)
    return dict(lr=lr,ce=ce,b=b,kind=kind,**h,dp=d,box=box,valid=v,gap=g,trace=tr)

def emit(r):
    dp_s=f"{r['dp']:.3f}" if r['dp'] is not None else "  --"
    box_s=f"{r['box']}" if r['box'] else " none"
    g_s=f"{r['gap']:+.4f}" if r['gap'] is not None else "  --"
    print(f"{r['lr']:>2} {r['ce']:>4} {r['ce']-CR:>4.0f} {r['b']:>5.0f}  {r['kind']:>4} | {r['SI']:8.2f} {r['DI']:8.2f} {r['CDI']:8.2f} {r['TBS']:8.2f} | {dp_s:>9} {box_s:>10} {str(r['valid']):>5} {g_s:>9}")
    sys.stdout.flush()
