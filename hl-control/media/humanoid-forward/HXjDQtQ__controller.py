"""Sampling-based model-predictive controller for the MuJoCo Humanoid (no learning, no gradients).

Control law
-----------
Every control step the controller rebuilds the plant state from ``obs["gym_obs"]`` and plans with an
internal MuJoCo copy of the humanoid (same XML, cheaper Euler integrator at 7.5 ms).  The plan is a
set of K joint-angle knots (interpolated over the horizon H) that a low-level PD law turns into motor
commands, ``u = kp*(q_target - q) - kd*qdot``.  N candidate knot sets (the previous best plus Gaussian
perturbations of scale ``sig``) are rolled out in parallel with ``mujoco.rollout``; the best one is
kept and its first PD command is applied (predictive sampling, warm-started by shifting the plan).

Rollout score (higher is better), summed over the horizon and zeroed after a predicted fall
(root height outside (1, 2) m):
  * 5 per healthy step  (the environment's alive bonus),
  * ``-vw (vx - vt)^2``  track a slow forward speed ``vt`` (the reward integrates x velocity),
  * ``-upw (1 - up_z)``  keep the torso upright,
  * ``-hw max(0, hz - z)`` stay tall (a fall is z < 1),
  * terminal ``-10 cpw dist(capture point, nearest foot)``: the horizontal capture point
    ``com + tcp * com_velocity`` must stay near a foot, which forces the planner to place steps.

Gains: ``kp``/``KP`` PD stiffness per joint group, ``KD`` damping (kept small: explicit 15 ms
sampling destabilises larger values), ``sig`` exploration scale in rad, ``N``/``H`` samples and horizon.
"""
import os
import numpy as np, mujoco
from mujoco import rollout
from hl_control.envs.humanoid import XML_PATH
P=dict(K=4,H=60,N=128,sig=0.3,sub=2,dti=0.0075,vw=2.0,upw=5.0,cw=0.0,hw=10.0,cpw=5.0,vt=0.5,hz=1.15,nt=min(4,os.cpu_count() or 1),kp=1.5,tcp=0.3,ms=0,elite=0)
SPEC=mujoco.mjtState.mjSTATE_FULLPHYSICS
KP=np.array([1.0]*3+[1.5,1.0,2.0,1.5]*2+[0.3]*6)
KD=np.array([.03]*3+[.01,.01,.06,.03]*2+[.01]*6)
LO=np.array([-1.3,-0.78,-0.6, -.44,-1.05,-1.9,-2.8, -.44,-1.05,-1.9,-2.8,-1.5,-1.5,-1.57,-1.05,-1.05,-1.57])
HI=np.array([.52,.78,.6, .09,.6,.35,-.035, .09,.6,.35,-.035,1.05,1.05,.87,1.5,1.5,.87])
class Planner:
    def __init__(s):
        s.m=mujoco.MjModel.from_xml_path(str(XML_PATH)); s.nt=P['nt']
        s.m.opt.integrator=mujoco.mjtIntegrator.mjINT_EULER; s.m.opt.timestep=P['dti']
        s.ds=[mujoco.MjData(s.m) for _ in range(s.nt)]
        s.d=mujoco.MjData(s.m)
        s.knots=np.zeros((P['K'],17)); s.rng=np.random.default_rng(0)
        s.foot=[s.m.body('right_foot').id,s.m.body('left_foot').id]
        s.kp=KP*P['kp']
        s.tk=np.linspace(0,P['H']-1,P['K']); s.ti=np.arange(P['H'])
    def interp(s,knots):  # (...,K,17)->(...,H,17)
        K=knots.shape[-2]; H=P['H']
        w=np.zeros((H,K))
        for h in range(H):
            f=h/(H-1)*(K-1); i=min(int(f),K-2); a=f-i; w[h,i]=1-a; w[h,i+1]=a
        return np.einsum('hk,nkj->nhj',w,knots)
    def plan(s,obs):
        p=P; g=obs["gym_obs"]
        st=np.zeros(48); st[3:25]=g[:22]; st[25:48]=g[22:45]
        K,H,N,sub=p['K'],p['H'],p['N'],p['sub']
        cand=np.repeat(s.knots[None],N,0)
        sg=np.full(N,p['sig'])
        if p['ms']: sg=np.array([0.1,0.2,0.4,0.7])[np.arange(N)%4]*p['sig']/0.3
        cand[1:]+=s.rng.normal(0,1,cand[1:].shape)*sg[1:,None,None]
        cand=np.clip(cand,LO,HI)
        T=s.interp(cand)   # (N,H,17) target angles
        S=np.tile(st,(N,1))
        zs=np.zeros((N,H)); upz=np.zeros((N,H)); vx=np.zeros((N,H)); cu=np.zeros((N,H))
        for h in range(H):
            q=S[:,8:25]; qd=S[:,31:48]
            u=np.clip(s.kp*(T[:,h]-q)-KD*qd,-0.4,0.4)
            out,_=rollout.rollout(s.m,s.ds,S,np.repeat(u[:,None],sub,1),nstep=sub)
            S=out[:,-1]
            zs[:,h]=S[:,3]; qq=S[:,4:8]; upz[:,h]=1-2*(qq[:,1]**2+qq[:,2]**2); vx[:,h]=S[:,25]; cu[:,h]=np.sum(u**2,1)
            if h==0: u0=u[:,:].copy()
        alive=np.cumprod((zs>1.0)&(zs<2.0),axis=1)
        reward=(-p['vw']*(vx-p['vt'])**2+5.0-p['cw']*cu-p['upw']*(1-upz)-p['hw']*np.maximum(0,p['hz']-zs))*alive
        ret=reward.sum(1)
        for n in range(N):
            if alive[n,-1]==0: continue
            s.d.qpos[:]=S[n,1:25]; s.d.qvel[:]=S[n,25:48]; mujoco.mj_forward(s.m,s.d)
            com=s.d.subtree_com[0]; v=s.d.subtree_linvel[0]
            cp=com[:2]+p['tcp']*v[:2]
            f=np.array([s.d.xpos[b][:2] for b in s.foot])
            ret[n]-=p['cpw']*10*np.min(np.linalg.norm(f-cp,axis=1))
        b=np.argmax(ret)
        if p['elite']:
            idx=np.argsort(ret)[-p['elite']:]; w=np.exp((ret[idx]-ret[idx].max())/ max(1e-6,p.get('lam',1.0))); w/=w.sum()
            s.cur=np.einsum('e,ekj->kj',w,cand[idx]); Tm=s.interp(s.cur[None])[0]
            return Tm[0], ret[b], ret[0]
        s.cur=cand[b]
        return T[b,0], ret[b], ret[0]
    def shift(s):
        # knots for next plan: advance time by one step
        K,H=P['K'],P['H']
        T=s.interp(s.cur[None])[0]
        nxt=np.concatenate([T[1:],T[-1:]],0)
        s.knots=np.stack([np.interp(s.tk,s.ti,nxt[:,j]) for j in range(17)],1)
planner=None
def controller(obs):
    global planner
    if obs["step"]==0: planner=Planner()
    tgt,_,_=planner.plan(obs); planner.shift()
    q=obs["joint_angles"]; qd=obs["joint_rates"]
    u=planner.kp*(tgt-q)-KD*qd
    return np.clip(u,-0.4,0.4)
