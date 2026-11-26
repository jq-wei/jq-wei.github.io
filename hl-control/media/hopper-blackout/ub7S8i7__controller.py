"""Hopper through sensing blackouts: open-loop periodic hop gait + pitch feedback + model dead-reckoning.

Control law (125 Hz):
  * Steps < tset: stand (ankle PD on pitch, hip PD to a slightly bent thigh, knee pressed straight).
  * Afterwards a clock phase ph = t/T drives cosine-interpolated 6-knot joint-angle tables (th*, kn*, ft*),
    one cycle per hop (crouch -> thrust -> tuck -> land). Each joint is tracked by a PD law
    command = kj*(target - q) - kd*qdot   (kj [1/rad] stiffness, kd [s] damping, in units of the 200 N m gear).
  * Pitch feedback: the thigh target is shifted by kph*pitch + kpd*pitch_rate (the hip is the attitude actuator,
    thigh kept off 0 by the tables), and the ankle gets -kpa*pitch while the foot is on the ground.
  * Blackouts: a repeated observation is detected by exact equality of (height, pitch, thigh). The plant is
    deterministic and nominal, and the full state is measured at the blackout's first step, so the controller
    re-simulates its own MuJoCo copy of the hopper with the executed commands (obs['last_action']) and feeds the
    controller the dead-reckoned state; a blackout then changes nothing.
The knot values and gains were found by random hill-climbing on the visible seeds.
"""
import numpy as np
import mujoco
from hl_control.envs.hopper import HopperEnv, HopperConfig

K = 6
P={'T': 0.5825, 'tset': 28.2811, 'kj': 4.8918, 'kd': 0.4314, 'kph': 2.7125, 'kpd': 0.1283, 'kpa': 0.2489, 'th0': -0.3484, 'th1': -0.4794, 'th2': 0.0, 'th3': -0.3839, 'th4': -0.2921, 'th5': 0.0, 'kn0': -0.3681, 'kn1': -0.1417, 'kn2': 0.0, 'kn3': -1.3, 'kn4': -0.254, 'kn5': -0.7777, 'ft0': 0.7, 'ft1': 0.4096, 'ft2': 0.3022, 'ft3': -0.7, 'ft4': -0.1964, 'ft5': -0.0123}

_sim = HopperEnv(HopperConfig())   # nominal plant copy used to dead-reckon the state through blackouts
st = dict(prev=None, stale=0)

def tab(P,name):
    return np.array([P[name+str(i)] for i in range(K)])
def interp(v,ph):
    x=ph*K; i=int(x)%K; f=x-int(x)
    f=0.5-0.5*np.cos(np.pi*f)
    return v[i]*(1-f)+v[(i+1)%K]*f
def _read(sim):
    d=sim.data; q=d.qpos; v=d.qvel
    return dict(pitch=q[2],pitch_rate=v[2],thigh=q[3],thigh_rate=v[3],leg=q[4],leg_rate=v[4],foot=q[5],foot_rate=v[5],foot_contact=sim.foot_contact())
def estimate(o):
    """State estimate; during a blackout (repeated observation) re-simulate the plant from the last fresh state."""
    key=(o['height'],o['pitch'],o['thigh'])
    d=_sim.data
    if o['step']<=1: st['prev']=None
    if st['prev']==key:
        st['stale']+=1
        d.ctrl[:]=np.clip(o['last_action'],-1,1)
        mujoco.mj_step(_sim.model,d,nstep=_sim.config.frame_skip)
    else:
        st['stale']=0
        d.qpos[:]=[0.0,o['height'],o['pitch'],o['thigh'],o['leg'],o['foot']]
        d.qvel[:]=[o['x_velocity'],o['z_velocity'],o['pitch_rate'],o['thigh_rate'],o['leg_rate'],o['foot_rate']]
        mujoco.mj_forward(_sim.model,d)
    st['prev']=key
    if st['stale']==0:
        return dict((k,o[k]) for k in ('pitch','pitch_rate','thigh','thigh_rate','leg','leg_rate','foot','foot_rate','foot_contact'))
    return _read(_sim)
def controller(o,P=P):
    e=estimate(o)
    p=e['pitch']; w=e['pitch_rate']; step=o['step']
    th,kn,ft=e['thigh'],e['leg'],e['foot']
    if step<P['tset']:
        af=-(3.0*p+0.5*w); ah=-(2.0*(th+0.15)+0.3*e['thigh_rate']); ak=0.3
        return np.clip([ah,ak,af],-1,1)
    ph=((step-P['tset'])*o['dt']/P['T'])%1.0
    tth=interp(tab(P,'th'),ph); tkn=interp(tab(P,'kn'),ph); tft=interp(tab(P,'ft'),ph)
    tth-= P['kph']*p+P['kpd']*w      # pitch feedback through hip target (sign tuned by search)
    ah=P['kj']*(tth-th)-P['kd']*e['thigh_rate']
    ak=P['kj']*(tkn-kn)-P['kd']*e['leg_rate']
    af=P['kj']*(tft-ft)-P['kd']*e['foot_rate']-P['kpa']*(p if e['foot_contact'] else 0)
    return np.clip(np.array([ah,ak,af]),-1,1)
