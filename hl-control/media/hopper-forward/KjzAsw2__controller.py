"""Vibration-shuffle controller for the MuJoCo Hopper (found by evolution-strategy tuning of ~19 gains).

Control law (all joints; commands in [-1, 1], gear 200 N m):
  phase  w = 2*pi*(t - t_touchdown)/T            -- a clock restarted at every foot touchdown
  knee   uk = kb + ka*sin(w+pk) + kvk*(vx-vd)    -- periodic crouch/extend, plus kT thrust while rising in contact
  hip    uh = hb + ha*sin(w+ph) + kv*(vx-vd) - kp*(pitch-pd) - kd*pitch_rate
                                                 -- PD on torso pitch; a NEGATIVE hip command pitches the torso
                                                    back, so the feedback gains enter with a minus sign
  ankle  ua = ab + aa*sin(w+pa) - ak*foot - ad*foot_rate  -- PD to level foot plus a small periodic term
The gains are clipped to BOUNDS so the ankle cannot be driven into an anti-damped windmill.
"""
import numpy as np
NAMES=['T','kb','ka','pk','hb','ha','ph','kp','kd','ab','aa','pa','kT','pd','ak','ad','kv','kvk','vd']
PARAMS={'T': 0.4069, 'kb': 0.1001, 'ka': 0.158, 'pk': -0.0381, 'hb': -0.0602, 'ha': -0.1487, 'ph': 0.0666, 'kp': 0.8636, 'kd': 0.3133, 'ab': 0.1194, 'aa': 0.1627, 'pa': -0.2131, 'kT': 0.3949, 'pd': 0.1484, 'ak': 0.6213, 'ad': 0.5459, 'kv': -0.12, 'kvk': -0.1633, 'vd': 0.2887}
BOUNDS=dict(T=(0.3,1.5),kp=(0,2),kd=(0,0.5),aa=(0,0.3),ak=(0,1.5),ad=(0,0.3),kT=(0,1),ka=(0,0.8),pd=(-0.1,0.15),ha=(0,0.4))
_S={'td':0.0,'c':False}
def controller(o,P=PARAMS):
    P={k:min(max(v,BOUNDS[k][0]),BOUNDS[k][1]) if k in BOUNDS else v for k,v in P.items()}
    if o['step']==0: _S['td']=0.0; _S['c']=False
    c=o['foot_contact']
    if c and not _S['c']: _S['td']=o['time']
    _S['c']=c
    th,lg,ft=o['joint_angles']
    w=2*np.pi*(o['time']-_S['td'])/P['T']
    ev=o['x_velocity']-P['vd']
    uk=P['kb']+P['ka']*np.sin(w+P['pk'])+P['kvk']*ev
    if c and o['z_velocity']>0: uk+=P['kT']
    uh=P['hb']+P['kv']*ev+P['ha']*np.sin(w+P['ph'])-P['kp']*(o['pitch']-P['pd'])-P['kd']*o['pitch_rate']
    ua=P['ab']+P['aa']*np.sin(w+P['pa'])+P['ak']*(0-ft)-P['ad']*o['foot_rate']
    return np.clip(np.array([uh,uk,ua]),-1,1)
