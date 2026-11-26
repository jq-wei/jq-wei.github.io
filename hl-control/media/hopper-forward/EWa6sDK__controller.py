"""Hopper controller: a Raibert-style hop cycle written on the centre of mass (COM).

A planar kinematic model of the Hopper XML (link lengths and masses) gives the
whole-body COM relative to the foot centre, ``c``, from (pitch, thigh, leg, foot);
its finite difference plus the torso speed gives the COM forward speed ``v``.

Phase machine (``S['phase']``), switched by foot contact and the knee angle:
  stance  - after touchdown.  Hip holds the torso pitch at ``pd`` with PD gains
            Kp, Kd (target shifted by ``kpv*(v - vd)``); with the foot planted a
            positive hip torque pitches the torso nose-down.  The ankle moves the
            centre of pressure so that the COM tracks ``cd + kc*(vd - v)`` ahead of
            the foot centre (Ka, Da).  The knee is a soft spring (Kks, Dk) that
            crouches to ``kn_st``.
  thrust  - after ``tcomp`` s of stance, once no longer falling: knee command
            ``thrust - kth*max(v - vd, 0)`` until the knee reaches ``kn_end``
            (stopping short of the 0 rad limit avoids the impact that pitches the
            torso); the ankle pushes off with ``push``.
  push    - knee held at ``kn_end``, ankle push until lift-off.
  flight  - the knee places the foot (the leg outweighs the torso, so the knee,
            not the hip, moves the foot in the air): it servos (Kkf, Dk) to the
            angle that puts the foot ``v*Ts/2 + kv*(v - vd)`` ahead of the COM at
            touchdown (Raibert's foot-placement law).  The hip mixes attitude
            (Kpf, Kdf toward ``pd``) with a weak pull (Kh, Dh) of the thigh toward
            the angle that would place the foot with the knee at ``kn_fl``.  The
            ankle keeps the foot flat for landing.
The first ``settle`` s are quiet standing (knee -0.15).  The structure is hand
designed; the numbers were refined by a random local search on training seeds
(cruise ~1.6 m/s; the gate needs 0.375 m/s).
"""

import numpy as np

# ---- planar kinematics of the Hopper (torso, thigh, leg, foot) ----
M = dict(torso=3.6651914291880923, thigh=4.057890510886818, leg=2.7813566998741355, foot=5.3155675816993375)
def R(a, v):
    return np.array([v[0]*np.cos(a)+v[1]*np.sin(a), -v[0]*np.sin(a)+v[1]*np.cos(a)])
def com_rel_foot(pitch, th, lg, ft):
    pitch, th, lg, ft = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in (pitch, th, lg, ft)])
    at, ah = pitch, pitch-th; as_ = ah-lg; af = as_-ft
    T = np.zeros((2,)+np.broadcast(pitch,th,lg,ft).shape); hip = T+R(at,(0,-0.2)); thc = hip+R(ah,(0,-0.225)); knee = hip+R(ah,(0,-0.45))
    shc = knee+R(as_,(0,-0.25)); ank = knee+R(as_,(0,-0.5)); fc = ank+R(af,(0.065,0))
    c = (M['torso']*T+M['thigh']*thc+M['leg']*shc+M['foot']*fc)/sum(M.values())
    return c - fc, fc

# ---- controller ----
P = {'pd': -0.0911, 'Kp': 2.7533, 'Kd': 0.1903, 'Kpf': 0.9652, 'Kdf': 0.1911, 'cd': 0.024, 'Ka': 3.4167, 'Da': 0.1901, 'kc': 0.0488, 'vd': 0.7623, 'Ts': 0.2396, 'kv': 0.086, 'Kkf': 3.3789, 'Dk': 0.1072, 'thrust': 0.6032, 'settle': 0.4562, 'kn_st': -0.6312, 'Kks': 1.3956, 'tcomp': 0.125, 'push': 0.3913, 'kn_end': -0.0802, 'Kh': 0.3266, 'Dh': 0.039, 'kth': -0.0042, 'kpv': 0.0, 'kn_fl': -0.1224}
S={}
LGS=np.linspace(-1.2,0,61)
THS=np.linspace(-1.2,0,61)
def knee_for(pitch, th, xc):
    c,_=com_rel_foot(pitch,th,LGS,pitch-th-LGS)
    return LGS[int(np.argmin(np.abs(c[0]-xc)))]
def thigh_for(pitch, lg, xc):
    c,_=com_rel_foot(pitch,THS,lg,pitch-THS-lg)
    return THS[int(np.argmin(np.abs(c[0]-xc)))]
def controller(obs):
    p=P
    pitch,th,lg,ft=obs['pitch'],obs['thigh'],obs['leg'],obs['foot']
    vz,t=obs['z_velocity'],obs['time']
    c,fc=com_rel_foot(pitch,th,lg,ft)
    ct = c+fc   # COM rel torso centre
    if obs['step']==0: S.update(c=c[0], ct=ct[0], phase='stance', t0=0.0, vc=0.0)
    cdot=(c[0]-S['c'])/obs['dt']; S['c']=c[0]
    vc = obs['x_velocity'] + (ct[0]-S['ct'])/obs['dt']; S['ct']=ct[0]
    S['vc']=0.5*S['vc']+0.5*vc; vx=S['vc']
    contact=obs['foot_contact']
    ph=S['phase']
    if ph=='flight' and contact: ph='stance'; S['t0']=t
    if ph=='thrust' and not contact: ph='flight'
    if ph=='thrust' and lg>p['kn_end']: ph='push'
    if ph=='push' and not contact: ph='flight'
    if ph=='stance' and t>p['settle'] and t-S['t0']>p['tcomp'] and vz>=-0.05 and lg<p['kn_st']+0.15: ph='thrust'
    S['phase']=ph
    a=np.zeros(3)
    if ph!='flight':
        a[0]=-p['Kp']*(pitch-p['pd']+p['kpv']*(vx-p['vd']))-p['Kd']*obs['pitch_rate']
        cdes = p['cd'] + p['kc']*(p['vd']-vx)
        a[2]=-p['Ka']*(c[0]-cdes)-p['Da']*cdot
        if ph=='thrust': a[1]=p['thrust']-p['kth']*max(vx-p['vd'],0); a[2]=-p['push']
        elif ph=='push': a[1]=p['Kks']*(p['kn_end']-lg)-p['Dk']*obs['leg_rate']; a[2]=-p['push']
        else:
            knd = -0.15 if t<p['settle'] else p['kn_st']
            a[1]=p['Kks']*(knd-lg)-p['Dk']*obs['leg_rate']
    else:
        xf = vx*p['Ts']/2 + p['kv']*(vx-p['vd'])
        thd = thigh_for(pitch, p['kn_fl'], -xf)
        a[0]=-p['Kpf']*(pitch-p['pd'])-p['Kdf']*obs['pitch_rate']+p['Kh']*(thd-th)-p['Dh']*obs['thigh_rate']
        knd = knee_for(pitch, th, -xf)
        a[1]=p['Kkf']*(knd-lg)-p['Dk']*obs['leg_rate']
        a[2]=1.0*((pitch-th-lg)-ft)-0.05*obs['foot_rate']
    return np.clip(a,-1,1)
