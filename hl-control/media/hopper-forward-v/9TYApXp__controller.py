"""Parametrised hop state machine (stance-down / stance-up / flight) with a hip attitude loop.

hip   = -kp*(pitch - p0) - kd*pitch_rate + phase bias        (attitude loop, bias keeps thigh off its stop)
knee  = stance: crouch torque until the knee passes `crouch`, then thrust; flight: PD to fold/land angle
ankle = PD to a phase target + coupling to pitch
p0 is shifted by kv*(x_velocity - vt) so the lean sets the speed.
"""
import numpy as np

PARAMS = dict(kp=1.7867,
              kd=0.1691,
              p0=0.1457,
              kv=-0.385,
              vt=1.1965,
              hb_down=0.3585,
              hb_up=0.2446,
              hb_fold=0.0516,
              hb_land=0.0776,
              k_down=-0.1765,
              thrust=0.6318,
              crouch=-0.261,
              fold=-0.2046,
              land=0.0,
              kpl=0.879,
              ft_st=0.0878,
              ka=-2.3412,
              kda=-0.3198,
              ft_fl=0.2334,
              kt=-0.5238,
              th_fl=-0.5894)


class _S:
    phase = 'down'


class _R:
    phase = 'down'


def _rail(obs):
    """Vertical-rail hop (pitch locked): crouch to a knee angle, then full knee thrust; fold in flight."""
    th, lg, ft = obs['thigh'], obs['leg'], obs['foot']
    ank = -1.0 * ft - 0.05 * obs['foot_rate']
    if obs['step'] == 0:
        _R.phase = 'down'
    if obs['foot_contact']:
        if _R.phase == 'fl':
            _R.phase = 'down'
        if _R.phase == 'down' and lg < -0.7:
            _R.phase = 'up'
        knee = -0.5 if _R.phase == 'down' else 1.0
        hip = (-0.3 - th) if _R.phase == 'down' else (-0.1 - th)
    else:
        _R.phase = 'fl'
        tgt = -0.6 if obs['z_velocity'] > 0 else -0.1
        knee = 2.0 * (tgt - lg) - 0.1 * obs['leg_rate']
        hip = 2.0 * ((-0.3 if obs['z_velocity'] > 0 else -0.2) - th) - 0.1 * obs['thigh_rate']
    return np.clip([hip, knee, ank], -1, 1)


def controller(obs):
    P = PARAMS
    if 'rooty' in obs['locked_joints']:
        return _rail(obs)
    th, lg, ft = obs['thigh'], obs['leg'], obs['foot']
    pitch, pr = obs['pitch'], obs['pitch_rate']
    p0 = P['p0'] + P['kv'] * (obs['x_velocity'] - P['vt'])
    att = -P['kp'] * (pitch - p0) - P['kd'] * pr
    if obs['step'] == 0:
        _S.phase = 'down'
    if obs['foot_contact']:
        if _S.phase.startswith('fl'):
            _S.phase = 'down'
        if _S.phase == 'down' and lg < P['crouch']:
            _S.phase = 'up'
        if _S.phase == 'down':
            hip, knee = att + P['hb_down'], P['k_down']
        else:
            hip, knee = att + P['hb_up'], P['thrust']
        ank = 1.0 * (P['ft_st'] - ft) - 0.05 * obs['foot_rate'] + P['ka'] * pitch + P['kda'] * pr
    else:
        up = obs['z_velocity'] > 0
        _S.phase = 'fl_up' if up else 'fl_dn'
        tgt = P['fold'] if up else P['land']
        knee = np.clip(P['kpl'] * (tgt - lg) - 0.05 * obs['leg_rate'], -0.3, 0.3)
        hip = att + (P['hb_fold'] if up else P['hb_land']) + P['kt'] * (P['th_fl'] - th)
        ank = 1.0 * (P['ft_fl'] - ft) - 0.05 * obs['foot_rate']
    return np.clip([hip, knee, ank], -1, 1)
