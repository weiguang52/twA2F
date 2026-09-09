"""Versioned per-robot ridge fit. Candidate fits never silently become live."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .a2f169_to_arkit52 import ARKIT_52_NAMES
from .settings import MOTOR_CFG

DOFS=list(MOTOR_CFG)
NEUTRAL=np.array([MOTOR_CFG[n]['neutral'] for n in DOFS])


def robot_name(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',value):
        raise ValueError('robot_id must be 1..64 ASCII letters, digits, _ or -')
    return value


def fit_calibration(x,y,robot_id,alpha=.1,seed=0,groups=None):
    robot_name(robot_id)
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=52 or y.shape!=(len(x),13) or len(x)<66:
        raise ValueError('Calibration requires >=66 paired rows of [N,52] and [N,13]')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((x<0)|(x>1)) or np.any((y<0)|(y>1)):
        raise ValueError('Paired calibration arrays must be finite and in [0,1]')
    if not np.isfinite(alpha) or alpha<=0:raise ValueError('Ridge alpha must be positive')
    groups=np.arange(len(x)) if groups is None else np.asarray(groups)
    if groups.shape!=(len(x),):raise ValueError('pose_groups must have one value per pair')
    unique=np.unique(groups)
    if len(unique)<5:raise ValueError('Need >=5 independent pose groups for held-out validation')
    rng=np.random.default_rng(seed);order=rng.permutation(unique)
    test=np.isin(groups,order[:max(1,int(len(order)*.2))]);train=~test
    def solve(a,b):return np.linalg.solve(a.T@a+alpha*np.eye(52),a.T@(b-NEUTRAL)).T
    w=solve(x[train],y[train]);pred=np.clip(x[test]@w.T+NEUTRAL,0,1)
    rmse=np.sqrt(np.mean((pred-y[test])**2,axis=0))
    baseline=np.sqrt(np.mean((y[train].mean(0)-y[test])**2,axis=0))
    w=solve(x,y)
    return dict(schema='a2f_robot_ridge_v1',robot_id=robot_id,approved=False,
                provenance='paired_samples_unreviewed',arkit_names=ARKIT_52_NAMES,dof_names=DOFS,
                neutral=NEUTRAL.tolist(),weights=w.tolist(),ridge_alpha=alpha,
                validation=dict(seed=seed,train_rows=int(train.sum()),holdout_rows=int(test.sum()),
                    holdout_rmse=rmse.tolist(),constant_baseline_rmse=baseline.tolist(),
                    observed_channels=[ARKIT_52_NAMES[i] for i in range(52) if np.ptp(x[:,i])>.01]),
                sample_count=len(x))


class RetargetCalibration:
    def __init__(self,path,robot_id=None,allow_candidate=False):
        path=Path(path)
        raw=path.read_bytes();data=json.loads(raw)
        if data.get('schema')!='a2f_robot_ridge_v1':raise ValueError('Unsupported calibration schema')
        robot_name(data.get('robot_id'))
        if robot_id is not None and data['robot_id']!=robot_id:raise ValueError('Calibration robot_id mismatch')
        if data.get('approved') is not True and not allow_candidate:
            raise ValueError('Calibration is a candidate; review/approve it before live use')
        if data.get('arkit_names')!=ARKIT_52_NAMES or data.get('dof_names')!=DOFS:
            raise ValueError('Calibration channel order mismatch')
        self.weights=np.asarray(data['weights'],dtype=np.float64)
        self.neutral=np.asarray(data['neutral'],dtype=np.float64)
        if self.weights.shape!=(13,52) or self.neutral.shape!=(13,) or not np.isfinite(self.weights).all() or not np.isfinite(self.neutral).all():
            raise ValueError('Invalid calibration matrix')
        if not np.allclose(self.neutral,NEUTRAL):raise ValueError('Calibration must preserve mechanism neutral')
        self.weights.flags.writeable=False;self.neutral.flags.writeable=False
        self.robot_id=data['robot_id'];self.sha256=hashlib.sha256(raw).hexdigest()

    def predict(self,bs):
        x=np.array([bs.get(n,0.) for n in ARKIT_52_NAMES],dtype=np.float64)
        if not np.isfinite(x).all():raise ValueError('Nonfinite ARKit input')
        y=np.clip(self.neutral+self.weights@np.clip(x,0,1),0,1)
        return dict(zip(DOFS,map(float,y)))
