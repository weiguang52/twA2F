"""Fit a review-required calib_{robot}.json from real ARKit/ideal-DOF pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from ..motion_core.retarget_calibration import fit_calibration,ARKIT_52_NAMES,DOFS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',required=True)
    p.add_argument('--robot',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--alpha',type=float,default=.1)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError('Refusing to overwrite '+str(out))
    with np.load(args.samples,allow_pickle=False) as d:
        names=lambda k:[v.decode() if isinstance(v,bytes) else str(v) for v in d[k]]
        if names('arkit_names')!=ARKIT_52_NAMES or names('dof_names')!=DOFS:
            raise ValueError('Sample channel names/order mismatch')
        fit=fit_calibration(d['arkit_values'],d['dof_values'],args.robot,args.alpha,args.seed,
                            d['pose_groups'] if 'pose_groups' in d.files else None)
    fit['source_sha256']=hashlib.sha256(Path(args.samples).read_bytes()).hexdigest()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(fit,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('Candidate saved (NOT enabled):',out)
    print('Held-out per-DOF RMSE:',fit['validation']['holdout_rmse'])


if __name__=='__main__':main()
