#!/usr/bin/env python3
"""Top-down protection heat map from voxel_protection_*.npz: GT voxels averaged per
(x,y) column, colored by covered/demanded fraction in the 0-5 m bands."""
import os, sys, json
DATA_ROOT = os.environ.get('GG_DATA_ROOT', os.path.expanduser('~/glassguard_data'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))  # eval_occupancy.py lives next to this file
import numpy as np, open3d as o3d
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from pathlib import Path
from eval_occupancy import viewer_to_world, discover_frames

def render(scene, cam):
    root=Path(f'{DATA_ROOT}/{scene}_live/inputs')
    z=np.load(root/f'gts/voxel_protection_{cam}.npz')
    C=z['centers']; dem=z['dem02']+z['dem25']; cov=z['cov02']+z['cov25']
    frames=discover_frames(root)
    fc=root/'frame_convention.txt'; world=fc.exists() and 'world' in fc.read_text()
    bg=[]; path=[]
    for i,f in enumerate(frames):
        path.append(viewer_to_world(np.zeros((1,3)),*f['pose'])[0])
        if i%20==0:
            pc=np.asarray(o3d.io.read_point_cloud(str(f['cloud'])).points)
            if not world: pc=viewer_to_world(pc,*f['pose'])
            bg.append(pc[::9])
    BG=np.concatenate(bg); path=np.array(path)
    # column-average: group voxels by (x,y) cell
    key=np.round(C[:,:2]).astype(int)
    cols={}
    for k,(dd,cc) in zip(map(tuple,key),zip(dem,cov)):
        a=cols.setdefault(k,[0,0]); a[0]+=dd; a[1]+=cc
    xs=[];ys=[];fr=[];nod_x=[];nod_y=[]
    for (x,y),(dd,cc) in cols.items():
        if dd==0: nod_x.append(x); nod_y.append(y)
        else: xs.append(x); ys.append(y); fr.append(cc/dd)
    fig,ax=plt.subplots(figsize=(14,14))
    ax.scatter(BG[:,0],BG[:,1],s=0.2,c='0.85',lw=0)
    ax.scatter(nod_x,nod_y,s=40,marker='s',c='0.6',lw=0,label='GT: never demanded 0-5m')
    sc=ax.scatter(xs,ys,s=110,marker='s',c=fr,cmap='RdYlGn',vmin=0,vmax=1,
                  edgecolors='k',lw=0.4)
    ax.plot(path[:,0],path[:,1],'k-',lw=1.0,alpha=0.8,label='robot path')
    cb=fig.colorbar(sc,ax=ax,fraction=0.03,pad=0.01)
    cb.set_label('protection: covered / demanded voxel-frames (0-5 m bands)',fontsize=12)
    ax.set_aspect('equal'); ax.legend(loc='upper right',fontsize=12)
    ax.set_title(f'{scene} {cam} -- per-voxel current-map protection (column-averaged top-down)',fontsize=14)
    out=f'{DATA_ROOT}/protection_maps/{scene}_{cam}_protection.png'
    Path(out).parent.mkdir(exist_ok=True)
    fig.savefig(out,dpi=120,bbox_inches='tight'); plt.close(fig)
    print('saved',out)
    return out

if __name__=='__main__':
    render(sys.argv[1], sys.argv[2])
