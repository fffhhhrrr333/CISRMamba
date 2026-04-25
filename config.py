import os
DEVICE='cuda'
EPOCHS=60
BATCH_SIZE=16
LR=1e-4
ratio=0.5 
sample_num=3


MAXMIN=False
height,width = (256, 256)

ENCODER='CISRMamba'

WEIGHTS='scratch'

name='CISRMamba'

basedir = '/CISRMamba/'

os.makedirs(basedir, exist_ok=True)
outptfile = os.path.join(basedir, f'{ENCODER}_{WEIGHTS}_{name}.pt')

loadstate = False
loadstateptfile = outptfile

def log(traintxt, ds):
    with open(traintxt, 'a') as f:
        f.write(ds)

# --- Dataset and workspace paths ---
OPTICAL_DIR = "/train/opt"
RADAR_DIR = "/train/vv"
LABEL_DIR = "/train/flood_vv"
# Checkpoint and metrics output
CHECKPOINT_DIR = '/CISRMamba/final2'
METRICS_CSV = '/CISRMamba/final2/ours.csv'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)



