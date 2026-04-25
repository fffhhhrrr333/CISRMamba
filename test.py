import sys
import os
import torch
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️ Warning: 'thop' library not found. Skipping complexity calculation.")
    print("👉 Install via: pip install thop")

import config
config.sample_num = 0 

import util
util.sample_num = 0 

from util import eval_fn 
from CISRMamba import CISRMamba
from dataloaded import CustomDataset 
from config import DEVICE, height, width


CHECKPOINT_PATH = "/CISRMamba/final2/CISRMamba_scratch_CISRMamba_best.pt"
TEST_OPT_DIR = "/test/opt"
TEST_SAR_DIR = "/test/vv"
TEST_LBL_DIR = "/test/flood_vv"
RESULT_IMG_PATH = "/root/autodl-tmp/UCDFormer"
CSV_SAVE_PATH = "test_results_standalone.csv"
# ===========================================

def main():
    print(f"Using Device: {DEVICE}")

 
    model = CISRMamba(num_classes=1)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from: {CHECKPOINT_PATH}")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
    else:
        print(f"❌ Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        return

    if THOP_AVAILABLE:
        print("\n" + "="*40)
        print("📊 Calculating Model Complexity...")
        
        try:

            dummy_input = torch.randn(1, 5, height, width).to(DEVICE)
            

            macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
            
            print(f"Model: CISRMamba (Dual Stream: 4-ch Opt + 1-ch SAR)")
            print(f"Input Shape : ({height}, {width})")
            print("-" * 40)
            print(f"Params      : {params / 1e6 :.2f} M (百万)")

            print(f"GFLOPs      : {macs / 1e9 :.2f} G (十亿)") 
            print("="*40 + "\n")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not calculate FLOPs.")
            print(f"Error details: {e}")
            print("="*40 + "\n")

    print(f"Loading Test Dataset from: {TEST_OPT_DIR}")

    test_transform = A.Compose([
        A.ToFloat(255),
        A.Resize(height, width),
        ToTensorV2(),
    ])

    try:
        test_dataset = CustomDataset(
            optical_dir=TEST_OPT_DIR,
            radar_dir=TEST_SAR_DIR,
            label_dir=TEST_LBL_DIR,
            transform=test_transform
        )
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        print(f"✅ Test Loader Ready. Samples: {len(test_dataset)}")

    except Exception as e:
        print(f"❌ Error initializing dataset: {e}")
        return

    print("\n🚀 Starting Evaluation...")

    try:
        with torch.no_grad():

            test_loss, test_metrics = eval_fn(test_loader, model, RESULT_IMG_PATH)

        if test_metrics is not None:
            print("\n" + "="*40)
            print(f"🏆 Evaluation Results")
            print("="*40)
            print(f"Test Loss : {test_loss:.4f}")
            print("-" * 20)
        
            print(f"mIoU      : {test_metrics.get('mIoU', 0.0):.2f}%")
            print(f"F1 Score  : {test_metrics.get('F1', 0.0):.2f}%")
            print(f"Accuracy  : {test_metrics.get('Accuracy', 0.0):.2f}%")
            print(f"Precision : {test_metrics.get('Precision', 0.0):.2f}%")
            print(f"Recall    : {test_metrics.get('Recall', 0.0):.2f}%")
            print(f"Kappa     : {test_metrics.get('Kappa', 0.0):.2f}%")
            print("="*40)
            print(f"Sample visualization saved to: {RESULT_IMG_PATH}")

            df = pd.DataFrame([test_metrics])
            df.to_csv(CSV_SAVE_PATH, index=False)
            print(f"Metrics saved to {CSV_SAVE_PATH}")
        else:
            print("❌ Error: eval_fn returned None.")

    except IndexError as e:
        print(f"\n❌ IndexError: {e}")
        print("Tip: Check if your dataset lists are empty or paths are incorrect.")
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()