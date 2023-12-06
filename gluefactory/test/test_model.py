from pathlib import Path

import torch
from omegaconf import OmegaConf

from gluefactory.eval.io import get_eval_parser, parse_eval_args, load_model
from gluefactory.settings import TRAINING_PATH

default_conf = {
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        }
    }



parser = get_eval_parser()
args = parser.parse_intermixed_args()

default_conf = OmegaConf.create(default_conf)
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.get('experience'):
    checkpoint = OmegaConf.load(
        TRAINING_PATH / args.get('experience') / "config.yaml"
    )
        # 只留下model
    mconf = OmegaConf.create(
        {
            "model": checkpoint.get("model", {})
        }
    )
    model = load_model(mconf.model, args.get('experience'))
    model = model.to(device).eval()

if args.get('experience_2views'):
    checkpoint_2viewsconf = OmegaConf.load(
        TRAINING_PATH / args.get('experience_2views') / "config.yaml"
    )
        # 只留下model
    mconf_2views = OmegaConf.create(
        {
            "model_2views": checkpoint_2viewsconf.get("model", {})
        }
    )

    model_2views = load_model(mconf.model, args.get('experience_2views'))
    model_2views = model_2views.to(device).eval()


data_root_path = r"/"
image_path1 = data_root_path + "image_name1.jpg"
image_path2 = data_root_path + "image_name2.jpg"

# 加载数据并预处理



