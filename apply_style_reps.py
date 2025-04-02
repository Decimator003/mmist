import argparse
import itertools
import os
import random

import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
from tqdm import tqdm

import utils
from model_bridge import load_adaattn_model, make_adaattn_input

def contextual_loss(feat_1, feat_2, h=0.5, device = 'cuda'):
    """
    Compute contextual loss between feature maps
    Args:
        feat_x: Features of shape (N, C, H, W)
        feat_y: Features of shape (N, C, H, W)
        h: Bandwidth parameter
        eps: Small constant for numerical stability
    """

    assert isinstance(feat_1, torch.Tensor), "feat_1 must be a torch.Tensor"
    assert isinstance(feat_2, torch.Tensor), "feat_2 must be a torch.Tensor"

    feat_1 = feat_1.to(device)
    feat_2 = feat_2.to(device)

    N, C, H, W = feat_1.size()

    # Reshape and normalize features
    feat_1 = feat_1.view(N, C, -1)
    feat_2 = feat_2.view(N, C, -1)
    
    feat_1 = feat_1 / torch.norm(feat_1, p=2, dim=1, keepdim=True)
    feat_2 = feat_2 / torch.norm(feat_2, p=2, dim=1, keepdim=True)

    # Calculate cosine distance
    dist = 1 - torch.bmm(feat_1.transpose(1,2), feat_2)  # (N, H*W, H*W)

    # Calculate relative distances
    dist_min, _ = torch.min(dist, dim=2, keepdim=True)  # (N, H*W, 1)
    dist_tilde = dist / (dist_min + 1e-5)

    # Calculate weights
    w = torch.exp((1 - dist_tilde) / h)
    
    # Calculate contextual loss
    cx = w / torch.sum(w, dim=2, keepdim=True)  # (N, H*W, H*W)
    cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)  # (N,)
    
    cx_loss = torch.mean(-torch.log(cx + 1e-5))
    return cx_loss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style_reps_dir', type=str, required=True, help='path to source folder')
    parser.add_argument('--output_dir', type=str, required=True, help='path to target folder')
    parser.add_argument('--content_paths', nargs='+', type=str, default=[], help='paths to content file or dirs')
    parser.add_argument('--adaattn_path', type=str, default='./AdaAttN', help='path to AdaAttN model')
    parser.add_argument('--n_ens', type=int, default=None, help='number of ensemble')
    parser.add_argument('--lambda_cx', type=float, default=1.0, help='Weight for contextual loss')
    parser.add_argument('--cx_layers', nargs='+', type=str, 
                       default=['relu3_1', 'relu4_1', 'relu5_1'],
                       help='VGG layers for contextual loss')
    parser.add_argument('--use_contextual', action='store_true', help='Use contextual loss')
    parser.add_argument('--cx_weight', type=float, default=0.1, help='Weight for contextual loss')
    return parser.parse_args()

def build_sty_path_dict(result_dir):
    img_paths = os.listdir(result_dir)
    img_paths = [name for name in img_paths if f'-style.' in name]

    sty_path_dict = {}
    for img_path in img_paths:
        style = img_path.split('_')[0]
        if style not in sty_path_dict:
            sty_path_dict[style] = []
        sty_path_dict[style].append(os.path.join(result_dir, img_path))

    return sty_path_dict

def main(args):
    args.adaattn_path = os.path.abspath(args.adaattn_path)
    contents = {}
    for content_path in args.content_paths:
        assert os.path.exists(content_path), f'Content file {content_path} does not exist.' 
        if os.path.isfile(content_path):
            content_name = os.path.splitext(os.path.basename(content_path))[0]
            contents[content_name] = content_path
        elif os.path.isdir(content_path):
            for root, dirs, files in os.walk(content_path):
                for name in files:
                    if name.endswith('.png') or name.endswith('.jpg'):
                        content_name = os.path.splitext(name)[0]
                        contents[content_name] = os.path.join(root, name)
                    
    adaattn = load_adaattn_model(args.adaattn_path)

    @torch.no_grad()
    def multi_adaattn_ens(content_path, style_paths):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        content = utils.load_image_512(content_path).to(device)
        style_imgs = []
        for sp in style_paths:
          s = utils.load_image_512(sp).to(device)
          sstd = s.std()
          if sstd < 0.03:
            print(f'Warning: {sp} has std {sstd}, this style image will be ignored.')
          else:
            style_imgs.append(s)
            
        adaattn.set_input(make_adaattn_input(content, style_imgs))
        adaattn.forward()
        target = adaattn.cs.to(device)
    
        if args.use_contextual:
          # Get VGG features
          content_features = utils.get_vgg_features(content)
          target_features = utils.get_vgg_features(target)
          style_features = [utils.get_vgg_features(s) for s in style_imgs]
        
          # Calculate contextual loss for multiple feature layers
          cx_loss = 0
          for key in ['conv4_2', 'conv5_2']:
            cx_loss += contextual_loss(
              target_features[key],
              content_features[key]
            )
        
        # Average style features
          # style_feat = torch.stack([sf[key] for sf in style_features]).mean(0)
          # cx_loss += contextual_loss(
          #   target_features[key], 
          #   style_feat
          # )
    
        # Combine with original output

          blend_weight = args.cx_weight * torch.clamp(cx_loss, 0, 1)
          target = (1 - blend_weight) * target + blend_weight * content
          target = target.clamp(0, 1)  # Ensure values stay in valid range
          # alpha = 0.5 # Weight for contextual loss
          # target = target * alpha * (cx_loss) 
        
        return target

    sd = build_sty_path_dict(args.style_reps_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    for s, c in tqdm(itertools.product(sd, contents), total=len(sd)*len(contents)):
        content_path = contents[c]

        if args.n_ens is None:
            result = multi_adaattn_ens(content_path, sd[s])
        else:
            result = multi_adaattn_ens(content_path, random.sample(sd[s], args.n_ens))

        result_path = os.path.join(args.output_dir, f'{s}_{c}.png')
        result_pil = ToPILImage()(result.squeeze().clamp(0,1))
        result_pil.save(result_path)
        
if __name__ == '__main__':
    main(get_args())