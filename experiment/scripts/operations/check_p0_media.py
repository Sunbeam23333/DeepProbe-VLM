#!/usr/bin/env python3
"""Decode/process every unique pilot clip on CPU; this is not a speed result."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--media-root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from transformers import AutoImageProcessor
    from deepprobe_vlm.data import load_frames
    from deepprobe_vlm.manifest import load_manifest, validate_disjoint
    cfg=json.loads(args.config.read_text())['model']
    processor=AutoImageProcessor.from_pretrained(cfg['vision_model'],revision=cfg['vision_revision'],use_fast=False,local_files_only=True)
    groups=[load_manifest(args.data_root/name,args.media_root) for name in ('train.jsonl','validation.jsonl')]
    validate_disjoint(*groups)
    unique={}
    for group in groups:
        for row in group:
            key=(row['video_path'],row.get('start_seconds'),row.get('end_seconds'))
            unique.setdefault(key,row)
    results=[]
    for key,row in sorted(unique.items()):
        frames=load_frames(row,args.media_root,cfg['num_frames'])
        pixels=processor(images=frames,return_tensors='pt')['pixel_values']
        assert len(frames)==cfg['num_frames'] and pixels.shape[0]==cfg['num_frames']
        assert torch.isfinite(pixels).all()
        results.append({'video_path':row['video_path'],'source_video_id':row['source_video_id'],
                        'decoded_frames':len(frames),'pixel_shape':list(pixels.shape),'finite':True})
    report={'status':'all_selected_media_decoded_and_processed','device':'cpu','clips':results,
            'unique_clips':len(results),'quality_result':False,'gpu_validated':False,
            'manifest_sha256':{name:hashlib.sha256((args.data_root/name).read_bytes()).hexdigest()
                               for name in ('train.jsonl','validation.jsonl')}}
    with args.output.open('x') as stream: json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':report['status'],'unique_clips':len(results)}))


if __name__=='__main__':main()
