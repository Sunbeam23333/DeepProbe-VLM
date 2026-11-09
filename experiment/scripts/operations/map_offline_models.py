#!/usr/bin/env python3
"""Map already verified local model assets into a pinned offline Hub cache."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--hub-cache',type=Path,required=True)
    args=p.parse_args()
    for name in ('ouro','siglip'):
        receipts=list((args.assets/name).glob('*/ASSET_COMMITTED.json'))
        if len(receipts)!=1:
            raise ValueError(f'Expected one committed revision for {name}')
        receipt=receipts[0]; item=json.loads(receipt.read_text())
        assert item['downloaded'] and item['repo_type']=='model'
        root=args.hub_cache/('models--'+item['repo_id'].replace('/','--'))/'snapshots'/item['revision']
        for entry in item['files']:
            source=receipt.parent/entry['path']
            digest=hashlib.sha256()
            with source.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''): digest.update(block)
            if source.stat().st_size!=entry['size_bytes'] or digest.hexdigest()!=entry['local_sha256']:
                raise ValueError(f'Changed staged model file: {entry["path"]}')
            target=root/entry['path'];target.parent.mkdir(parents=True,exist_ok=True)
            if target.is_symlink():
                if target.resolve()!=source.resolve(): raise ValueError(f'Cache conflict: {target}')
            elif target.exists():
                raise ValueError(f'Refusing to replace existing cache file: {target}')
            else: target.symlink_to(source.resolve())
        print(name,item['revision'],len(item['files']))


if __name__=='__main__':main()
