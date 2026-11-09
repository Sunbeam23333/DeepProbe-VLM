#!/usr/bin/env python3
"""Coordinate the 31 pinned archive cache; HF credentials are stdin-only."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

BOOTSTRAP=('import hashlib,runpy,sys; p=sys.argv.pop(1); expected=sys.argv.pop(1); '
           'assert hashlib.sha256(open(p,"rb").read()).hexdigest()==expected; '
           'sys.argv[0]=p; runpy.run_path(p,run_name="__main__")')
SAFE_FIELDS={'status','asset_id','bytes','sha256','receipt_sha256','source_mode','error_code',
             'archive_count','total_bytes','plan_sha256','archives','catalog_sha256','all_benchmark_assets_ready'}


class SafeError(Exception):pass
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def load_validator(expected_sha):
    path=Path(__file__).with_name('hf_to_stream_oss_cache.py')
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha:
        raise SafeError('local_validator_sha_changed')
    spec=importlib.util.spec_from_file_location('frozen_cache_validator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.validate_plan


def resolve_signed(asset,token,opener):
    url='https://huggingface.co/datasets/'+asset['repo_id']+'/resolve/'+asset['revision']+'/'+asset['path']
    request=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Range':'bytes=0-0',
        'Accept-Encoding':'identity','Cache-Control':'no-cache'})
    try:response=opener.open(request,timeout=45)
    except urllib.error.HTTPError as error:response=error
    with response:
        if response.code!=302:raise SafeError('hf_resolve_non_redirect')
        return response.headers['Location']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--config-sha256',required=True)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    raw=args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.config_sha256:raise SafeError('config_sha_changed')
    config=json.loads(raw)
    raw_plan=Path(config['local_plan']).read_bytes()
    if hashlib.sha256(raw_plan).hexdigest()!=config['plan_sha256']:raise SafeError('local_plan_changed')
    plan=json.loads(raw_plan)
    if not re.fullmatch('[0-9a-f]{64}',config['worker_sha256']):raise SafeError('worker_sha_missing')
    # Validate the fixed repository/revision/file set before any bearer header.
    validate_plan=load_validator(config['worker_sha256'])
    assets=validate_plan(plan)
    if not args.execute:
        print(json.dumps({'status':'PREPARED_NO_MUTATIONS','archives':len(assets),'bytes':plan['total_bytes']}));return
    token=sys.stdin.readline(4098).strip()
    if not re.fullmatch('hf_[A-Za-z0-9]{1,4093}',token):raise SafeError('invalid_stdin_credential')
    opener=urllib.request.build_opener(NoRedirect(),urllib.request.ProxyHandler({}))
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=30',
         '-o','ServerAliveCountMax=3','haopengjin2ali']
    def call(mode,asset=None,signed=None):
        command=[config['python'],'-c',BOOTSTRAP,config['worker'],config['worker_sha256'],mode,
                 '--plan',config['remote_plan'],'--plan-sha256',config['plan_sha256']]
        if mode!='status':command.append('--execute')
        if asset:command+=['--asset-id',asset['asset_id']]
        response=subprocess.run(ssh+[shlex.join(command)],input=json.dumps({'signed_url':signed}) if signed else '',
                      text=True,capture_output=True,timeout=6*3600)
        try:result=json.loads(response.stdout)
        except Exception:raise SafeError('remote_transport_or_protocol_uncertain') from None
        print(json.dumps({k:v for k,v in result.items() if k in SAFE_FIELDS}),flush=True)
        if response.returncode not in (0,2):raise SafeError('remote_transport_uncertain')
        return response.returncode,result
    rc,state=call('initialize')
    if rc or state.get('status')!='CACHE_INITIALIZED':raise SafeError('cache_initialization_failed')
    for asset in sorted(assets,key=lambda a:(not bool(a.get('local_source')),a['size_bytes'])):
        for attempt in range(1,7):
            rc,state=call('status',asset)
            if rc:raise SafeError('exact_object_status_failed')
            if state['status']=='CACHED_VERIFIED':break
            if state['status'] not in {'NEEDS_SIGNED_URL','READY_LOCAL_SOURCE','OBJECT_VERIFIED_RECEIPT_PENDING'}:
                raise SafeError('unexpected_object_state')
            signed=None
            if state['status']=='NEEDS_SIGNED_URL':
                for resolve_attempt in range(6):
                    try:signed=resolve_signed(asset,token,opener);break
                    except Exception:
                        if resolve_attempt==5:raise SafeError('hf_link_resolution_exhausted') from None
                        time.sleep(min(30,2**resolve_attempt))
            rc,result=call('upload',asset,signed);signed=None
            if not rc and result.get('status')=='CACHED_VERIFIED':break
            code=result.get('error_code','')
            # New attempts always inspect the exact final key and bound multipart
            # state first. Never retry identity/hash conflicts or uncertain init.
            if not code.startswith('operation_') or attempt==6:
                raise SafeError('asset_upload_failed_'+code if re.fullmatch('[A-Za-z0-9_]+',code) else 'asset_upload_failed')
            time.sleep(min(30,2**attempt))
        else:raise SafeError('asset_attempt_budget_exhausted')
    rc,result=call('finalize')
    if rc or result.get('status')!='ARCHIVE_CATALOG_COMPLETE':raise SafeError('archive_catalog_incomplete')
    print(json.dumps({'event':'PINNED_HF_ZIP_CACHE_COMPLETE','archives':31,'bytes':plan['total_bytes'],
                      'catalog_sha256':result['catalog_sha256'],'all_benchmark_assets_ready':False}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        code=str(error) if isinstance(error,SafeError) else 'coordinator_'+type(error).__name__
        print(json.dumps({'event':'HF_OSS_CACHE_COORDINATOR_FAILED','error_code':code}),flush=True);sys.exit(2)
