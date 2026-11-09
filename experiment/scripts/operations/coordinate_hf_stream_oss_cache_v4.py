#!/usr/bin/env python3
"""Coordinate the 31 pinned archive cache; HF credentials are stdin-only."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BOOTSTRAP=('import hashlib,runpy,sys; p=sys.argv.pop(1); expected=sys.argv.pop(1); '
           'assert hashlib.sha256(open(p,"rb").read()).hexdigest()==expected; '
           'sys.argv[0]=p; runpy.run_path(p,run_name="__main__")')
SAFE_FIELDS={'status','asset_id','bytes','sha256','receipt_sha256','source_mode','error_code',
             'archive_count','total_bytes','plan_sha256','archives','catalog_sha256','all_benchmark_assets_ready'}
REDIRECT_CODES={301,302,303,307,308}
MAX_REDIRECTS=6
MAX_STREAM_REFRESHES=6
STREAM_LINE_LIMIT=20_000
STREAM_TIMEOUT_SECONDS=6*3600
CDN_HOSTS={'cas-bridge.xethub.hf.co','cdn-lfs.hf.co','cdn-lfs-us-1.hf.co',
           'cdn-lfs-eu-1.hf.co','us.aws.cdn.hf.co'}


class SafeError(Exception):pass
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def load_validator(expected_sha):
    path=Path(__file__).with_name('hf_to_stream_oss_cache_v3.py')
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha:
        raise SafeError('local_validator_sha_changed')
    spec=importlib.util.spec_from_file_location('frozen_cache_validator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.validate_plan


def resolve_signed(asset,token,opener):
    # Source identity remains bound to the original pinned plan. The sole
    # accepted repository rename was observed on the official HF endpoint.
    aliases={asset['repo_id']}
    if asset['repo_id']=='lmms-lab/Video-MME':aliases.add('lmms-eval/Video-MME')
    allowed_paths={'/datasets/'+repo+'/resolve/'+asset['revision']+'/'+asset['path'] for repo in aliases}
    url='https://huggingface.co/datasets/'+asset['repo_id']+'/resolve/'+asset['revision']+'/'+asset['path']
    visited=set()
    for _ in range(MAX_REDIRECTS):
        parsed=urllib.parse.urlsplit(url)
        if (parsed.scheme!='https' or parsed.hostname!='huggingface.co' or parsed.port not in (None,443)
                or parsed.username or parsed.password or parsed.fragment or parsed.path not in allowed_paths):
            raise SafeError('hf_redirect_source_identity_changed')
        if url in visited:raise SafeError('hf_redirect_cycle')
        visited.add(url)
        # NoRedirect prevents urllib from forwarding this header anywhere.
        # A new credential-bearing Request is built only after the HF check.
        request=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Range':'bytes=0-0',
            'Accept-Encoding':'identity','Cache-Control':'no-cache'})
        try:response=opener.open(request,timeout=45)
        except urllib.error.HTTPError as error:response=error
        with response:
            if response.code not in REDIRECT_CODES:raise SafeError('hf_resolve_non_redirect')
            location=response.headers.get('Location')
            if (not isinstance(location,str) or not 0<len(location)<=16384
                    or any(ord(c)<=32 or ord(c)>=127 for c in location)):
                raise SafeError('hf_redirect_location_invalid')
            following=urllib.parse.urljoin(url,location)
        target=urllib.parse.urlsplit(following)
        if (target.scheme!='https' or target.port not in (None,443) or target.username
                or target.password or target.fragment):raise SafeError('hf_redirect_target_unsafe')
        if target.hostname=='huggingface.co':
            if target.path not in allowed_paths:raise SafeError('hf_redirect_source_identity_changed')
            url=following
            continue
        if target.hostname not in CDN_HOSTS:raise SafeError('hf_redirect_cdn_host_unapproved')
        # Return the signed URL without requesting it here. The remote frozen
        # worker performs its own CDN validation and complete source SHA check.
        return following
    raise SafeError('hf_redirect_limit_exceeded')


def safe_result(result):
    if not isinstance(result,dict) or not set(result)<=SAFE_FIELDS:
        raise SafeError('remote_result_fields_invalid')
    numeric={'bytes','archive_count','total_bytes','archives'}
    hashes={'sha256','receipt_sha256','plan_sha256','catalog_sha256'}
    for key,value in result.items():
        if key in numeric:
            if type(value) is not int or value<0:raise SafeError('remote_result_number_invalid')
        elif key in hashes:
            if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
                raise SafeError('remote_result_sha_invalid')
        elif key=='all_benchmark_assets_ready':
            if value is not False:raise SafeError('remote_readiness_unexpected')
        elif not isinstance(value,str) or not re.fullmatch('[A-Za-z0-9:_-]{1,256}',value):
            raise SafeError('remote_result_text_invalid')
    if not isinstance(result.get('status'),str):raise SafeError('remote_result_status_missing')
    return result


def stream_json(line):
    def unique(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise SafeError('remote_stream_duplicate_field')
            value[key]=item
        return value
    try:value=json.loads(line,object_pairs_hook=unique)
    except (ValueError,UnicodeError):raise SafeError('remote_stream_non_json') from None
    if not isinstance(value,dict):raise SafeError('remote_stream_not_object')
    return value


def run_stream_upload(command,asset,signed,resolver,popen_factory=None,timeout=STREAM_TIMEOUT_SECONDS):
    """Maintain one SSH worker while resolving replacement URLs only locally.

    Binary, nonblocking pipes avoid TextIOWrapper read-ahead and keep every wait
    bounded. Raw stdout errors and all stderr are discarded rather than logged.
    """
    process=(popen_factory or subprocess.Popen)(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                                 stderr=subprocess.PIPE,bufsize=0)
    poller=selectors.DefaultSelector();deadline=time.monotonic()+timeout
    initial={'asset_id':asset['asset_id']}
    if signed is not None:initial['signed_url']=signed
    pending=(json.dumps(initial,separators=(',',':'))+'\n').encode();buffer=bytearray()
    refresh_count=0;pending_refresh=None;terminal=None
    streams={'stdin':process.stdin,'stdout':process.stdout,'stderr':process.stderr}
    def close_input():
        if not process.stdin.closed:
            try:poller.unregister(process.stdin)
            except KeyError:pass
            process.stdin.close()
    try:
        for name,stream in streams.items():
            os.set_blocking(stream.fileno(),False)
            poller.register(stream,selectors.EVENT_WRITE if name=='stdin' else selectors.EVENT_READ,name)
        while poller.get_map():
            remaining=deadline-time.monotonic()
            if remaining<=0:raise SafeError('remote_stream_deadline_exceeded')
            for ready,_ in poller.select(min(5,remaining)):
                name=ready.data;stream=ready.fileobj
                if name=='stdin':
                    try:count=os.write(stream.fileno(),pending)
                    except BlockingIOError:continue
                    if count<=0:raise SafeError('remote_stream_stdin_closed')
                    pending=pending[count:]
                    if not pending:
                        poller.unregister(stream)
                        if pending_refresh is not None:
                            print(json.dumps({'event':'HF_SIGNED_URL_REFRESH_RESPONSE_SENT',
                                              'asset_id':asset['asset_id'],
                                              'refresh_count':pending_refresh}),flush=True)
                            pending_refresh=None
                    continue
                try:data=os.read(stream.fileno(),4096)
                except BlockingIOError:continue
                if not data:
                    poller.unregister(stream)
                    if name=='stdout' and buffer:raise SafeError('remote_stream_partial_line')
                    continue
                if name=='stderr':continue
                buffer.extend(data)
                if len(buffer)>STREAM_LINE_LIMIT:raise SafeError('remote_stream_line_oversized')
                while b'\n' in buffer:
                    line,rest=bytes(buffer).split(b'\n',1);buffer=bytearray(rest)
                    if terminal is not None:raise SafeError('remote_stream_output_after_terminal')
                    message=stream_json(line)
                    if message.get('status')=='SIGNED_URL_REFRESH_REQUIRED':
                        if set(message)!={'status','asset_id','refresh_count'}:
                            raise SafeError('remote_refresh_fields_invalid')
                        if message['asset_id']!=asset['asset_id']:raise SafeError('remote_refresh_asset_mismatch')
                        count=message['refresh_count']
                        if type(count) is not int or count!=refresh_count+1 or count>MAX_STREAM_REFRESHES:
                            raise SafeError('remote_refresh_sequence_invalid')
                        if pending:raise SafeError('remote_stream_input_backlog')
                        # resolver closes over the local in-memory HF token;
                        # only its returned signed CDN URL crosses SSH stdin.
                        fresh=resolver(asset)
                        if time.monotonic()>=deadline:raise SafeError('remote_stream_deadline_exceeded')
                        response={'asset_id':asset['asset_id'],'refresh_count':count,'signed_url':fresh}
                        pending=(json.dumps(response,separators=(',',':'))+'\n').encode()
                        if len(pending)>STREAM_LINE_LIMIT:raise SafeError('remote_refresh_response_oversized')
                        poller.register(process.stdin,selectors.EVENT_WRITE,'stdin')
                        refresh_count=count;pending_refresh=count
                    else:
                        if pending or pending_refresh is not None:
                            raise SafeError('remote_terminal_before_input_consumed')
                        terminal=safe_result(message)
                        if terminal['status'] not in {'CACHED_VERIFIED','FAILED'}:
                            raise SafeError('remote_upload_terminal_status_invalid')
                        if terminal['status']=='CACHED_VERIFIED':
                            required={'status','asset_id','bytes','sha256','receipt_sha256'}
                            if set(terminal) not in (required,required|{'source_mode'}):
                                raise SafeError('remote_upload_success_fields_invalid')
                        elif set(terminal)!={'status','error_code'}:
                            raise SafeError('remote_upload_failure_fields_invalid')
                        close_input()
        remaining=deadline-time.monotonic()
        if remaining<=0:raise SafeError('remote_stream_deadline_exceeded')
        returncode=process.wait(timeout=remaining)
        if terminal is None or returncode not in (0,2):raise SafeError('remote_stream_transport_uncertain')
        if returncode==0:
            if (terminal.get('status')!='CACHED_VERIFIED' or terminal.get('asset_id')!=asset['asset_id'] or
                    terminal.get('bytes')!=asset['size_bytes'] or terminal.get('sha256')!=asset['sha256']):
                raise SafeError('remote_upload_success_binding_mismatch')
        elif terminal.get('status')!='FAILED':raise SafeError('remote_upload_exit_status_mismatch')
        print(json.dumps(terminal,sort_keys=True),flush=True)
        return returncode,terminal
    except Exception:
        # EOF lets a live worker fail closed at its next range/part boundary or
        # control read. Close only this call's SSH child, never another worker.
        close_input()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=5)
        raise
    finally:
        poller.close()
        for stream in streams.values():
            if not stream.closed:stream.close()


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
    def refresh_signed(asset):
        # <=3 *6 *45s +3s backoff, within worker's900s response deadline.
        for attempt in range(3):
            try:return resolve_signed(asset,token,opener)
            except Exception:
                if attempt==2:raise SafeError('hf_live_refresh_resolution_exhausted') from None
                time.sleep(2**attempt)
    def call(mode,asset=None,signed=None):
        command=[config['python'],'-c',BOOTSTRAP,config['worker'],config['worker_sha256'],mode,
                 '--plan',config['remote_plan'],'--plan-sha256',config['plan_sha256']]
        if mode!='status':command.append('--execute')
        if asset:command+=['--asset-id',asset['asset_id']]
        if mode=='upload':
            command.append('--stream-control')
            return run_stream_upload(ssh+[shlex.join(command)],asset,signed,refresh_signed)
        response=subprocess.run(ssh+[shlex.join(command)],input=json.dumps({'signed_url':signed}) if signed else '',
                      text=True,capture_output=True,timeout=6*3600)
        try:result=safe_result(stream_json(response.stdout))
        except Exception:raise SafeError('remote_transport_or_protocol_uncertain') from None
        if response.returncode not in (0,2):raise SafeError('remote_transport_uncertain')
        print(json.dumps(result),flush=True)
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
