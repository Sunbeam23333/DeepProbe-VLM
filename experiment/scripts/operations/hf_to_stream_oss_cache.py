#!/usr/bin/env python3
"""Cache pinned HF archives in stream OSS with bounded RAM and no local ZIP writes.

All mutation modes require --execute. The sole stdin secret is a signed source
URL. Existing OSS credentials are loaded on the remote host only, in memory.
"""
from __future__ import annotations
import argparse
import configparser
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import urllib.parse
import urllib.request

BUCKET = 'ss-devbox-stream'
PREFIX = 'jinhaopeng/Paper/DeepProbe-VLM/benchmark-cache/20261013-v1/hf-archives/'
ENDPOINT = 'https://oss-cn-wulanchabu.aliyuncs.com'
P1 = Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM-p1')
CONTROL = P1 / 'benchmark-stream-cache-20261013/control'
OSS_CONFIG = Path('/home/jinhaopeng/.ossutilconfig')
PART_SIZE = 16 * 1024 * 1024
NO_OVERWRITE = {'x-oss-forbid-overwrite': 'true'}
REVISIONS = {'OpenGVLab/MVBench': '230a2d4fac8900333c61754641c7a13e069ac9c6',
             'lmms-lab/Video-MME': 'ead1408f75b618502df9a1d8e0950166bf0a2a0b'}
CDNS = {'cas-bridge.xethub.hf.co', 'cdn-lfs.hf.co', 'cdn-lfs-us-1.hf.co',
        'cdn-lfs-eu-1.hf.co', 'us.aws.cdn.hf.co'}
MV_NAMES = {'clevrer','data0613','FunQA_test','Moments_in_Time_Raw','perception',
            'scene_qa','ssv2_video','sta','star','tvqa','vlnqa'}
LOCAL_MV = {'ssv2_video','vlnqa','data0613','Moments_in_Time_Raw'}


class SafeError(Exception):
    pass


def require(value, code):
    if not value: raise SafeError(code)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def sha_bytes(data): return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(8*1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def safe_path(path):
    path=Path(path)
    require(all(not item.is_symlink() for item in [path,*path.parents]), 'symlink_path')
    return path


def save_state(path, value):
    safe_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.writing-',dir=path.parent)
    with os.fdopen(fd,'wb') as stream:
        stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
    os.replace(name,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


@contextmanager
def cache_lock():
    safe_path(CONTROL);CONTROL.mkdir(parents=True,exist_ok=True)
    fd=os.open(safe_path(CONTROL/'.cache.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:os.close(fd)


def validate_plan(plan):
    require(plan.get('bucket')==BUCKET and plan.get('prefix')==PREFIX and
            plan.get('endpoint')==ENDPOINT and plan.get('region')=='cn-wulanchabu', 'unapproved_destination')
    require(plan.get('part_size_bytes')==PART_SIZE,'unapproved_part_size')
    assets=plan.get('assets',[])
    expected={('OpenGVLab/MVBench','video/'+name+'.zip') for name in MV_NAMES}
    expected|={('lmms-lab/Video-MME',f'videos_chunked_{n:02d}.zip') for n in range(1,21)}
    require(len(assets)==31 and {(a.get('repo_id'),a.get('path')) for a in assets}==expected,'archive_set_changed')
    require(len({a['asset_id'] for a in assets})==31,'duplicate_asset_id')
    for a in assets:
        require(a.get('revision')==REVISIONS[a['repo_id']], 'source_revision_changed')
        require(type(a.get('size_bytes')) is int and 0<a['size_bytes']<=12_000_000_000,'invalid_source_size')
        require(bool(re.fullmatch('[0-9a-f]{64}',a.get('sha256',''))),'invalid_source_sha')
        require(bool(re.fullmatch('[A-Za-z0-9:_-]+',a['asset_id'])),'invalid_asset_id')
        if a.get('local_source'):
            require(Path(a['local_source'])==approved_local_path(a),'unapproved_local_source')
    require(plan.get('archive_count')==31 and plan.get('total_bytes')==sum(a['size_bytes'] for a in assets),'plan_totals_changed')
    return assets


def approved_local_path(asset):
    if asset['repo_id']=='OpenGVLab/MVBench' and Path(asset['path']).stem in LOCAL_MV:
        return P1/'benchmark-mvbench-assets-20261013'/asset['path']
    if asset['repo_id']=='lmms-lab/Video-MME' and asset['path']=='videos_chunked_11.zip':
        return P1/'benchmark-bucket-transition-20261013/videos_chunked_11.zip'
    return None


def key_for(asset): return PREFIX+asset['repo_id']+'/'+asset['revision']+'/'+asset['path']
def receipt_key(asset): return key_for(asset)+'.CACHE_VERIFIED.json'

def binding(asset,plan_sha):
    return {'plan_sha256':plan_sha,'bucket':BUCKET,'key':key_for(asset),'repo_id':asset['repo_id'],
            'revision':asset['revision'],'source_path':asset['path'],'size_bytes':asset['size_bytes'],'sha256':asset['sha256']}


class OSSStore:
    """Thin SDK adapter; exceptions are never rendered with their raw text."""
    def __init__(self):
        import oss2
        require(oss2.__version__=='2.19.1','unreviewed_oss_sdk')
        config=configparser.ConfigParser();config.read(safe_path(OSS_CONFIG))
        require(config.has_section('default') and config['default'].get('region')=='cn-wulanchabu','oss_region_changed')
        auth=oss2.Auth(config['default']['accesskeyid'],config['default']['accesskeysecret'])
        self.bucket=oss2.Bucket(auth,ENDPOINT,BUCKET,connect_timeout=60,enable_crc=True)
        self.oss2=oss2
    def missing(self,error): return isinstance(error,self.oss2.exceptions.NoSuchKey)
    def exists(self,key):
        try:self.bucket.head_object(key);return True
        except Exception as error:
            if self.missing(error):return False
            raise
    def open(self,key):return self.bucket.get_object(key)
    def get_small(self,key,maximum=2_000_000):
        try:stream=self.open(key)
        except Exception as error:
            if self.missing(error):return None
            raise
        try:
            value=stream.read(maximum+1);require(len(value)<=maximum,'metadata_oversized');return value
        finally:stream.close()
    def put_new(self,key,data):return self.bucket.put_object(key,data,headers=NO_OVERWRITE)
    def keys(self):
        marker=''
        while True:
            page=self.bucket.list_objects(prefix=PREFIX,marker=marker,max_keys=1000)
            for item in page.object_list:yield item.key
            if not page.is_truncated:return
            require(page.next_marker and page.next_marker!=marker,'object_pagination_stalled');marker=page.next_marker
    def uploads(self,key):
        key_marker=upload_marker=''
        while True:
            page=self.bucket.list_multipart_uploads(prefix=key,key_marker=key_marker,
                         upload_id_marker=upload_marker,max_uploads=1000)
            for item in page.upload_list:
                if item.key==key:yield item.upload_id
            if not page.is_truncated:return
            nxt=(page.next_key_marker,page.next_upload_id_marker)
            require(nxt!=(key_marker,upload_marker),'upload_pagination_stalled')
            key_marker,upload_marker=nxt
    def initiate(self,key,asset):
        headers={**NO_OVERWRITE,'x-oss-meta-sha256':asset['sha256'],'x-oss-meta-source-revision':asset['revision']}
        return self.bucket.init_multipart_upload(key,headers=headers).upload_id
    def parts(self,key,upload_id):
        marker='0'
        while True:
            page=self.bucket.list_parts(key,upload_id,marker=marker,max_parts=1000)
            for part in page.parts:yield {'number':part.part_number,'etag':part.etag,'size_bytes':part.size}
            if not page.is_truncated:return
            nxt=str(page.next_marker);require(nxt!=marker,'part_pagination_stalled');marker=nxt
    def upload(self,key,upload_id,number,data):
        return self.bucket.upload_part(key,upload_id,number,data).etag
    def complete(self,key,upload_id,parts):
        values=[self.oss2.models.PartInfo(p['number'],p['etag']) for p in parts]
        return self.bucket.complete_multipart_upload(key,upload_id,values,headers=NO_OVERWRITE)


def put_immutable(store,key,data):
    existing=store.get_small(key)
    if existing is not None:
        require(existing==data,'immutable_object_conflict');return
    try:store.put_new(key,data)
    except Exception:
        # A lost response or same-plan race can only resolve to identical bytes.
        require(store.get_small(key)==data,'immutable_object_publication_failed')
    require(store.get_small(key)==data,'metadata_readback_mismatch')


def initialize(store,plan,plan_sha):
    expected={PREFIX+'CACHE_PLAN.json',PREFIX+'ARCHIVE_CATALOG_COMPLETE.json'}
    for a in plan['assets']:expected.update({key_for(a),receipt_key(a)})
    existing=set(store.keys());require(existing<=expected,'unexpected_existing_prefix_object')
    receipt={'plan_sha256':plan_sha,'plan':plan}
    if PREFIX+'CACHE_PLAN.json' not in existing:
        require(not existing,'prefix_not_empty_without_plan')
        for a in plan['assets']:require(not list(store.uploads(key_for(a))),'unowned_multipart_upload')
    put_immutable(store,PREFIX+'CACHE_PLAN.json',canonical(receipt))
    return {'status':'CACHE_INITIALIZED','plan_sha256':plan_sha,'archive_count':31,'total_bytes':plan['total_bytes']}


def require_initialized(store,plan,plan_sha):
    require(store.get_small(PREFIX+'CACHE_PLAN.json')==canonical({'plan_sha256':plan_sha,'plan':plan}),'cache_plan_missing_or_changed')


def verify_oss_object(store,asset):
    digest=hashlib.sha256();count=0;stream=store.open(key_for(asset))
    try:
        while True:
            data=stream.read(min(8*1024*1024,asset['size_bytes']-count+1))
            if not data:break
            count+=len(data);require(count<=asset['size_bytes'],'oss_object_oversized');digest.update(data)
    finally:stream.close()
    require(count==asset['size_bytes'] and digest.hexdigest()==asset['sha256'],'oss_full_readback_mismatch')
    return {'bytes':count,'sha256':digest.hexdigest(),'verification':'complete_OSS_GET_SHA256_and_size'}


def status(store,asset,plan_sha):
    if store.exists(key_for(asset)):
        verified=verify_oss_object(store,asset)
        raw=store.get_small(receipt_key(asset))
        if raw is None:return {'status':'OBJECT_VERIFIED_RECEIPT_PENDING','asset_id':asset['asset_id']}
        receipt=json.loads(raw)
        require(receipt.get('binding')==binding(asset,plan_sha) and receipt.get('oss_readback')==verified
                and receipt.get('source_stream_sha256')==asset['sha256'],'cache_receipt_conflict')
        return {'status':'CACHED_VERIFIED','asset_id':asset['asset_id'],'bytes':asset['size_bytes'],
                'sha256':asset['sha256'],'receipt_sha256':sha_bytes(raw)}
    require(store.get_small(receipt_key(asset)) is None,'receipt_without_payload')
    path=approved_local_path(asset)
    if asset.get('local_source'):
        safe_path(path);require(path.is_file() and path.stat().st_size==asset['size_bytes'],'required_local_source_missing_or_changed')
        return {'status':'READY_LOCAL_SOURCE','asset_id':asset['asset_id']}
    return {'status':'NEEDS_SIGNED_URL','asset_id':asset['asset_id']}


def validate_url(url):
    require(isinstance(url,str) and len(url)<=16384 and all(32<ord(c)<127 for c in url),'invalid_signed_url')
    parsed=urllib.parse.urlsplit(url)
    require(parsed.scheme=='https' and parsed.hostname in CDNS and parsed.port in (None,443)
            and not parsed.username and not parsed.password and not parsed.fragment,'unapproved_cdn_url')
    return url


class CDNRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,response,code,message,headers,newurl):
        validate_url(newurl)
        return super().redirect_request(request,response,code,message,headers,newurl)


@contextmanager
def open_source(asset,url,opener=None):
    path=approved_local_path(asset)
    if asset.get('local_source'):
        safe_path(path);require(path.is_file(),'required_local_source_missing_or_changed')
        before=path.stat();require(before.st_size==asset['size_bytes'],'local_source_size_changed')
        with path.open('rb') as stream:yield stream,'verified_local_candidate'
        after=path.stat();require((before.st_ino,before.st_size,before.st_mtime_ns)==
                                 (after.st_ino,after.st_size,after.st_mtime_ns),'local_source_changed_during_read')
    else:
        validate_url(url)
        opener=opener or urllib.request.build_opener(CDNRedirect(),urllib.request.ProxyHandler({}))
        request=urllib.request.Request(url,headers={'Range':f"bytes=0-{asset['size_bytes']-1}",'Accept-Encoding':'identity'})
        with opener.open(request,timeout=60) as stream:
            require(stream.status==206 and stream.headers.get('Content-Range')==
                    f"bytes 0-{asset['size_bytes']-1}/{asset['size_bytes']}",'hf_range_identity_mismatch')
            yield stream,'pinned_hf_range_stream'


def read_part(source,maximum):
    chunks=[];remaining=maximum
    while remaining:
        chunk=source.read(remaining)
        if not chunk:break
        require(len(chunk)<=remaining,'source_reader_exceeded_requested_limit')
        chunks.append(chunk);remaining-=len(chunk)
    return b''.join(chunks)


def upload_asset(store,asset,plan_sha,url=None,source_opener=None):
    bind=binding(asset,plan_sha);key=key_for(asset)
    state_path=CONTROL/(sha_bytes(key.encode())+'.json')
    state=None
    if state_path.exists():
        safe_path(state_path);state=json.loads(state_path.read_text())
        require(state.get('binding')==bind,'multipart_state_binding_changed')
    if store.exists(key):
        verification=verify_oss_object(store,asset)
        existing=store.get_small(receipt_key(asset))
        if existing is not None:
            require(json.loads(existing).get('binding')==bind,'cache_receipt_conflict')
            return status(store,asset,plan_sha)
        # The final content digest is also the source digest; a completion whose
        # response was lost can be recovered without any source/network replay.
        receipt={'binding':bind,'source_stream_sha256':asset['sha256'],'source_bytes':asset['size_bytes'],
                 'oss_readback':verification,'source_mode':'recovered_pinned_final_object','formal_evaluation_ready':False}
        put_immutable(store,receipt_key(asset),canonical(receipt))
        return status(store,asset,plan_sha)
    if state is None:
        require(not list(store.uploads(key)),'unowned_multipart_upload_requires_inspection')
        state={'binding':bind,'parts':{},'phase':'intent_before_multipart_init'};save_state(state_path,state)
        upload_id=store.initiate(key,asset)
        state.update(upload_id=upload_id,phase='uploading');save_state(state_path,state)
    require(isinstance(state.get('upload_id'),str) and state['upload_id'],'multipart_init_outcome_uncertain')
    remote_parts={part['number']:part for part in store.parts(key,state['upload_id'])}
    require(len(remote_parts)<=((asset['size_bytes']+PART_SIZE-1)//PART_SIZE),'unexpected_extra_remote_parts')
    digest=hashlib.sha256();total=0;parts=[]
    with open_source(asset,url,source_opener) as (source,source_mode):
        number=0
        while total<asset['size_bytes']:
            data=read_part(source,min(PART_SIZE,asset['size_bytes']-total))
            require(data,'truncated_source_stream');number+=1;total+=len(data);digest.update(data)
            part_sha=sha_bytes(data);saved=state['parts'].get(str(number));remote=remote_parts.get(number)
            require(len(data)==PART_SIZE or total==asset['size_bytes'],'short_nonfinal_part')
            if not (saved and remote and saved.get('sha256')==part_sha and saved.get('size_bytes')==len(data)
                    and saved.get('etag')==remote['etag'] and remote['size_bytes']==len(data)):
                etag=store.upload(key,state['upload_id'],number,data)
                saved={'number':number,'size_bytes':len(data),'sha256':part_sha,'etag':etag}
                state['parts'][str(number)]=saved;save_state(state_path,state)
            parts.append(saved)
        require(not source.read(1),'source_stream_oversized')
    require(total==asset['size_bytes'] and digest.hexdigest()==asset['sha256'],'source_full_sha_mismatch')
    require(set(remote_parts)<=set(range(1,len(parts)+1)),'unexpected_remote_part_numbers')
    state.update(source_stream_sha256=digest.hexdigest(),source_bytes=total,phase='source_verified_before_complete')
    save_state(state_path,state)
    # This header must be applied to CompleteMultipartUpload itself.
    store.complete(key,state['upload_id'],parts)
    verification=verify_oss_object(store,asset)
    receipt={'binding':bind,'source_stream_sha256':digest.hexdigest(),'source_bytes':total,
             'oss_readback':verification,'source_mode':source_mode,'formal_evaluation_ready':False}
    put_immutable(store,receipt_key(asset),canonical(receipt))
    state.update(phase='oss_readback_verified',receipt_sha256=sha_bytes(canonical(receipt)))
    save_state(state_path,state)
    return {'status':'CACHED_VERIFIED','asset_id':asset['asset_id'],'bytes':total,'sha256':digest.hexdigest(),
            'receipt_sha256':sha_bytes(canonical(receipt)),'source_mode':source_mode}


def finalize(store,plan,plan_sha):
    entries=[]
    for asset in plan['assets']:
        result=status(store,asset,plan_sha);require(result['status']=='CACHED_VERIFIED','archive_catalog_incomplete')
        entries.append({**binding(asset,plan_sha),'receipt_sha256':result['receipt_sha256']})
    catalog={'plan_sha256':plan_sha,'archive_count':31,'total_bytes':plan['total_bytes'],
             'entries':entries,'all_pinned_zip_archives_cached':True,'ntu_media_included':False,
             'formal_evaluation_ready':False,'all_benchmark_assets_ready':False}
    put_immutable(store,PREFIX+'ARCHIVE_CATALOG_COMPLETE.json',canonical(catalog))
    return {'status':'ARCHIVE_CATALOG_COMPLETE','archives':31,'bytes':plan['total_bytes'],
            'catalog_sha256':sha_bytes(canonical(catalog)),'all_benchmark_assets_ready':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['initialize','status','upload','finalize'])
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--asset-id');parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    raw=safe_path(args.plan).read_bytes();require(sha_bytes(raw)==args.plan_sha256,'plan_sha_changed')
    plan=json.loads(raw);assets=validate_plan(plan)
    require(args.mode=='status' or args.execute,'mutation_requires_execute')
    payload_text=sys.stdin.read(20001);require(len(payload_text)<=20000,'stdin_payload_oversized')
    payload=json.loads(payload_text) if payload_text.strip() else {}
    require(isinstance(payload,dict) and set(payload)<={'signed_url'},'unexpected_stdin_fields')
    store=OSSStore()
    if args.mode=='status':
        require_initialized(store,plan,args.plan_sha256)
        asset=next(a for a in assets if a['asset_id']==args.asset_id)
        result=status(store,asset,args.plan_sha256)
    else:
        with cache_lock():
            if args.mode=='initialize':result=initialize(store,plan,args.plan_sha256)
            else:
                require_initialized(store,plan,args.plan_sha256)
                if args.mode=='finalize':result=finalize(store,plan,args.plan_sha256)
                else:
                    asset=next(a for a in assets if a['asset_id']==args.asset_id)
                    result=upload_asset(store,asset,args.plan_sha256,payload.get('signed_url'))
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        code=str(error) if isinstance(error,SafeError) else 'operation_'+type(error).__name__
        print(json.dumps({'status':'FAILED','error_code':code}),flush=True);sys.exit(2)
