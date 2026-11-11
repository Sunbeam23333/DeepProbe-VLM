"""GPU-only identity probe; no kernels, models, downloads or CPU-mode torch import.

The caller still runs the original all-rank BF16/NCCL preflight afterwards.
"""
import sys
sys.dont_write_bytecode = True
import argparse, csv, hashlib, importlib.util, io, json, os, re, stat, subprocess
from pathlib import Path

PREFLIGHT_SHA256 = '6c8894ba63cc3860ad5aa574743c1ea73bd32de73f7de5a2e1947c1ce529f990'
EXPECTED_GPU = 'B300'
FULL_MASK = '0,1,2,3,4,5,6,7'
SMI_ARGV = ['nvidia-smi', '--query-gpu=index,name,uuid', '--format=csv,noheader']
UUID_RE = r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def fingerprint(s):
    return (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def source_bytes(path, expected_sha):
    """Read only the exact reviewed bytes, rejecting links and concurrent edits."""
    path = Path(path)
    need(expected_sha == PREFLIGHT_SHA256, 'Unexpected frozen preflight SHA')
    need(path.is_absolute() and '..' not in path.parts
         and not any(p.is_symlink() for p in (path, *path.parents)), 'Unsafe preflight source path')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 128 * 1024, 'Invalid preflight source file')
        raw = stream.read(128 * 1024 + 1)
        after = os.fstat(stream.fileno())
    need(fingerprint(before) == fingerprint(after) == fingerprint(path.lstat()), 'Preflight source changed during read')
    need(len(raw) == before.st_size and hashlib.sha256(raw).hexdigest() == expected_sha, 'Frozen preflight source bytes changed')
    return raw, fingerprint(before)


def load_matcher(raw, path):
    # Import from the bytes already SHA-verified, never reopen with exec_module.
    spec = importlib.util.spec_from_loader('_frozen_preflight_gpu_identity', loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    need(callable(getattr(module, 'matches_expected_gpu', None)), 'Frozen GPU validator missing')
    return module.matches_expected_gpu


def parse_smi(text):
    need(isinstance(text, str) and 0 < len(text.encode()) <= 64 * 1024, 'Invalid GPU inventory length')
    rows = []
    for fields in csv.reader(io.StringIO(text)):
        if not fields:
            continue
        need(len(fields) == 3, 'Malformed nvidia-smi row')
        index, name, uuid = (field.strip() for field in fields)
        need(re.fullmatch('[0-7]', index) and name and re.fullmatch(UUID_RE, uuid), 'Malformed GPU identity')
        rows.append({'index': int(index), 'name': name, 'uuid': uuid})
    need(len(rows) == 8 and {r['index'] for r in rows} == set(range(8)), 'Exactly indices 0 through 7 required')
    need(len({r['uuid'].lower() for r in rows}) == 8, 'Eight unique GPU UUIDs required')
    return sorted(rows, key=lambda row: row['index'])


def validate_mask(visible):
    need(visible is None or visible == FULL_MASK, 'Unexpected CUDA_VISIBLE_DEVICES mask')


def uuid_text(value):
    # torch versions exposing device UUIDs may return a UUID object or text.
    text = str(value)
    return text if text.startswith('GPU-') else 'GPU-' + text


def inspect_devices(rows, torch, matcher, visible):
    validate_mask(visible)
    need(torch.cuda.is_available() and torch.cuda.device_count() == 8, 'Exactly eight visible CUDA devices required')
    devices = []
    for row in rows:
        i = row['index']; props = torch.cuda.get_device_properties(i)
        need(isinstance(props.name, str) and props.name
             and type(props.major) is int and type(props.minor) is int
             and type(props.total_memory) is int and props.total_memory > 0, 'Invalid actual CUDA properties')
        gpu = {'index': i, 'name': props.name, 'compute_capability': [props.major, props.minor],
               'total_memory_gib': props.total_memory / (1024 ** 3), 'total_memory_bytes': props.total_memory,
               'uuid': row['uuid'], 'nvidia_smi_name': row['name']}
        need(matcher(gpu, EXPECTED_GPU)
             and matcher({**gpu, 'name': row['name']}, EXPECTED_GPU), 'GPU name/architecture/memory does not match frozen B300 rule')
        actual_uuid = getattr(props, 'uuid', None)
        if actual_uuid is not None:
            actual_uuid = uuid_text(actual_uuid)
            need(re.fullmatch(UUID_RE, actual_uuid)
                 and actual_uuid.lower() == row['uuid'].lower(), 'CUDA and nvidia-smi UUID/index mapping differs')
            gpu['torch_uuid'] = actual_uuid; gpu['uuid_binding'] = 'torch_device_property'
        else:
            gpu['uuid_binding'] = 'nvidia_smi_index_complete_unmasked_inventory'
        devices.append(gpu)
    need(torch.cuda.device_count() == 8, 'CUDA device count changed during probe')
    return devices


def smi():
    result = subprocess.run(SMI_ARGV, capture_output=True, text=True, timeout=30, check=False)
    need(result.returncode == 0 and not result.stderr, 'nvidia-smi failed')
    return parse_smi(result.stdout)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--preflight-source', required=True, type=Path)
    parser.add_argument('--preflight-sha256', required=True)
    args = parser.parse_args(argv)
    raw, identity = source_bytes(args.preflight_source, args.preflight_sha256)
    matcher = load_matcher(raw, args.preflight_source)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES'); validate_mask(visible)
    rows = smi()
    # Imported only by this explicit GPU CLI, after source/allocation checks.
    import torch
    devices = inspect_devices(rows, torch, matcher, visible)
    need(smi() == rows and os.environ.get('CUDA_VISIBLE_DEVICES') == visible, 'GPU allocation changed during probe')
    final_raw, final_identity = source_bytes(args.preflight_source, args.preflight_sha256)
    need(final_raw == raw and final_identity == identity, 'Preflight source changed during probe')
    print(json.dumps({'status': 'GPU_IDENTITY_ACCEPTED', 'expected_gpu': EXPECTED_GPU,
                      'preflight_source_sha256': args.preflight_sha256,
                      'visible_device_count': 8, 'cuda_visible_devices': visible, 'devices': devices,
                      'kernels_tested': False, 'bf16_nccl_preflight_passed': False,
                      'requires_original_all_rank_preflight': True}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
