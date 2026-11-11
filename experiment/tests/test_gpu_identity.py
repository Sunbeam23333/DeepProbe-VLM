"""Mocked device/source regression tests. Never imports real torch or uses a GPU."""
import builtins, contextlib, copy, importlib.util, io, json, os, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.dont_write_bytecode = True
ROOT = Path(__file__).absolute().parent
SOURCE = ROOT.parent / 'scripts/preflight.py'


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT.parent / 'scripts/operations/verify_gpu_identity.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


g = load_helper('identity_under_test')
SOURCE_RAW, _ = g.source_bytes(SOURCE, g.PREFLIGHT_SHA256)
MATCHER = g.load_matcher(SOURCE_RAW, SOURCE)


def uuid(i):
    return 'GPU-00000000-0000-0000-0000-%012x' % (i + 1)


def smi_rows(name='NVIDIA-L20D'):
    return [{'index':i, 'name':name, 'uuid':uuid(i)} for i in range(8)]


def render(rows):
    return ''.join('%s, %s, %s\n' % (r['index'], r['name'], r['uuid']) for r in rows)


def fake_torch(name='NVIDIA-L20D', capability=(10,3), gib=240, count=8):
    properties = [SimpleNamespace(name=name, major=capability[0], minor=capability[1],
                  total_memory=int(gib * 1024**3), uuid=uuid(i)) for i in range(8)]
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True, device_count=lambda:count,
                           get_device_properties=lambda i:properties[i]), properties=properties)


class IdentityTests(unittest.TestCase):
    def test_genuine_alias_uses_frozen_rule_and_raw_identity(self):
        rows = g.parse_smi(render(smi_rows()))
        devices = g.inspect_devices(rows, fake_torch(), MATCHER, g.FULL_MASK)
        self.assertEqual(len(devices), 8)
        self.assertEqual(devices[0]['name'], 'NVIDIA-L20D')
        self.assertEqual(devices[0]['compute_capability'], [10,3])
        self.assertEqual(devices[0]['total_memory_gib'], 240)
        self.assertEqual(devices[0]['uuid'], uuid(0))
        self.assertEqual(devices[0]['uuid_binding'], 'torch_device_property')

    def test_real_b300_name_passes_only_with_correct_properties(self):
        g.inspect_devices(smi_rows('NVIDIA B300'), fake_torch(name='NVIDIA B300'), MATCHER, None)
        for capability in [(9,0),(10,0),(12,0)]:
            with self.subTest(capability=capability), self.assertRaises(ValueError):
                g.inspect_devices(smi_rows('NVIDIA B300'), fake_torch(name='NVIDIA B300',capability=capability), MATCHER, None)

    def test_alias_wrong_cc_or_low_vram_rejected(self):
        for kwargs in [{'capability':(9,0)}, {'capability':(10,0)}, {'gib':239.999}, {'gib':48}]:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'frozen B300 rule'):
                g.inspect_devices(smi_rows(), fake_torch(**kwargs), MATCHER, None)

    def test_bad_nonzero_device_is_not_hidden_by_device_zero(self):
        for index in range(1,8):
            torch=fake_torch();torch.properties[index].major=9
            with self.subTest(index=index), self.assertRaisesRegex(ValueError,'frozen B300 rule'):
                g.inspect_devices(smi_rows(),torch,MATCHER,None)

    def test_unknown_smi_or_cuda_name_rejected(self):
        for smi_name, torch_name in [('NVIDIA Unknown','NVIDIA-L20D'), ('NVIDIA-L20D','NVIDIA Unknown'),
                                     ('NVIDIA L20','NVIDIA L20')]:
            with self.subTest(names=(smi_name,torch_name)), self.assertRaises(ValueError):
                g.inspect_devices(smi_rows(smi_name), fake_torch(name=torch_name), MATCHER, None)

    def test_duplicate_uuid_or_index_and_wrong_smi_count_rejected(self):
        duplicate_uuid=smi_rows();duplicate_uuid[-1]['uuid']=duplicate_uuid[0]['uuid']
        duplicate_index=smi_rows();duplicate_index[-1]['index']=0
        for rows in [duplicate_uuid,duplicate_index,smi_rows()[:-1],smi_rows()+[smi_rows()[0]]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                g.parse_smi(render(rows))

    def test_visible_count_and_mask_mismatches_rejected(self):
        for count in [0,7,9]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                g.inspect_devices(smi_rows(),fake_torch(count=count),MATCHER,None)
        for mask in ['', '0', 'all', '1,0,2,3,4,5,6,7', '0,1,2,3,4,5,6', uuid(0)]:
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError,'mask'):
                g.inspect_devices(smi_rows(),fake_torch(),MATCHER,mask)

    def test_uuid_mapping_mismatch_rejected_and_missing_property_is_explicit(self):
        torch=fake_torch();torch.properties[0].uuid=uuid(1)
        with self.assertRaisesRegex(ValueError,'UUID/index mapping'):
            g.inspect_devices(smi_rows(),torch,MATCHER,None)
        torch=fake_torch()
        for prop in torch.properties: del prop.uuid
        devices=g.inspect_devices(smi_rows(),torch,MATCHER,None)
        self.assertTrue(all(d['uuid_binding']=='nvidia_smi_index_complete_unmasked_inventory' for d in devices))

    def test_changed_source_and_incorrect_sha_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td).resolve()/'preflight.py';path.write_bytes(SOURCE_RAW+b'\n')
            with self.assertRaisesRegex(ValueError,'source bytes changed'):
                g.source_bytes(path,g.PREFLIGHT_SHA256)
        with self.assertRaisesRegex(ValueError,'Unexpected frozen preflight SHA'):
            g.source_bytes(SOURCE,'0'*64)

    def test_symlink_source_or_parent_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).resolve();source=base/'preflight.py';source.write_bytes(SOURCE_RAW)
            link=base/'link.py';link.symlink_to(source)
            parent=base/'alias';parent.symlink_to(base,target_is_directory=True)
            for path in [link,parent/'preflight.py']:
                with self.assertRaisesRegex(ValueError,'Unsafe preflight source path'):
                    g.source_bytes(path,g.PREFLIGHT_SHA256)

    def test_concurrent_source_change_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td).resolve()/'preflight.py';path.write_bytes(SOURCE_RAW)
            original=os.fstat;calls=[]
            def change_on_second_fstat(fd):
                calls.append(fd)
                if len(calls)==2:path.write_bytes(SOURCE_RAW+b'\n')
                return original(fd)
            with patch.object(g.os,'fstat',side_effect=change_on_second_fstat):
                with self.assertRaisesRegex(ValueError,'source changed during read'):
                    g.source_bytes(path,g.PREFLIGHT_SHA256)

    def test_no_cpu_import_of_torch_and_invalid_source_before_gpu_calls(self):
        real_import=builtins.__import__
        def forbid_torch(name,*args,**kwargs):
            if name=='torch' or name.startswith('torch.'):raise AssertionError('real torch import')
            return real_import(name,*args,**kwargs)
        with patch.object(builtins,'__import__',side_effect=forbid_torch):
            module=load_helper('cpu_safe_import')
            module.load_matcher(SOURCE_RAW,SOURCE)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
                module.main(['--help'])
            self.assertEqual(raised.exception.code,0)
            with patch.object(module,'smi',side_effect=AssertionError('GPU query')):
                with self.assertRaisesRegex(ValueError,'Unexpected frozen preflight SHA'):
                    module.main(['--preflight-source',str(SOURCE),'--preflight-sha256','0'*64])

    def test_cli_success_contract_and_changing_inventory_rejected(self):
        args=['--preflight-source',str(SOURCE),'--preflight-sha256',g.PREFLIGHT_SHA256]
        with patch.dict(sys.modules,{'torch':fake_torch()}), patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':g.FULL_MASK}):
            with patch.object(g,'smi',return_value=smi_rows()), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(g.main(args),0)
            result=json.loads(output.getvalue())
            self.assertEqual(result['status'],'GPU_IDENTITY_ACCEPTED')
            self.assertEqual(result['visible_device_count'],8)
            self.assertTrue(result['requires_original_all_rank_preflight'])
            self.assertFalse(result['bf16_nccl_preflight_passed'])
            changed=smi_rows();changed[0]['uuid']=uuid(9)
            with patch.object(g,'smi',side_effect=[smi_rows(),changed]), self.assertRaisesRegex(ValueError,'allocation changed'):
                g.main(args)


if __name__=='__main__':
    unittest.main()
