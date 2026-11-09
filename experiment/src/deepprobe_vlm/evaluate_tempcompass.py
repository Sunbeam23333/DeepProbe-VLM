"""Strict TempCompass wrapper around the unchanged, pinned reference engine."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys


def main(argv=None):
    ops=Path(__file__).resolve().parents[2]/'scripts/operations'
    sys.path.insert(0,str(ops))
    import run_tempcompass_eval as runner
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--model',required=True);p.add_argument('--provenance',type=Path,required=True)
    a=p.parse_args(argv)
    runner.require(runner.sha(a.plan)==a.plan_sha256,'plan_sha_changed')
    plan=runner.validate_plan(runner.read(a.plan));source_sha=runner.verify_sources(plan)
    runner.require(a.model in plan['model_sequence'],'model_not_declared')
    binding=plan['model_bindings'][a.model]
    runner.require(os.environ.get('TEMP_COMPASS_ADMITTED_PLAN_SHA256')==a.plan_sha256,'parent_gpu_admission_required')
    data=runner.verify_accepted_data(runner.pinned(plan['accepted_data'],runner.PACKAGE),plan['accepted_data']['sha256'])
    provenance=runner.read(a.provenance)
    expected={'model':a.model,'plan_sha256':a.plan_sha256,'protocol':runner.PROTOCOL,'reference_engine_protocol':'reference',
              'manifest_sha256':runner.PINS['manifest'],'source_files_sha256':source_sha,
              'accepted_data_sha256':plan['accepted_data']['sha256'],'all_checkpoint_files_sha256_verified':True,
              'model_binding_sha256':runner.canonical_sha(binding),'training_seed':binding['seed'],
              'training_acceptance_sha256':binding['training_acceptance']['sha256'],
              'checkpoint':binding['checkpoint']['path'],'run_id':binding['checkpoint']['run_id'],
              'matched_sft_training_source_verified':True}
    runner.require(all(provenance.get(k)==v for k,v in expected.items()),'runner_model_provenance_changed')
    config=runner.read(Path(provenance['checkpoint'])/'deepprobe_config.json')
    runner.require(config==provenance['config'] and runner.sha(Path(provenance['checkpoint'])/'deepprobe_config.json')==provenance['checkpoint_config_sha256'],
                   'checkpoint_config_changed_after_preflight')
    runner.validate_model_config(config,binding)
    runner.check_training_sources(provenance['checkpoint'],data['records'],data['training_sources'])
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    runner.require(world==8 and 0<=rank<8,'one_node_eight_ranks_required')
    model_root=Path(plan['run_root'])/a.model
    runner.require(a.provenance==model_root/'MODEL_PROVENANCE.json','model_provenance_path_changed')
    path=runner.shard_file(model_root,rank)
    runner.json_once(path.with_suffix('.tempcompass.json'),{'schema_version':1,'protocol':runner.PROTOCOL,
        'reference_engine_protocol':'reference','reference_metrics_are_diagnostic_only':True,'rank':rank,'world_size':world,
        'plan_sha256':a.plan_sha256,'run_provenance_sha256':runner.sha(a.provenance),'source_files_sha256':source_sha,
        'manifest_sha256':runner.PINS['manifest'],'accepted_data_sha256':plan['accepted_data']['sha256'],
        'no_gold_prompt':True,'max_new_tokens':16,'frames':8,'pool_grid':4,'audio':False,'subtitles':False})
    # This import and call retain the proven model loader, frame/collator path,
    # greedy generator, shard ownership and raw-output format without changing
    # any original engine file or advertising its reference scores as official.
    from . import evaluate
    return evaluate.main(['--checkpoint',provenance['checkpoint'],'--manifest',data['manifest_path'],
        '--media-root',plan['media_root'],'--output',str(model_root/'raw/predictions.jsonl'),
        '--max-new-tokens','16','--protocol','reference','--distributed-env','--run-provenance',str(a.provenance)])


if __name__=='__main__':raise SystemExit(main())
