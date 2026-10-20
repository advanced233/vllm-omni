"""Post-process complete quality results without changing user evaluation outputs."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
NAMES=('default','fp32_state')

def avg(values):
    return statistics.fmean(values) if values else None

def group_summary(rows):
    return {'n':len(rows),'wer_mean':avg([r['wer'] for r in rows]),'sim_mean':avg([r['sim'] for r in rows]),'wer_over_0_5':sum(r['wer']>0.5 for r in rows),'wer_zero':sum(r['wer']==0 for r in rows),'asr_empty':sum(bool(r.get('asr_empty')) for r in rows),'at_frame_cap':sum(r['frames']>=500 for r in rows)}

def main():
    q={name:json.loads((ROOT/name/'quality.json').read_text()) for name in NAMES}
    req={name:[json.loads(l) for l in (ROOT/name/'requests.jsonl').read_text().splitlines()] for name in NAMES}
    analysis={name:json.loads((ROOT/name/'audio_analysis.json').read_text()) for name in NAMES}
    for name in NAMES:
        totals=q[name]['totals']; assert totals['utterances']==totals['scored']==totals['sim_scored']==1088 and not totals['errors']
        assert len(q[name]['items'])==len(req[name])==len(analysis[name])==1088
        assert [x['utt_id'] for x in q[name]['items']]==[x['utt_id'] for x in req[name]]
        assert all(i['stuck']==('noise-run' in a['flags']) for i,a in zip(q[name]['items'],analysis[name]))
    assert [i['utt_id'] for i in q['default']['items']]==[i['utt_id'] for i in q['fp32_state']['items']]
    pairs=[]
    for index,(a,b) in enumerate(zip(q['default']['items'],q['fp32_state']['items'])):
        pair={'index':index,'utt_id':a['utt_id'],'target_text':a['target_text'],'default':a,'fp32_state':b,'delta_wer_fp32_minus_default':round(b['wer']-a['wer'],4),'delta_sim_fp32_minus_default':round(b['sim']-a['sim'],4)}
        pairs.append(pair)
    groups={}
    for name in NAMES:
        rows=q[name]['items']
        groups[name]={'all':group_summary(rows),'noise_run':group_summary([r for r in rows if r['stuck']]),'without_noise_run':group_summary([r for r in rows if not r['stuck']]),'at_frame_cap':group_summary([r for r in rows if r['frames']>=500]),'not_at_frame_cap':group_summary([r for r in rows if r['frames']<500])}
    paired={}
    for metric,direction in [('wer',-1),('sim',1)]:
        ds=[p[f'delta_{metric}_fp32_minus_default'] for p in pairs]
        paired[metric]={'delta_direction':'fp32 minus default','mean_delta':avg(ds),'median_delta':statistics.median(ds),'fp32_better':sum(d*direction>0 for d in ds),'default_better':sum(d*direction<0 for d in ds),'equal_at_recorded_precision':sum(d==0 for d in ds)}
    noise_pairs=[p for p in pairs if p['default']['stuck'] or p['fp32_state']['stuck']]
    report={'rows':1088,'all_quality_complete':True,'pairing':'same row index, utt_id, target, reference, seed; unchanged online generator can vary with batch timing','groups':groups,'paired':paired,'noise_run_union_rows':len(noise_pairs),'interpretation':'Single run per dtype. Noise-run uses the supplied heuristic and is not per-sample human confirmation. WER includes empty ASR as 1 if present; SIM is the repository WavLM base-plus proxy. No formal B1 performance measurement.'}
    (ROOT/'quality_pair_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'paired_quality.json').write_text(json.dumps(pairs,indent=2,ensure_ascii=False)+'\n')
    (ROOT/'noise_quality_association.json').write_text(json.dumps(noise_pairs,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
