"""FTU-selected SFDA radar reader: complex ROI -> fixed offline views -> BPM.

The input is the device-specific frontend's complex [native_frames,cells] ROI.
No HR references or target labels are accepted. Source and adapted weights use
identical forward processing; the caller supplies an explicitly bound weight.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from src.models.dar_waveform_reader import Reader
from src.features.dar_transfer_views import views
from src.features.dar_transfer_readout import causal_readout


class FrozenDarPipeline:
    def __init__(self,checkpoint,expected_sha256):
        self.path=Path(checkpoint);self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.sha256!=expected_sha256:raise ValueError('Frozen checkpoint SHA256 mismatch')
        cp=torch.load(self.path,map_location='cpu',weights_only=True)
        if 'model_state' not in cp:raise ValueError('Checkpoint lacks frozen model_state')
        self.model=Reader();self.model.load_state_dict(cp['model_state'],strict=True);self.model.eval()
        self.checkpoint_metadata={k:cp[k] for k in ('family','arm','init','source_control','seed','target_cells','source_checkpoint_sha256') if k in cp}
        self.zero_input=cp.get('source_control',cp.get('arm'))=='no_radar'

    def predict(self,complex_stream,sample_rate_hz,window_starts,record_id):
        if not isinstance(record_id,str) or not record_id:raise ValueError('One explicit physical record ID is required')
        # views validates native time geometry, complex/finiteness and complete support.
        starts=np.asarray(window_starts)
        if starts.ndim!=1 or not np.issubdtype(starts.dtype,np.integer):raise ValueError('Original native window starts must be a one-dimensional integer array')
        feature=views(complex_stream,sample_rate_hz,starts)['wave10'];x=torch.from_numpy(feature)
        if self.zero_input:x=torch.zeros_like(x)
        with torch.no_grad():raw=torch.cat([self.model(xx) for xx in x.split(64)]).numpy().astype(np.float64)
        times=starts.astype(float)/float(sample_rate_hz);smooth=causal_readout(raw,times,0.,'ewma025')
        if not np.isfinite(raw).all():raise ValueError('Frozen reader produced nonfinite BPM')
        return dict(record_id=record_id,window_id=np.arange(len(starts),dtype=np.int64),start_frame=starts.copy(),start_seconds=times,raw_bpm=raw,ewma025_bpm=smooth,checkpoint_sha256=self.sha256,zero_radar_control=self.zero_input)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True,help='NPZ: complex_stream [T,C], integer window_starts, scalar sample_rate_hz');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--record-id',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(1)
    if args.output.exists() or args.output.with_suffix('.json').exists():raise FileExistsError('Choose a new output; frozen artifacts are not overwritten')
    with np.load(args.input,allow_pickle=False) as data:
        if set(data.files)!={'complex_stream','window_starts','sample_rate_hz'}:raise ValueError('Expected only complex_stream, window_starts, sample_rate_hz')
        pipeline=FrozenDarPipeline(args.checkpoint,args.checkpoint_sha256);result=pipeline.predict(data['complex_stream'],float(data['sample_rate_hz']),data['window_starts'],args.record_id)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as file:np.savez(file,**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
    with args.output.with_suffix('.json').open('x') as file:json.dump(dict(record_id=args.record_id,checkpoint=str(args.checkpoint),checkpoint_sha256=result['checkpoint_sha256'],checkpoint_metadata=pipeline.checkpoint_metadata,input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),predictions_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),source_candidate='sfda/no_temporal/ewma025',zero_radar_control=result['zero_radar_control'],offline_whole_record_filtering=True,reference_files_read=False),file,indent=2,allow_nan=False);file.write('\n')
if __name__=='__main__':main()
