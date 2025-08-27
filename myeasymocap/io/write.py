import os
from easymocap.mytools.file_utils import write_keypoints3d, write_smpl, write_vertices
from easymocap.annotator.file_utils import save_annot
from os.path import join
from tqdm import tqdm

class WriteSMPL:
    def __init__(self, name='smpl', write_vertices=False) -> None:
        self.name = name
        # TODO: make available
        self.write_vertices = write_vertices
    
    def __call__(self, params=None, results=None, meta=None, model=None):
        results_all = []
        if results is None and params is not None:
            # copy params to results
            results = {0: {'params': params, 'keypoints3d': None, 'frames': list(range(len(params['Rh'])))}}
        for index in tqdm(meta['index'], desc=self.name):
            results_frame = []
            for pid, result in results.items():
                if index >= result['frames'][0] and index <= result['frames'][-1]:
                    frame_rel = result['frames'].index(index)
                    results_frame.append({
                        'id': pid,
                        # 'keypoints3d': result['keypoints3d'][frame_rel]
                    })
                    for key in ['Rh', 'Th', 'poses', 'shapes']:
                        if result['params'][key].shape[0] == 1:
                            results_frame[-1][key] = result['params'][key]
                        else:
                            results_frame[-1][key] = result['params'][key][frame_rel:frame_rel+1]
                    param = results_frame[-1]
                    pred = model(param)['keypoints'][0]
                    results_frame[-1]['keypoints3d'] = pred
                    if self.write_vertices:
                        vert = model(param, ret_vertices=True)['keypoints'][0]
                        results_frame[-1]['vertices'] = vert
            write_smpl(join(self.output, self.name, '{:06d}.json'.format(meta['frame'][index])), results_frame)
            write_keypoints3d(join(self.output, 'keypoints3d', '{:06d}.json'.format(meta['frame'][index])), results_frame)
            if self.write_vertices:
                write_vertices(join(self.output, 'vertices', '{:06d}.json'.format(meta['frame'][index])), results_frame)
                for res in results_frame:
                    res.pop('vertices')
            results_all.append(results_frame)
        return {'results_perframe': results_all}