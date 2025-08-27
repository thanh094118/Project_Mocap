from tqdm import tqdm
import cv2
import os
from easymocap.visualize.pyrender_wrapper import plot_meshes
from os.path import join
import numpy as np
from easymocap.datasets.base import add_logo
from easymocap.mytools.vis_base import merge, plot_bbox
from easymocap.mytools.camera_utils import Undistort
from .vis import VisBase
import pickle

class Render(VisBase):
    def __init__(self, name='render', scale=0.5, backend='pyrender', **kwargs) -> None:
        super().__init__(name=name, scale=1., **kwargs)
        self.scale3d = scale

    def __call__(self, body_model, params, cameras, imgnames):
        vertices = body_model.vertices(params, return_tensor=False)
        faces = body_model.faces
        for nf, img in enumerate(tqdm(imgnames, desc=self.name)):
            basename = os.path.basename(img)
            assert os.path.exists(img), img
            vis = cv2.imread(img)
            vis = cv2.resize(vis, None, fx=self.scale3d, fy=self.scale3d)
            vert = vertices[nf]
            meshes = {}
            meshes[0] = {
                'vertices': vert,
                'faces': faces,
                'id': 0,
                'name': 'human_{}'.format(0)
            }
            K = cameras['K'][nf].copy()
            K[:2, :] *= self.scale3d
            R = cameras['R'][nf]
            T = cameras['T'][nf]
            ret = plot_meshes(vis, meshes, K, R, T, mode='image')
            self.merge_and_write([ret])

class Render_hand(VisBase):
    def __init__(self, name='render', scale=0.5, backend='pyrender',
                 part_segmentation_path='models/pare/data/smpl_partSegmentation_mapping.pkl',
                 **kwargs) -> None:
        super().__init__(name=name, scale=1., **kwargs)
        self.scale3d = scale

        # Tải file ánh xạ phân vùng (part segmentation) của mô hình SMPL
        self.part_mapping = self._load_part_segmentation(part_segmentation_path)
        # Danh sách ID của các bộ phận cần loại bỏ:
        # 20: L_Hand, 21: R_Hand, 22: Jaw, 23: L_Eye
        self.remove_ids = [20, 21, 22, 23]

        # Lấy toàn bộ chỉ số (indices) đỉnh (vertex) tương ứng với các bộ phận cần loại bỏ
        self.remove_indices = self._get_multiple_part_indices(self.remove_ids)

    def _load_part_segmentation(self, path):
        """Tải dữ liệu phân vùng bộ phận từ file .pkl"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Loaded part segmentation from: {path}")
            print(f"✓ Data keys: {list(data.keys())}")

            # Nếu dữ liệu chứa key 'smpl_index' -> đây là mảng phân vùng cho toàn bộ mesh
            if 'smpl_index' in data:
                part_mapping = data['smpl_index']
                print(f"✓ Using 'smpl_index' as segmentation array")
                print(f"✓ Array length: {len(part_mapping)}")  # số lượng đỉnh trong mô hình
                print(f"✓ Unique part IDs: {np.unique(part_mapping)}")  # các mã bộ phận duy nhất
                return part_mapping
            else:
                # Nếu không có key cần tìm -> trả về None
                print(f"✗ 'smpl_index' key not found in data")
                return None
        except Exception as e:
            # Báo lỗi khi load file
            print(f"✗ Error loading part segmentation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_multiple_part_indices(self, part_ids):
        """Lấy tất cả index của các đỉnh thuộc các bộ phận trong part_ids"""
        if self.part_mapping is None:
            print("✗ Part mapping is None")
            return np.array([])

        if isinstance(self.part_mapping, (list, np.ndarray)):
            indices = []
            for pid in part_ids:
                # Lấy các chỉ số đỉnh mà giá trị part_mapping == pid
                idx = np.where(np.array(self.part_mapping) == pid)[0]
                print(f"✓ Found part ID {pid} with {len(idx)} vertices")
                indices.extend(idx.tolist())
            return np.array(indices)
        else:
            print(f"✗ Unexpected part_mapping type: {type(self.part_mapping)}")
            return np.array([])

    def _remove_parts_mesh(self, vertices, faces):
        """Loại bỏ toàn bộ đỉnh và mặt (face) thuộc các bộ phận cần xóa"""
        # Tập hợp các index đỉnh cần xóa
        remove_indices = set(self.remove_indices.tolist())

        # Mặt nạ (mask) giữ lại các đỉnh không thuộc remove_indices
        keep_mask = np.array([i not in remove_indices for i in range(len(vertices))])

        # Ánh xạ từ chỉ số cũ sang chỉ số mới sau khi xóa
        old_to_new = {}
        new_idx = 0
        for old_idx, keep in enumerate(keep_mask):
            if keep:
                old_to_new[old_idx] = new_idx
                new_idx += 1

        # Lọc lại danh sách đỉnh
        new_vertices = vertices[keep_mask]

        # Lọc lại danh sách mặt (face)
        keep_faces = []
        for f in faces:
            # Giữ lại các mặt mà tất cả đỉnh của nó đều nằm trong danh sách giữ lại
            if all(v in old_to_new for v in f):
                keep_faces.append([old_to_new[v] for v in f])
        new_faces = np.array(keep_faces, dtype=np.int32)

        print(f"Removed {len(remove_indices)} vertices, {len(faces) - len(new_faces)} faces removed.")

        return new_vertices, new_faces

    def __call__(self, body_model, params, cameras, imgnames):
        """
        Thực hiện render mesh cho từng ảnh trong imgnames.
        """
        # Lấy danh sách đỉnh từ body_model (không ở dạng tensor)
        vertices = body_model.vertices(params, return_tensor=False)
        faces = body_model.faces

        # Lặp qua từng ảnh đầu vào
        for nf, img in enumerate(tqdm(imgnames, desc=self.name)):
            basename = os.path.basename(img)
            assert os.path.exists(img), img  # đảm bảo file ảnh tồn tại

            # Đọc ảnh và resize theo scale3d
            vis = cv2.imread(img)
            vis = cv2.resize(vis, None, fx=self.scale3d, fy=self.scale3d)

            # Lấy đỉnh tương ứng với frame hiện tại
            vert = vertices[nf]

            # Loại bỏ các bộ phận đã chọn
            filtered_vert, filtered_faces = self._remove_parts_mesh(vert, faces)

            # Chuẩn bị dữ liệu mesh để render
            meshes = {
                0: {
                    'vertices': filtered_vert,
                    'faces': filtered_faces,
                    'id': 0,
                    'name': f'human_{0}'
                }
            }

            # Lấy thông số camera và scale K (ma trận nội tại) theo scale3d
            K = cameras['K'][nf].copy()
            K[:2, :] *= self.scale3d
            R = cameras['R'][nf]  # ma trận xoay
            T = cameras['T'][nf]  # vector tịnh tiến

            # Gọi hàm render mesh lên ảnh
            ret = plot_meshes(vis, meshes, K, R, T, mode='image')

            # Gộp kết quả và lưu
            self.merge_and_write([ret])
