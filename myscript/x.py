import numpy as np
import json
import os
import glob
from scipy.spatial.distance import euclidean
from typing import Tuple, List, Dict
from collections import defaultdict

class VideoKeypointsAlignment:
    def __init__(self):
        """
        Khởi tạo class để căn chỉnh keypoints 3D từ 2 video (nhiều frames)
        Sử dụng Procrustes Analysis với các điểm mốc quan trọng
        """
        # Các chỉ số điểm quan trọng có độ tin cậy cao
        self.reference_points = [0, 1, 15, 16, 9, 12]  # Nose, Neck, REye, LEye, RHip, LHip
        self.midhip_index = 8  # MidHip làm gốc tọa độ
        self.precision = 7  # Làm tròn 7 chữ số sau dấu phẩy
        self.error_threshold = 100  # Ngưỡng đánh giá điểm sai (mm) - đã thay đổi từ 60 thành 100
        
        # Các điểm cần bỏ qua trong báo cáo (19-24: toe và heel points)
        self.ignore_points = {19, 20, 21, 22, 23, 24}
        
        # Tên các keypoints
        self.keypoint_names = [
            "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
            "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
            "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
            "REye", "LEye", "REar", "LEar", "LBigToe",
            "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
        ]
        
    def load_keypoints(self, file_path: str) -> np.ndarray:
        """
        Đọc keypoints từ file JSON
        
        Args:
            file_path: Đường dẫn tới file JSON
            
        Returns:
            numpy array shape (25, 3) chứa tọa độ 25 điểm keypoints
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            keypoints = np.array(data[0]['keypoints3d'])
            return keypoints
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path}: {e}")
            return None
    
    def get_frame_files(self, folder_path: str) -> List[str]:
        """
        Lấy danh sách tất cả file frames trong thư mục
        
        Args:
            folder_path: Đường dẫn thư mục chứa các file keypoints
            
        Returns:
            Danh sách đường dẫn file đã sắp xếp
        """
        pattern = os.path.join(folder_path, "*.json")
        files = glob.glob(pattern)
        files.sort()  # Sắp xếp theo thứ tự
        return files
    
    def translate_to_midhip_origin(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Dịch chuyển tọa độ để MidHip (điểm 8) làm gốc O=[0,0,0]
        
        Args:
            keypoints: Mảng keypoints shape (25, 3)
            
        Returns:
            Keypoints sau khi dịch chuyển
        """
        # Lấy tọa độ MidHip
        midhip_coord = keypoints[self.midhip_index].copy()
        
        # Dịch chuyển tất cả điểm về gốc tại MidHip
        translated_keypoints = keypoints - midhip_coord
        
        # Làm tròn theo độ chính xác yêu cầu
        translated_keypoints = np.round(translated_keypoints, self.precision)
        
        return translated_keypoints
    
    def extract_reference_points(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Trích xuất các điểm mốc quan trọng
        
        Args:
            keypoints: Mảng keypoints shape (25, 3)
            
        Returns:
            Mảng các điểm mốc shape (6, 3)
        """
        return keypoints[self.reference_points]
    
    def compute_rigid_transform(self, source_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Tính toán rigid transformation (rotation + translation + scale) sử dụng Procrustes Analysis
        
        Args:
            source_points: Điểm nguồn shape (N, 3)
            target_points: Điểm đích shape (N, 3)
            
        Returns:
            R: Ma trận rotation (3, 3)
            t: Vector translation (3,)
            s: Scale factor
        """
        # Tính centroid
        centroid_source = np.mean(source_points, axis=0)
        centroid_target = np.mean(target_points, axis=0)
        
        # Trừ centroid
        source_centered = source_points - centroid_source
        target_centered = target_points - centroid_target
        
        # Tính scale factor
        scale_source = np.sqrt(np.sum(source_centered ** 2))
        scale_target = np.sqrt(np.sum(target_centered ** 2))
        
        if scale_source > 1e-10:
            source_normalized = source_centered / scale_source
        else:
            source_normalized = source_centered
            
        if scale_target > 1e-10:
            target_normalized = target_centered / scale_target
        else:
            target_normalized = target_centered
        
        # Tính ma trận rotation sử dụng SVD
        H = source_normalized.T @ target_normalized
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Đảm bảo ma trận rotation có determinant = 1
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Tính scale và translation
        scale = scale_target / scale_source if scale_source > 1e-10 else 1.0
        t = centroid_target - scale * (R @ centroid_source)
        
        return R, t, scale
    
    def apply_transform(self, keypoints: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
        """
        Áp dụng rigid transformation lên keypoints
        
        Args:
            keypoints: Keypoints cần transform shape (25, 3)
            R: Ma trận rotation (3, 3)
            t: Vector translation (3,)
            s: Scale factor
            
        Returns:
            Keypoints sau khi transform
        """
        # Áp dụng scale, rotation và translation
        transformed = s * (keypoints @ R.T) + t
        
        # Làm tròn theo độ chính xác yêu cầu
        transformed = np.round(transformed, self.precision)
        
        return transformed
    
    def compute_frame_errors(self, keypoints1: np.ndarray, keypoints2: np.ndarray) -> np.ndarray:
        """
        Tính toán độ lệch giữa hai bộ keypoints cho 1 frame
        
        Args:
            keypoints1: Keypoints góc nhìn 1 (reference)
            keypoints2: Keypoints góc nhìn 2 (sau khi transform)
            
        Returns:
            Array chứa khoảng cách lỗi cho từng điểm (đơn vị: mm)
        """
        distances = []
        for i in range(len(keypoints1)):
            dist = euclidean(keypoints1[i], keypoints2[i])
            distances.append(dist * 1000)  # Chuyển sang mm
            
        return np.array(distances)
    
    def analyze_error_windows(self, all_errors: List[np.ndarray]) -> Dict:
        """
        Phân tích sai lệch theo cửa sổ 3 frames liên tục
        
        Args:
            all_errors: List các mảng lỗi cho từng frame
            
        Returns:
            Dictionary chứa thông tin điểm sai
        """
        error_reports = []
        
        # Duyệt qua các cửa sổ 3 frames liên tục
        for start_frame in range(len(all_errors) - 2):
            end_frame = start_frame + 2
            frame_range = f"{start_frame+1:06d}-{end_frame+1:06d}"
            
            # Tính trung bình lỗi cho 3 frames
            window_errors = []
            for frame_idx in range(start_frame, end_frame + 1):
                window_errors.append(all_errors[frame_idx])
            
            mean_errors = np.mean(window_errors, axis=0)
            
            # Tìm điểm sai (> ngưỡng) và bỏ qua các điểm 19-24
            error_points = []
            for point_idx, error_mm in enumerate(mean_errors):
                # Bỏ qua các điểm toe và heel (19-24)
                if point_idx in self.ignore_points:
                    continue
                    
                if error_mm > self.error_threshold:
                    error_points.append({
                        'id': point_idx,
                        'name': self.keypoint_names[point_idx],
                        'error_mm': round(error_mm, 2)
                    })
            
            # Nếu có điểm sai, ghi nhận
            if error_points:
                error_reports.append({
                    'frame_range': frame_range,
                    'error_points': error_points
                })
        
        return error_reports
    
    def save_aligned_keypoints(self, aligned_keypoints1: List[np.ndarray], 
                             aligned_keypoints2: List[np.ndarray], 
                             output_dir: str, frame_files1: List[str]):
        """
        Lưu keypoints đã căn chỉnh cho tất cả frames
        
        Args:
            aligned_keypoints1: List keypoints góc nhìn 1 đã căn chỉnh
            aligned_keypoints2: List keypoints góc nhìn 2 đã căn chỉnh  
            output_dir: Thư mục output
            frame_files1: Danh sách tên file gốc
        """
        # Tạo thư mục
        output_dir1 = os.path.join(output_dir, "keypoints3d_1")
        output_dir2 = os.path.join(output_dir, "keypoints3d_2")
        os.makedirs(output_dir1, exist_ok=True)
        os.makedirs(output_dir2, exist_ok=True)
        
        # Lưu từng frame
        for i, (kp1, kp2, original_file) in enumerate(zip(aligned_keypoints1, aligned_keypoints2, frame_files1)):
            frame_name = os.path.basename(original_file)
            
            # Cấu trúc JSON
            data1 = [{"id": 0, "keypoints3d": kp1.tolist()}]
            data2 = [{"id": 0, "keypoints3d": kp2.tolist()}]
            
            # Lưu file
            with open(os.path.join(output_dir1, frame_name), 'w') as f:
                json.dump(data1, f, indent=2)
                
            with open(os.path.join(output_dir2, frame_name), 'w') as f:
                json.dump(data2, f, indent=2)
    
    def print_error_table(self, error_reports: List[Dict]):
        """
        In bảng điểm sai theo định dạng yêu cầu
        
        Args:
            error_reports: Danh sách báo cáo lỗi
        """
        print("\n" + "="*80)
        print("BẢNG ĐIỂM SAI (NGƯỠNG > 100mm)")
        print("="*80)
        print(f"{'Khoảng Frame':<20} {'Điểm Sai':<30} {'Khoảng Cách (mm)':<20}")
        print("-"*80)
        
        if not error_reports:
            print("KHÔNG CÓ ĐIỂM SAI NÀO > 100mm")
            print("Chất lượng căn chỉnh: TỐT")
        else:
            for report in error_reports:
                frame_range = report['frame_range']
                
                # In từng điểm sai trong khoảng frame này
                for i, point in enumerate(report['error_points']):
                    if i == 0:
                        # Dòng đầu tiên có frame range
                        print(f"{frame_range:<20} {point['id']:2d}-{point['name']:<25} {point['error_mm']:>15.2f}")
                    else:
                        # Các dòng sau để trống cột frame range
                        print(f"{'':<20} {point['id']:2d}-{point['name']:<25} {point['error_mm']:>15.2f}")
                
                # Thêm dòng trống giữa các khoảng frame
                if report != error_reports[-1]:
                    print()
        
        print("="*80)
    
    def process_videos(self, folder1: str, folder2: str, output_dir: str = "aligned_results"):
        """
        Xử lý toàn bộ video với nhiều frames
        
        Args:
            folder1: Thư mục chứa keypoints góc nhìn 1
            folder2: Thư mục chứa keypoints góc nhìn 2
            output_dir: Thư mục lưu kết quả
        """
        print("=== VIDEO KEYPOINTS ALIGNMENT - PROCRUSTES ANALYSIS ===\n")
        
        # Bước 1: Lấy danh sách files
        print("Bước 1: Quét files trong thư mục...")
        files1 = self.get_frame_files(folder1)
        files2 = self.get_frame_files(folder2)
        
        if len(files1) != len(files2):
            print(f"CẢNH BÁO: Số lượng frame không khớp!")
            print(f"- Góc nhìn 1: {len(files1)} frames")
            print(f"- Góc nhìn 2: {len(files2)} frames")
            min_frames = min(len(files1), len(files2))
            files1 = files1[:min_frames]
            files2 = files2[:min_frames]
            print(f"- Sử dụng {min_frames} frames đầu tiên\n")
        else:
            print(f"Tổng số frames: {len(files1)}\n")
        
        # Danh sách lưu kết quả
        aligned_keypoints1 = []
        aligned_keypoints2 = []
        all_errors = []
        
        print("Bước 2-5: Xử lý từng frame...")
        for i, (file1, file2) in enumerate(zip(files1, files2)):
            if i % 50 == 0:  # Hiển thị progress mỗi 50 frames
                print(f"Đang xử lý frame {i+1}/{len(files1)}...")
            
            # Đọc keypoints
            keypoints1 = self.load_keypoints(file1)
            keypoints2 = self.load_keypoints(file2)
            
            if keypoints1 is None or keypoints2 is None:
                print(f"Bỏ qua frame {i+1} do lỗi đọc dữ liệu")
                continue
            
            # Dịch chuyển về gốc tại MidHip cho cả 2 góc nhìn
            translated1 = self.translate_to_midhip_origin(keypoints1)
            translated2 = self.translate_to_midhip_origin(keypoints2)
            
            # Trích xuất điểm mốc
            ref_points1 = self.extract_reference_points(translated1)
            ref_points2 = self.extract_reference_points(translated2)
            
            # Tính toán rigid transformation
            R, t, scale = self.compute_rigid_transform(ref_points2, ref_points1)
            
            # Áp dụng transformation lên góc nhìn 2
            aligned_kp2 = self.apply_transform(translated2, R, t, scale)
            
            # Lưu kết quả
            aligned_keypoints1.append(translated1)
            aligned_keypoints2.append(aligned_kp2)
            
            # Tính lỗi cho frame này
            frame_errors = self.compute_frame_errors(translated1, aligned_kp2)
            all_errors.append(frame_errors)
        
        print(f"Hoàn thành xử lý {len(aligned_keypoints1)} frames\n")
        
        # Bước 6: Phân tích sai lệch theo cửa sổ 3 frames
        print("Bước 6: Phân tích sai lệch theo cửa sổ 3 frames liên tục...")
        error_reports = self.analyze_error_windows(all_errors)
        
        # Bước 7: Lưu kết quả
        print("Bước 7: Lưu kết quả đã căn chỉnh...")
        self.save_aligned_keypoints(aligned_keypoints1, aligned_keypoints2, output_dir, files1)
        print(f"Đã lưu {len(aligned_keypoints1)} frames vào thư mục: {output_dir}\n")
        
        # Hiển thị bảng điểm sai
        self.print_error_table(error_reports)
        
        # Thống kê tổng quan
        self.print_summary_stats(all_errors, error_reports)
        
        return {
            'aligned_keypoints1': aligned_keypoints1,
            'aligned_keypoints2': aligned_keypoints2,
            'error_reports': error_reports,
            'total_frames': len(aligned_keypoints1)
        }
    
    def print_summary_stats(self, all_errors: List[np.ndarray], error_reports: List[Dict]):
        """
        In thống kê tổng quan (chỉ tính toán cho các điểm không bị bỏ qua)
        
        Args:
            all_errors: Danh sách lỗi tất cả frames
            error_reports: Báo cáo điểm sai
        """
        print("\n" + "="*60)
        print("THỐNG KÊ TỔNG QUAN")
        print("="*60)
        
        # Tính RMSE trung bình trên tất cả frames (chỉ tính cho các điểm không bị bỏ qua)
        all_frame_rmse = []
        for frame_errors in all_errors:
            # Chỉ tính RMSE cho các điểm không bị bỏ qua
            valid_errors = []
            for i, error in enumerate(frame_errors):
                if i not in self.ignore_points:
                    valid_errors.append(error)
            
            if valid_errors:
                rmse = np.sqrt(np.mean(np.array(valid_errors) ** 2))
                all_frame_rmse.append(rmse)
        
        avg_rmse = np.mean(all_frame_rmse) if all_frame_rmse else 0
        
        print(f"Tổng số frames xử lý: {len(all_errors)}")
        print(f"RMSE trung bình (điểm 0-18): {avg_rmse:.2f} mm")
        print(f"Số khoảng frames có điểm sai (>100mm): {len(error_reports)}")
        
        # Thống kê điểm sai thường xuyên (chỉ cho các điểm không bị bỏ qua)
        point_error_count = defaultdict(int)
        for report in error_reports:
            for point in report['error_points']:
                point_error_count[f"{point['id']}-{point['name']}"] += 1
        
        if point_error_count:
            print(f"\nĐiểm sai thường xuyên nhất:")
            sorted_errors = sorted(point_error_count.items(), key=lambda x: x[1], reverse=True)
            for point_name, count in sorted_errors[:5]:
                print(f"- {point_name}: {count} lần")
        
        # Đánh giá chất lượng tổng thể
        if avg_rmse <= 30:
            quality = "TỐT"
        elif avg_rmse <= 60:
            quality = "KHẢ DỤNG"
        else:
            quality = "CẦN CẢI THIỆN"
            
        print(f"\nĐánh giá chất lượng tổng thể: {quality}")
        print(f"(Không tính điểm 19-24: LBigToe, LSmallToe, LHeel, RBigToe, RSmallToe, RHeel)")
        print("="*60)

def main():
    """
    Hàm main để chạy quá trình căn chỉnh video keypoints
    """
    # Khởi tạo class
    aligner = VideoKeypointsAlignment()
    
    # Đường dẫn thư mục input
    folder1 = "output1/sv1p/keypoints3d"  # Video góc nhìn 1 (reference)
    folder2 = "output2/sv1p/keypoints3d"  # Video góc nhìn 2
    
    # Thư mục output
    output_dir = "aligned_results"
    
    # Kiểm tra thư mục tồn tại
    if not os.path.exists(folder1):
        print(f"Không tìm thấy thư mục: {folder1}")
        return
        
    if not os.path.exists(folder2):
        print(f"Không tìm thấy thư mục: {folder2}")
        return
    
    # Thực hiện căn chỉnh video
    result = aligner.process_videos(folder1, folder2, output_dir)
    
    if result:
        print(f"\n✅ HOÀN THÀNH: Đã xử lý {result['total_frames']} frames")
        print(f"📁 Kết quả lưu tại: {output_dir}/")
        print(f"   - keypoints3d_1/ : {result['total_frames']} files (góc nhìn 1 đã căn chỉnh)")
        print(f"   - keypoints3d_2/ : {result['total_frames']} files (góc nhìn 2 đã căn chỉnh)")

if __name__ == "__main__":
    main()