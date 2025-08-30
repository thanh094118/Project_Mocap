import numpy as np
import json
import os
from glob import glob

# ===== Umeyama Alignment =====
def umeyama_alignment(X, Y, with_scale=True):
    n, m = X.shape
    mean_X = X.mean(axis=0)
    mean_Y = Y.mean(axis=0)
    Xc = X - mean_X
    Yc = Y - mean_Y
    C = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    if with_scale:
        var_X = (Xc**2).sum() / n
        s = np.sum(D) / var_X
    else:
        s = 1.0
    t = mean_Y - s * R @ mean_X
    return s, R, t

def transform(P, s, R, t):
    return (s * (R @ P.T).T) + t

def compute_errors(P1_aligned, P2):
    return np.linalg.norm(P1_aligned - P2, axis=1)

def get_threshold_from_user():
    while True:
        try:
            threshold_m = float(input("Nhập ngưỡng sai lệch (đơn vị m): "))
            if threshold_m > 0:
                return threshold_m
            else:
                print("Ngưỡng phải lớn hơn 0. Vui lòng nhập lại.")
        except ValueError:
            print("Vui lòng nhập một số hợp lệ.")

def get_window_size_from_user():
    while True:
        try:
            window_size = int(input("Nhập kích thước window (số frame liên tục để phân tích): "))
            if window_size > 0:
                return window_size
            else:
                print("Window size phải lớn hơn 0. Vui lòng nhập lại.")
        except ValueError:
            print("Vui lòng nhập một số nguyên hợp lệ.")

def get_frame_offset_from_user():
    while True:
        try:
            offset = int(input("Nhập giá trị frame lệch: "))
            if offset >= 0:
                break
            else:
                print("Frame lệch phải >= 0. Vui lòng nhập lại.")
        except ValueError:
            print("Vui lòng nhập một số nguyên hợp lệ.")
    
    while True:
        try:
            direction = int(input("Nhập '1' nếu folder1 lệch lên so với folder2, '2' nếu folder2 lệch lên so với folder1: "))
            if direction in [1, 2]:
                return offset, direction
            else:
                print("Vui lòng nhập 1 hoặc 2.")
        except ValueError:
            print("Vui lòng nhập một số nguyên hợp lệ.")

def analyze_continuous_errors(error_log, threshold, keypoint_names, window_size):
    """Phân tích lỗi trên nhiều frame liên tục - so sánh 19 keypoints (0-18)"""
    num_frames, num_points = error_log.shape
    results = []
    
    for start_frame in range(num_frames - window_size + 1):
        # Lấy window của các frame liên tục
        window = error_log[start_frame:start_frame + window_size]
        mean_errors = window.mean(axis=0)  # Lỗi trung bình của từng điểm trong window
        
        # Tìm các điểm có lỗi vượt ngưỡng (xét 19 điểm từ 0-18)
        bad_points = np.where(mean_errors > threshold)[0]
        
        if len(bad_points) > 0:
            frame_range = f"{start_frame+1:06d}-{start_frame+window_size:06d}"
            bad_keypoints = []
            error_values = []
            
            for pid in bad_points:
                name = keypoint_names.get(pid, f"kp{pid}")
                bad_keypoints.append(f"{pid}:{name}")
                # Giữ nguyên đơn vị m
                error_values.append(f"{mean_errors[pid]:.6f}m")
            
            results.append({
                'frame_range': frame_range,
                'bad_keypoints': ', '.join(bad_keypoints),
                'error_values': ', '.join(error_values)
            })
    
    return results

# ===== Main Processing =====
def process_videos(folder1, folder2, outfolder, ids=[0,1,8,15,16,9,12]):
    files1 = sorted(glob(os.path.join(folder1, "*.json")))
    files2 = sorted(glob(os.path.join(folder2, "*.json")))
    
    if len(files1) == 0:
        print("Không tìm thấy file nào trong folder1!")
        return
    if len(files2) == 0:
        print("Không tìm thấy file nào trong folder2!")
        return
    
    # Hỏi frame offset
    offset, direction = get_frame_offset_from_user()
    
    # Xử lý offset
    if direction == 1:  # folder1 lệch lên so với folder2
        files1_processed = files1[offset:]
        files2_processed = files2
        print(f"Folder1 lệch lên {offset} frame: bỏ qua {offset} frame đầu của folder1")
    else:  # direction == 2, folder2 lệch lên so với folder1
        files1_processed = files1
        files2_processed = files2[offset:]
        print(f"Folder2 lệch lên {offset} frame: bỏ qua {offset} frame đầu của folder2")
    
    # Lấy số frame nhỏ hơn để xử lý
    min_frames = min(len(files1_processed), len(files2_processed))
    print(f"Sẽ xử lý {min_frames} frame sau khi điều chỉnh offset")
    
    if min_frames == 0:
        print("Không có frame nào để xử lý sau khi điều chỉnh offset!")
        return

    # Tạo thư mục output
    out1 = os.path.join(outfolder, "keypoints3d_1")
    out2 = os.path.join(outfolder, "keypoints3d_2")
    os.makedirs(out1, exist_ok=True)
    os.makedirs(out2, exist_ok=True)

    # tên keypoints - chỉ 19 điểm (0-18)
    keypoint_names = {
        0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
        5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
        10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
        15: "REye", 16: "LEye", 17: "REar", 18: "LEar"
    }

    error_log = []

    # Xử lý các frame sau khi điều chỉnh offset
    for i in range(min_frames):
        f1 = files1_processed[i]
        f2 = files2_processed[i]
        
        frame_id1 = os.path.basename(f1).replace(".json","")
        frame_id2 = os.path.basename(f2).replace(".json","")
        
        print(f"Đang xử lý: {frame_id1} (folder1) vs {frame_id2} (folder2)")

        with open(f1,"r") as f:
            data1 = json.load(f)
        with open(f2,"r") as f:
            data2 = json.load(f)

        keypoints1 = np.array(data1[0]["keypoints3d"])
        keypoints2 = np.array(data2[0]["keypoints3d"])

        # chọn mốc
        X = keypoints1[ids]
        Y = keypoints2[ids]

        # align
        s, R, t = umeyama_alignment(X, Y, with_scale=True)
        aligned1 = transform(keypoints1, s, R, t)
        aligned2 = keypoints2

        # làm tròn tọa độ đến 7 chữ số thập phân
        aligned1_rounded = np.round(aligned1, 7)
        aligned2_rounded = np.round(aligned2, 7)
        
        # format đúng cấu trúc
        data1_output = [{"id": 0, "keypoints3d": aligned1_rounded.tolist()}]
        data2_output = [{"id": 0, "keypoints3d": aligned2_rounded.tolist()}]
        
        # lưu lại (tên file theo frame gốc)
        with open(os.path.join(out1, frame_id1+".json"), "w") as f:
            json.dump(data1_output, f, indent=2)
        with open(os.path.join(out2, frame_id2+".json"), "w") as f:
            json.dump(data2_output, f, indent=2)

        # tính sai số (đơn vị m) - chỉ so sánh 19 điểm đầu (0-18)
        errors = compute_errors(aligned1[:19], aligned2[:19])
        error_log.append(errors)

    error_log = np.array(error_log)

    # Hỏi người dùng ngưỡng và window size
    threshold = get_threshold_from_user()
    window_size = get_window_size_from_user()
    
    # Phân tích lỗi trên nhiều frame liên tục
    results = analyze_continuous_errors(error_log, threshold, keypoint_names, window_size)
    
    # In bảng kết quả
    if results:
        print(f"\n{'='*90}")
        print(f"{'BẢNG PHÂN TÍCH ĐIỂM SAI':^90}")
        print(f"{'='*90}")
        print(f"{'Khoảng Frame':<15} {'Điểm Sai':<45} {'Khoảng Cách Sai':<30}")
        print(f"{'-'*90}")
        
        for result in results:
            print(f"{result['frame_range']:<15} {result['bad_keypoints']:<45} {result['error_values']:<30}")
        
        print(f"{'-'*90}")
        print(f"Offset: {offset} (direction: {direction}) | Ngưỡng: {threshold}m | Window: {window_size} frames | Tổng: {len(results)} khoảng frame có điểm sai")
    else:
        print(f"\nKhông tìm thấy điểm sai nào với ngưỡng {threshold}m và window {window_size} frames")

# ===== Run =====
if __name__ == "__main__":
    folder1 = "output1/sv1p/keypoints3d"
    folder2 = "output2/sv1p/keypoints3d"
    outfolder = "aligned"
    process_videos(folder1, folder2, outfolder)