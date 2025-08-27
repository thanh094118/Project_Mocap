#!/usr/bin/env python3
"""
check_smpl_pose_mismatch.py

So sánh "poses" (axis-angle) giữa hai thư mục smpl JSON theo frame.
Phát hiện khớp "nghi ngờ sai" dựa trên geodesic angle trên SO(3).

Output: CSV với (frame, joint_idx, joint_name, angle_deg, threshold_deg, flagged)
"""

import os, json, argparse, math, csv
import numpy as np

# --- Cấu hình tên khớp (mapping cho 23 pose joints) ---
# Giữ nhất quán với giả định: poses là 69 chiều (23*3), không chứa root (Rh/Th ở ngoài).
SMPL_JOINTS = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"
]
# Pose vector we have corresponds to indices 1..23 above (exclude 'pelvis' root)
POSE_INDEX_TO_SMPL = SMPL_JOINTS[1:]  # length 23

# Joints to ưu tiên (tay + chân) -> ngưỡng chặt hơn
TIGHT_JOINT_NAMES = {
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "left_hip","right_hip",
    "left_knee","right_knee",
    "left_ankle","right_ankle"
}

# --- Hàm tiện ích: axis-angle -> ma trận xoay (Rodrigues) ---
def axisangle_to_R(v):
    # v: (3,) axis-angle vector (rotvec). Nếu norm ~0 -> I
    theta = np.linalg.norm(v)
    if theta < 1e-8:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R

def pose69_to_Rlist(pose69):
    p = np.asarray(pose69).reshape(-1)
    if p.size != 69:
        raise ValueError(f"Expect pose vector length 69, got {p.size}")
    Rs = []
    for i in range(23):
        vec = p[i*3:(i+1)*3]
        Rs.append(axisangle_to_R(vec))
    return Rs

def geodesic_angle_between_R(R1, R2):
    Rrel = R1.T @ R2
    tr = np.trace(Rrel)
    c = (tr - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    theta = math.acos(c)
    return theta  # radians

# --- Đọc poses từ JSON (robust) ---
def load_pose_from_smpl_json(path):
    # Hỗ trợ nhiều biến thể: "poses": [[69]] hoặc "poses": [69]
    d = json.load(open(path, "r"))
    if isinstance(d, list) and len(d)>0 and isinstance(d[0], dict):
        d0 = d[0]
    elif isinstance(d, dict):
        d0 = d
    else:
        # fallback: if file contains dict with key "poses"
        d0 = d if isinstance(d, dict) else None

    if d0 is None:
        raise ValueError(f"Cannot parse JSON as smpl dict: {path}")

    if "poses" not in d0:
        raise ValueError(f"No 'poses' key in {path}")

    poses = d0["poses"]
    # poses could be list of lists, or one list
    if isinstance(poses, list) and len(poses)>0 and isinstance(poses[0], list):
        arr = np.array(poses[0], dtype=float)
    else:
        arr = np.array(poses, dtype=float)

    arr = arr.reshape(-1)
    if arr.size not in (69,):
        raise ValueError(f"Wrong poses length {arr.size} in {path}. Expect 69.")
    return arr

# --- Main compare function ---
def compare_dirs(dir1, dir2, out_csv,
                 tight_deg=30.0, loose_deg=45.0, min_consecutive=2):
    # List json files in each (only names)
    files1 = sorted([f for f in os.listdir(dir1) if f.lower().endswith(".json")],
                    key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
    files2 = sorted([f for f in os.listdir(dir2) if f.lower().endswith(".json")],
                    key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

    # Align by filename intersection
    common = sorted(list(set(files1).intersection(set(files2))),
                    key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
    if not common:
        raise RuntimeError("Không tìm thấy file JSON trùng tên trong hai thư mục.")

    T = len(common)
    angles_deg_all = np.zeros((T, 23), dtype=float)
    thresholds = np.zeros(23, dtype=float)
    for j, name in enumerate(POSE_INDEX_TO_SMPL):
        thresholds[j] = tight_deg if name in TIGHT_JOINT_NAMES else loose_deg

    flagged_mask = np.zeros((T, 23), dtype=bool)

    # compute angles
    for t, fname in enumerate(common):
        p1 = load_pose_from_smpl_json(os.path.join(dir1, fname))
        p2 = load_pose_from_smpl_json(os.path.join(dir2, fname))
        R1s = pose69_to_Rlist(p1)
        R2s = pose69_to_Rlist(p2)
        for j in range(23):
            theta_rad = geodesic_angle_between_R(R1s[j], R2s[j])
            ang_deg = math.degrees(theta_rad)
            angles_deg_all[t, j] = ang_deg
            if ang_deg > thresholds[j]:
                flagged_mask[t, j] = True

    # temporal filtering: chỉ giữ những run >= min_consecutive
    if min_consecutive > 1:
        filtered = np.zeros_like(flagged_mask)
        for j in range(23):
            arr = flagged_mask[:, j].astype(int)
            i = 0
            while i < T:
                if arr[i] == 1:
                    # count run
                    k = i
                    while k < T and arr[k] == 1:
                        k += 1
                    run_len = k - i
                    if run_len >= min_consecutive:
                        filtered[i:k, j] = True
                    i = k
                else:
                    i += 1
        flagged_mask = filtered

    # write csv
    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame_filename", "frame_index", "joint_idx", "joint_name",
                         "angle_deg", "threshold_deg", "flagged"])
        for t, fname in enumerate(common):
            for j in range(23):
                writer.writerow([fname, t, j, POSE_INDEX_TO_SMPL[j],
                                 f"{angles_deg_all[t,j]:.3f}",
                                 f"{thresholds[j]:.1f}",
                                 int(flagged_mask[t,j])])

    # print quick summary
    total_flags = np.sum(flagged_mask)
    print(f"Processed {T} frames. Total flagged (after temporal filter) = {total_flags}")
    if total_flags > 0:
        print("Một vài ví dụ flagged (frame, joint, angle_deg, thresh):")
        for t in range(T):
            for j in range(23):
                if flagged_mask[t,j]:
                    print(f"  {common[t]} (#{t}): {POSE_INDEX_TO_SMPL[j]} - {angles_deg_all[t,j]:.1f}° > {thresholds[j]:.1f}°")
    else:
        print("Không phát hiện flag nào (theo ngưỡng hiện tại).")
    print(f"CSV report written to: {out_csv}")

# --- CLI ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir1", required=True, help="Thư mục smpl của video1 (JSON frames)")
    ap.add_argument("--dir2", required=True, help="Thư mục smpl của video2 (JSON frames)")
    ap.add_argument("--out", default="report.csv", help="File CSV output")
    ap.add_argument("--tight_deg", type=float, default=30.0, help="ngưỡng (deg) cho tay/chân")
    ap.add_argument("--loose_deg", type=float, default=45.0, help="ngưỡng (deg) cho khớp khác")
    ap.add_argument("--min_consecutive", type=int, default=2, help="số frame liên tiếp tối thiểu để chấp nhận flag")
    args = ap.parse_args()

    compare_dirs(args.dir1, args.dir2, args.out, args.tight_deg, args.loose_deg, args.min_consecutive)
