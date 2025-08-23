<!--
 * @Date: 2025-08-24
 * @Author: Your Name
 * @Project: Mocap - Monocular Video
 * @FilePath: /Readme.md
-->

<div align="center">
    <img src="logo.png" width="40%">
</div>

**Mocap - Monocular Video** là một dự án demo dựa trên [EasyMocap](https://github.com/zju3dv/EasyMocap), tập trung vào **3D human motion capture** từ **video monocular (một camera)**.  
Dự án cung cấp pipeline: **extract frames → keypoints2D → pose3D → visualize motion**.

![python](https://img.shields.io/badge/python-3.8+-blue)
![pytorch](https://img.shields.io/badge/pytorch-1.8+-orange)
![star](https://img.shields.io/github/stars/zju3dv/EasyMocap?style=social)

---

## ✨ Tính năng chính
- 🖼️ **Extract Frames**: Trích xuất khung hình từ video.  
- 📍 **Keypoints 2D**: Phát hiện keypoints 2D bằng OpenPose / HRNet / MediaPipe.  
- 🧍 **Pose Estimation 3D**: Ước lượng tư thế người trong không gian 3D.  
- 🎥 **Monocular Mocap Pipeline**: Toàn bộ quy trình từ video → 3D motion.  
- 🔎 **Phân tích & tối ưu**:  
  - Lọc keypoints theo confidence score.  
  - Tinh chỉnh pose bằng thuật toán tối ưu.  
  - Giảm nhiễu trong chuyển động khung xương.  

<div align="center">
    <img src="demo_monocular.gif" width="80%">
    <br>
    <sup>Ví dụ pipeline monocular video → 3D skeleton</sup>
</div>

---

## ⚡ Cài đặt & Chạy

```bash
# 1. Clone repo EasyMocap
git clone https://github.com/zju3dv/EasyMocap.git
cd EasyMocap

# 2. Tạo môi trường Python
conda create -n easymocap python=3.8 -y
conda activate easymocap
pip install -r requirements.txt

# 3. Tải pretrained models (theo hướng dẫn)
# 👉 https://chingswy.github.io/easymocap-public-doc/install/install.html

# 4. Extract frames từ video
python apps/preprocess/extract_video.py --input input.mp4 --output output/images

# 5. Extract keypoints 2D
python apps/preprocess/extract_keypoints.py --input output/images --output output/keypoints2d

# 6. Chạy monocular mocap pipeline
python apps/demo/mocap_video.py --input input.mp4 --output output/mocap
