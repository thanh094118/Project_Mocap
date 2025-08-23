# 🎥 Dự án Mocap - Monocular Video

## 📌 Mô tả tổng quan
Dự án này sử dụng thư viện [EasyMocap](https://github.com/zju3dv/EasyMocap) để thực hiện **3D Human Motion Capture** từ **video monocular (một camera duy nhất)**.  
Hệ thống hỗ trợ trích xuất khung hình, keypoints 2D, ước lượng dáng người 3D và tái tạo chuyển động theo thời gian.

---

## ⚡ Các tính năng chính
- 🖼️ **Extract Image**: Tách khung hình từ video đầu vào.  
- 📍 **Extract Keypoint2D**: Dự đoán keypoints 2D từ từng khung hình.  
- 🧍 **Estimate Pose 3D**: Ước lượng tư thế người ở không gian 3D.  
- 🎞️ **Reconstruction**: Tái dựng và theo dõi chuyển động khung xương.  
- 🔎 **Phân tích, tối ưu**:  
  - Lọc keypoints dựa trên confidence score.  
  - Đồng bộ và tinh chỉnh chuyển động theo khung hình.  
  - Tối ưu tham số khung xương để giảm nhiễu trong pose.  

---

## 🛠️ Cài đặt

1. Clone repository gốc
```bash
git clone https://github.com/zju3dv/EasyMocap.git
cd EasyMocap

2. Cài đặt môi trường

Khuyến nghị dùng conda:
conda create -n easymocap python=3.8
conda activate easymocap
pip install -r requirements.txt
3. Cài đặt thêm các mô hình hỗ trợ

Tải pretrained models theo hướng dẫn tại: https://chingswy.github.io/easymocap-public-doc/
Cách chạy
1. Extract frames từ video
python apps/preprocess/extract_video.py --input input.mp4 --output output/images

2. Dự đoán keypoints 2D
python apps/preprocess/extract_keypoints.py --input output/images --output output/keypoints2d

3. Ước lượng pose 3D (monocular pipeline)
python apps/demo/mocap_video.py --input input.mp4 --output output/mocap

📚 Nguồn tài liệu

Thư viện gốc: https://github.com/zju3dv/EasyMocap

Tài liệu hướng dẫn setup & chạy: https://chingswy.github.io/easymocap-public-doc/