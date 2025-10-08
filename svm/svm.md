## 🧩 1. HOG (Histogram of Oriented Gradients)

### 📄 Bài báo gốc:

> **Dalal, N., & Triggs, B. (2005).**
> *Histograms of Oriented Gradients for Human Detection.*
> *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2005),* 1, 886–893.
> [DOI: 10.1109/CVPR.2005.177](https://doi.org/10.1109/CVPR.2005.177)

### 🔍 Ý nghĩa:

* Đây là bài báo **kinh điển**, lần đầu tiên giới thiệu HOG — phương pháp trích đặc trưng hình dạng dựa trên hướng gradient cục bộ.
* HOG được thiết kế cho **phát hiện người**, nhưng nhanh chóng trở thành đặc trưng phổ biến cho **phân loại hình ảnh y tế, phương tiện, khuôn mặt**, v.v.
* Trong code của bạn, hàm:

  ```python
  hog(img_resized, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2))
  ```

  chính là **triển khai ý tưởng của Dalal & Triggs (2005)** — biểu đồ hướng gradient chuẩn hóa trên khối.

---

## 📉 2. PCA (Principal Component Analysis)

### 📄 Bài báo gốc:

> **Pearson, K. (1901).**
> *On Lines and Planes of Closest Fit to Systems of Points in Space.*
> *Philosophical Magazine,* 2(11), 559–572.
> [Link (archive)](https://statml.univie.ac.at/fileadmin/user_upload/p_statml/Lehre/PCA_1901_Pearson.pdf)

### 📄 Bài tham khảo quan trọng hiện đại:

> **Jolliffe, I. T. (1986).**
> *Principal Component Analysis.* Springer Series in Statistics. Springer, New York.
> ISBN 978-0-387-95442-4

### 🔍 Ý nghĩa:

* PCA được dùng để **giảm chiều dữ liệu** mà vẫn giữ phần lớn phương sai (năng lượng) — giảm nhiễu, tăng tốc độ huấn luyện SVM.
* Trong code:

  ```python
  pca = PCA(n_components=0.95)
  ```

  là cách gọi trong scikit-learn để **giữ lại 95% phương sai** (theo khuyến nghị của Jolliffe).
* PCA là bước trung gian rất phổ biến trong pipeline **HOG → PCA → SVM**, đặc biệt trong phân loại ảnh y học, vì HOG thường tạo ra vector rất dài (hàng ngàn chiều).

---

## ⚙️ 3. SVM (Support Vector Machine)

### 📄 Bài báo gốc:

> **Cortes, C., & Vapnik, V. (1995).**
> *Support-Vector Networks.*
> *Machine Learning,* 20(3), 273–297.
> [DOI: 10.1007/BF00994018](https://doi.org/10.1007/BF00994018)

### 📄 Bài mở rộng cho phân loại đa lớp (multi-class):

> **Hsu, C. W., & Lin, C. J. (2002).**
> *A Comparison of Methods for Multiclass Support Vector Machines.*
> *IEEE Transactions on Neural Networks,* 13(2), 415–425.
> [DOI: 10.1109/72.991427](https://doi.org/10.1109/72.991427)

### 🔍 Ý nghĩa:

* Cortes & Vapnik (1995) là bài báo **khai sinh ra SVM** — một trong những thuật toán nền tảng của machine learning.
* Hsu & Lin (2002) là bài báo **chuẩn hoá khái niệm “one-vs-one” (OvO)** và “one-vs-rest” (OvR) cho SVM — đúng với phần bạn dùng:

  ```python
  SVC(decision_function_shape='ovo')
  ```

  để giải bài toán phân loại đa lớp.

---

## ⚖️ 4. Class Weighting for Imbalanced Data

### 📄 Bài báo tham khảo:

> **He, H., & Garcia, E. A. (2009).**
> *Learning from Imbalanced Data.*
> *IEEE Transactions on Knowledge and Data Engineering,* 21(9), 1263–1284.
> [DOI: 10.1109/TKDE.2008.239](https://doi.org/10.1109/TKDE.2008.239)

### 🔍 Ý nghĩa:

* Trình bày các kỹ thuật cân bằng trọng số lớp (`class_weight='balanced'` trong code của bạn).
* Đây là tài liệu tổng hợp quan trọng nhất cho vấn đề **mất cân bằng lớp (class imbalance)** trong SVM và các mô hình học máy khác.

---

## 🔄 5. Stratified K-Fold Cross Validation

### 📄 Bài báo tham khảo:

> **Kohavi, R. (1995).**
> *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.*
> *Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI),* 2, 1137–1143.

### 🔍 Ý nghĩa:

* Cơ sở lý thuyết của việc dùng `StratifiedKFold` trong GridSearchCV để đảm bảo tỉ lệ lớp ổn định giữa các folds.
* Đây là thực hành chuẩn trong tất cả pipeline học máy hiện đại.

---

## 📚 Tổng hợp trích dẫn chuẩn (APA)

1. Dalal, N., & Triggs, B. (2005). *Histograms of oriented gradients for human detection.* IEEE CVPR, 1, 886–893.
2. Pearson, K. (1901). *On lines and planes of closest fit to systems of points in space.* Philosophical Magazine, 2(11), 559–572.
3. Jolliffe, I. T. (1986). *Principal component analysis.* Springer Series in Statistics.
4. Cortes, C., & Vapnik, V. (1995). *Support-vector networks.* Machine Learning, 20(3), 273–297.
5. Hsu, C. W., & Lin, C. J. (2002). *A comparison of methods for multiclass support vector machines.* IEEE Transactions on Neural Networks, 13(2), 415–425.
6. He, H., & Garcia, E. A. (2009). *Learning from imbalanced data.* IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263–1284.
7. Kohavi, R. (1995). *A study of cross-validation and bootstrap for accuracy estimation and model selection.* IJCAI, 1137–1143.

---

## 🔗 Tóm tắt kết nối với đoạn code của bạn

| Thành phần trong code                | Kỹ thuật         | Bài báo chính                            | Vai trò                            |
| ------------------------------------ | ---------------- | ---------------------------------------- | ---------------------------------- |
| `skimage.feature.hog()`              | HOG              | Dalal & Triggs (2005)                    | Trích đặc trưng hình dạng từ ảnh   |
| `PCA(n_components=0.95)`             | PCA              | Pearson (1901), Jolliffe (1986)          | Giảm chiều dữ liệu                 |
| `SVC(decision_function_shape='ovo')` | SVM (One-vs-One) | Cortes & Vapnik (1995), Hsu & Lin (2002) | Phân loại đa lớp                   |
| `class_weight='balanced'`            | Cân bằng lớp     | He & Garcia (2009)                       | Giảm bias cho lớp hiếm             |
| `StratifiedKFold`                    | Cross-validation | Kohavi (1995)                            | Đảm bảo phân phối lớp đều trong CV |

