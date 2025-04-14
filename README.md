<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<h1 align="center"><b>Information Retrieval</b></h1>

<div align="center">
  <table>
    <thead>
      <tr>
        <th>STT</th>
        <th>MSSV</th>
        <th>Họ và Tên</th>
        <th>Chức vụ</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>22520646</td>
        <td>Nguyễn Quốc Khánh</td>
        <td>Nhóm trưởng</td>
      </tr>
      <tr>
        <td>2</td>
        <td>22520775</td>
        <td>Nguyễn Xuân Linh</td>
        <td>Thành viên</td>
      </tr>
      <tr>
        <td>3</td>
        <td>22521677</td>
        <td>Nguyễn Thế Vĩnh</td>
        <td>Thành viên</td>
      </tr>
      <tr>
        <td>4</td>
        <td>22521671</td>
        <td>Lưu Khánh Vinh</td>
        <td>Thành viên</td>
      </tr>
    </tbody>
  </table>
</div>

# COURSE INTRODUCTION
* **Course Name:** MLOps - Phát triển và vận hành hệ thống máy học.
* **Class Code:** CS317.P21.
* **Academic Year:** HK2 (2024 - 2025).
* **Lecturer**: Th.S Đỗ Văn Tiến, Lê Trần Trọng Khiêm
* **Subject**: Thực hành Lab01

## Training Pipeline 
Pipeline được chia thành 4 phần: 
- Data Preprocessing
- Training
- Validation
- Evaluation

Data sau khi được xử lý sẽ đưa qua bước training, sử dụng hyperparameters tuning để tối ưu model, đánh giá việc tuning bằng bước Validation, sau khi có được model tốt nhất thì chuyển qua bước Evaluation để thực hiện việc dự đoán data chưa được học.
1. **Data Preprocessing**
- Data được lấy từ [kaggle](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset?)
- Khử nhiễu bằng thuật toán Gaussian
- Resize về cùng 1 kích thước :(224,224)
- Sử dụng Standard Scaler để chuẩn hóa dữ liệu
- Convert từ numpy sang tensor để tương thích với framework Pytorch
- Sử dụng Dataloader của pytorch với batchsize = 16, để tận dụng việc xử lý đa luồng của GPU (Nếu có)
- Chia data thành 3 tập:
  - Train : 70% 
  - Validation: 20%
  - Evaluation: 10%
2. **Training**
- Sử dụng model CNN với 3 layer
- Các hyperparameter trong quá trình train:
  - optimizer: Adam
  - loss_function : Cross_Entropy
  - tuing learning_rate : [0.001; 0.01; 0.1 ]
  - epochs : 10
- Điểm mới:
Các metrics và hyperparameters sau khi training sẽ được lưu trên mlflow, thay vì log ra màn hình. Sử dụng mlflow để lưu trữ và quản lý  các checkpoint.
3. **Validation**
- Sau khi model được training qua mỗi epochs, sẽ được đánh giá bằng tập validation
- Thông qua loss và accuracy của bước này sẽ đánh giá được việc tuning có hiệu quá không, từ đó đưa ra các điều chỉnh phù hợp.
- Các metrics và loss của bươc này cũng được log lại trên mlflow
4. **Evaluation**
- Sau khi đã lựa chọn được các hyperparameters và model cho được kết quả tốt nhất thì tiến hành đánh giá khả năng ứng dụng thực tế của model thông qua tập Evaluation
- Nếu đạt kết quả đặt ra thì tiến hành deploy, còn không thì tiếp tục quay trở lại bước data preprocessing tiếp tục đi thử nghiệm và đánh giá phương pháp khác.
## Framework và Công nghệ sử dụng:
Dự án được xây dựng trên nền tảng Python với hệ sinh thái thư viện phong phú. PyTorch được chọn làm framework deep learning chính nhờ sự linh hoạt, khả năng tính toán GPU và Dataloader hiệu quả cho mô hình CNN. MLflow đóng vai trò then chốt trong việc theo dõi, quản lý thử nghiệm và mô hình, đảm bảo tính tái lập. Các thư viện như NumPy, Scikit-learn (cho StandardScaler), Scipy, Pillow và OpenCV hỗ trợ đắc lực cho việc xử lý dữ liệu số và ảnh (resize, khử nhiễu). Dữ liệu được lấy từ Kaggle và mã nguồn được quản lý bằng Git. Chi tiết được thể hiện dưới đây:
* **Ngôn ngữ lập trình:** Python
* **Thư viện Deep Learning:** PyTorch (cho mô hình CNN, Dataloader, optimizer, loss function)
* **Experiment Tracking:** MLflow (để log tham số, metrics, quản lý checkpoints và kết quả thử nghiệm)
* **Xử lý dữ liệu:**
    * NumPy
    * Scikit-learn
    * Scipy
    * Pillow, OpenCV
* **Nguồn dữ liệu:** Kaggle
* **Quản lý phiên bản:** Git
## Hướng dẫn chạy mlflow và pipeline training
1. **Install python 3.10**
- Linux :
```sh
$ sudo apt-get update
$ sudo apt-get install python3.10 python3.10-venv
```
- Window: có thể tải về máy ([tại đây](https://www.python.org/downloads/))
2. **Clone git**
- Đảm bảo đã tải git về máy ([tải tại đây](https://git-scm.com/downloads))
- Mở Folder để lưu dữ liệu được tải về, sau đó khởi tạo git
```sh
git init
```
- Clone code về:
```sh
git clone git@github.com:NguyenQuocKhanh1301/CS317_Lab01.git
```
3. **Tạo môi trường ảo (venv or conda ...)**
- Chuyển qua file đã clone về
```sh
cd your_path/CS317_Lab01
```
- Mở terminal và khởi tạo môi trường ảo với venv:
  ```sh
  python -m venv venv
  ```
- Activate môi trường ảo
  - Linux
  ```sh
  source venv/bin/activate
  ```
  - Window
 ```sh
.\venv\Scripts\activate
 ```
4. **Cài đặt thư viện**

Cài đặt các thư viện cần thiết:
```sh
pip install -r requirements.txt
```
5. **Mở giao diện mlflow**

Sử dụng lệnh sau để ui của mlflow với
- host: 0.0.0.0 
- port: 5003
```sh
mlflow ui --host 0.0.0.0 --port 5003
```
6. **Run pipeline**

Để chạy pipelin training sử dụng câu lệnh sau:
```sh
python lab01.py
```
7. **Load model và đánh giá**

Sau khi chon được model tốt nhất thì tiến hành đánh giá:
```ssh
python Evaluation.py
```
## Video demo:

