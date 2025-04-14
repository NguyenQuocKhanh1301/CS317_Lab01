# CS317 : Phát triển và vận hành hệ thống máy học
# Thực hành Lab01

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
1. Install python 3.10
2. Clone git
3. Tạo môi trường ảo
4. Cài đặt thư viện
5. Mở giao diện mlflow
6. run pipeline
7. Load model và đánh giá
## Video demo:

