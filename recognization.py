import cv2
import numpy as np
import torch
from model.mtcnn import MTCNN
from model.inception_resnet_v1 import InceptionResnetV1
import pymysql

# 加载MTCNN人脸检测器和FaceNet人脸识别模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 读取员工信息
UID = "001"

# 连接到 MySQL 数据库
mydb = pymysql.connect(
  host="localhost",
  port=3306,
  user="root",
  password="root",
  database="face_attendance"
)

# 获取指向数据库的游标
mycursor = mydb.cursor()

# 查询数据库中的嵌入
mycursor.execute("SELECT embedding FROM embeddings")

# 获取查询结果中所有的 embedding 数据
results = mycursor.fetchall()

# 存储所有 embedding 数据到一个列表中
embeddings = []
for embedding in results:
    embeddings.append(np.frombuffer(embedding[0], dtype=np.float32))


# 关闭游标和数据库连接
mycursor.close()
mydb.close()

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # 检测人脸
    boxes, probs = mtcnn.detect(frame)

    if boxes is None:
        print("No face detected")
        continue

    if len(boxes) > 1:
        print("Make sure that only you are in the box")
        continue

    if boxes is not None:
        x1, y1, x2, y2 = boxes[0].astype(np.int32)

        # 提取人脸图像
        face = frame[y1:y2, x1:x2]

        # 转换输入数据的通道数
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # 将BGR格式的图像转为RGB格式
        face_tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).float().to(device)[0]

        # 进行人脸识别
        face_embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()
        face_embedding /= np.linalg.norm(face_embedding, axis=1, keepdims=True)
        distances = np.linalg.norm(face_embedding - embeddings, axis=1)
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]

        # 判断是否为员工
        if min_distance < 0.5:
            print("Success")
            break
            # 在画面中显示员工姓名和打卡时间
            # ...
        else:
            print("Not the person whose UID is " + UID)
            break
            # 在画面中显示未知人脸信息
            # ...

        # 在画面中标注人脸框和姓名
        # ...

    cv2.imshow('Face Recognition Attendance System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
