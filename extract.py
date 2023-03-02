import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pymysql

# 加载MTCNN人脸检测器和FaceNet人脸识别模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 读取员工人脸图像并生成特征向量
embedding = []
UID = "002"

for j, img_path in enumerate(os.listdir(f'./{UID}')):  # 一个UID对应一张人脸？？
    img = cv2.imread(os.path.join(f'./{UID}', img_path))

    # 检测人脸
    boxes, probs = mtcnn.detect(img)
    if boxes is not None:
        # 提取人脸图像并进行人脸识别
        x1, y1, x2, y2 = boxes[0].astype(np.int32)
        face = img[y1:y2, x1:x2]
        # 转换输入数据的通道数
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # 将BGR格式的图像转为RGB格式
        face_tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).float().to(device)[0]
        face_embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()
        face_embedding /= np.linalg.norm(face_embedding, axis=1, keepdims=True)
        embedding = face_embedding[0]



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

# 创建储存人脸embedding的表
mycursor.execute("CREATE TABLE IF NOT EXISTS embeddings (id INT AUTO_INCREMENT PRIMARY KEY, UID VARCHAR(255), embedding BLOB)")

# 将人脸embedding存入数据库
mycursor.execute("INSERT INTO embeddings (UID, embedding) VALUES (%s, %s)", (UID, embedding.tobytes()))

# 提交更改
mydb.commit()

# 关闭游标和数据库连接
mycursor.close()
mydb.close()

