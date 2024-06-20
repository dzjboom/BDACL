from sklearn.random_projection import GaussianRandomProjection
import numpy as np

# 输入向量 b
b = np.array([0, 0, 1])

# 升维到 1024 维
n_components = 1024

# 创建 GaussianRandomProjection 模型
random_projector = GaussianRandomProjection(n_components=n_components, random_state=42)

# 使用 fit_transform 对向量 b 进行随机投影
b_projected = random_projector.fit_transform(b.reshape(1, -1))

# 打印结果
print("Original vector (b):", b)
print("Projected vector:", b_projected.flatten())
print("Shape of the projected vector:", b_projected.shape)
