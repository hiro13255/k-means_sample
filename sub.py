import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# データセット取得関数
def make_dataset():
    x = np.r_[np.random.normal(size=20, loc=0, scale=2), np.random.normal(size=20, loc=10, scale=2), np.random.normal(
        size=20, loc=15, scale=2), np.random.normal(size=20, loc=40, scale=2)]


    y = np.r_[np.random.normal(size=20, loc=15, scale=2), np.random.normal(
        size=20, loc=3, scale=2), np.random.normal(size=20, loc=40, scale=2), np.random.normal(size=20, loc=0, scale=2)]
    return x, y

'''
 関数名 ：k平均法アルゴリズム関数
 引数　 ：
        k                   重心の個数
        x                   データ
        centers             重心点の位置
        times_recurring_max 繰り返し回数
 戻り値 ：なし
'''
def kmeans(k, x, centers, times_recurring_max):
    x_data_size = x.shape[0]

    old_centers = np.zeros((centers.shape[0], 2))

    distance = np.zeros(x_data_size)
    for limit in range(times_recurring_max):
        for i in range(x_data_size):
            # 重心とデータの距離の2乗が一番近い値を格納
            distance[i] = np.argmin(np.sum((x[i,:] - centers) ** 2, axis=1))

        # 重心の個数分ループして新しい位置を記録
        for j in range(k):
            centers[j, :] = x[distance == j, :].mean(axis=0)

        # 前の重心と一致していた場合は終了
        if np.array_equal(old_centers, centers):
            print("終了")
            break

        # 一つ前の重心を保持
        old_centers = centers

if __name__ == '__main__':
    dataset_x, dataset_y = make_dataset()

    X = np.c_[dataset_x, dataset_y]
    centers = np.array([[0, 5], [5, 0], [10, 15], [20, 10]])

    # 現状のデータ、重心を描画
    plt.scatter(X[:, 0], X[:, 1], c="black", s=10, alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], color=["r", "b", "g", "pink"])
    plt.show()

    # k平均法を適応したデータ、重心を表示
    kmeans(len(centers), X, centers, 10)
    plt.scatter(X[:, 0], X[:, 1], c="black", s=10, alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], color=["r", "b", "g", "pink"])
    plt.show()
