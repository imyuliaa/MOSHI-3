import random 
from scipy.spatial.distance import euclidean 
from scipy.cluster.hierarchy import linkage, fcluster  
from scipy.spatial.distance import pdist  
import matplotlib.pyplot as plt  

def generate_data(N):
    return [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(N)]  # Генерація тестових даних

def distance(a, b):
    return euclidean(a, b)  # Функція для обчислення відстані між двома точками

def kmeans(data, k, max_iterations=100):
    centers = random.sample(data, k)  # Вибір випадкових центрів кластерів
    for i in range(max_iterations):
        clusters = [[] for _ in range(k)]  # Ініціалізація кластерів
        for point in data:
            distances = [distance(point, center) for center in centers]  # Обчислення відстаней між точкою та центрами кластерів
            cluster_index = distances.index(min(distances))  # Визначення індексу найближчого кластера
            clusters[cluster_index].append(point)  # Додавання точки до відповідного кластера
        new_centers = []
        for j in range(k):
            if clusters[j]:
                new_center = (
                    sum(point[0] for point in clusters[j]) / len(clusters[j]),  # Обчислення нового центру кластера
                    sum(point[1] for point in clusters[j]) / len(clusters[j])
                )
                new_centers.append(new_center)
            else:
                new_centers.append(centers[j])  # Збереження поточного центру, якщо кластер порожній
        if new_centers == centers:  # Перевірка на зупинку алгоритму, якщо центри не змінюються
            break
        centers = new_centers  # Оновлення центрів кластерів
    return clusters, centers  # Повернення кластерів та їх центрів

def hierarchical_clustering(data, k):
    distances = pdist(data)  # Обчислення відстаней між парами точок
    K = linkage(distances, method='ward')  # Виконання ієрархічної кластеризації
    clusters = fcluster(K, k, criterion='maxclust')  # Вибір кластерів
    return clusters  # Повернення міток кластерів

def compare_clusters(true_labels, pred_labels):
    n_clusters_true = len(set(true_labels))  # Визначення кількості унікальних міток кластерів у вихідних даних
    n_clusters_pred = len(set(pred_labels))  # Визначення кількості унікальних міток кластерів у кластеризованих даних
    print("Кількість кластерів у вихідних даних:", n_clusters_true)
    print("Кількість кластерів у кластеризованих даних:", n_clusters_pred)

    # Оцінка якості кластеризації (середньо-зважені розміри кластерів)
    cluster_sizes_true = {label: true_labels.count(label) for label in set(true_labels)}  # Підрахунок розмірів кластерів у вихідних даних
    cluster_sizes_pred = {label: pred_labels.count(label) for label in set(pred_labels)}  # Підрахунок розмірів кластерів у кластеризованих даних

    weighted_mean_true = sum(size * size for size in cluster_sizes_true.values()) / len(true_labels)  # Обчислення середньо-зваженого розміру кластерів у вихідних даних
    weighted_mean_pred = sum(size * size for size in cluster_sizes_pred.values()) / len(pred_labels)  # Обчислення середньо-зваженого розміру кластерів у кластеризованих даних

    print("Середньо-зважене розміри кластерів у вихідних даних:", weighted_mean_true)
    print("Середньо-зважене розміри кластерів у кластеризованих даних:", weighted_mean_pred)

# Генерація тестових даних
N = 1000
data = generate_data(N)

# Кількість кластерів
k = 10

# Кластеризація за методом K-середніх та ієрархічним методом
kmeans_clusters, kmeans_centers = kmeans(data, k)
hierarchical_clusters = hierarchical_clustering(data, k)

# Порівняння результатів кластеризації
print("K-means:")
for i, cluster in enumerate(kmeans_clusters):
    print("Cluster {}: {} points, center at {}".format(i + 1, len(cluster), kmeans_centers[i]))

print("\nHierarchical:")
for i in range(1, k + 1):
    cluster = [data[j] for j in range(len(hierarchical_clusters)) if hierarchical_clusters[j] == i]
    center = (
        sum(point[0] for point in cluster) / len(cluster),
        sum(point[1] for point in cluster) / len(cluster)
    )
    print("Cluster {}: {} points, center at {}".format(i, len(cluster), center))

# Виберемо кластерні центри з кластерів методу k-means
kmeans_cluster_labels = []
for i, cluster in enumerate(kmeans_clusters):
    for point in cluster:
        kmeans_cluster_labels.append(i)

# Порівняння кількості кластерів та якості кластеризації
print("\nПорівняння результатів кластеризації:")
compare_clusters([data.index(point) for point in data], kmeans_cluster_labels)

# Відображення точок даних та центрів кластерів для методу K-середніх
plt.subplot(1, 2, 1)
for i, cluster in enumerate(kmeans_clusters):
    x = [point[0] for point in cluster]
    y = [point[1] for point in cluster]
    plt.scatter(x, y, label="Cluster {}".format(i + 1))
for center in kmeans_centers:
    plt.scatter(center[0], center[1], s=100, marker='x', color='black', linewidths=2)
plt.title("K-means")

# Відображення точок даних для ієрархічної кластеризації
plt.subplot(1, 2, 2)
for i in range(1, k + 1):
    cluster = [data[j] for j in range(len(hierarchical_clusters)) if hierarchical_clusters[j] == i]
    x = [point[0] for point in cluster]
    y = [point[1] for point in cluster]
    plt.scatter(x, y, label="Cluster {}".format(i))
plt.title("Hierarchical")

plt.show()
