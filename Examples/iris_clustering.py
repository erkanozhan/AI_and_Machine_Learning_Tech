from __future__ import annotations

import os
from collections import Counter

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler


def main() -> None:
    iris = load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=3, n_init=20, random_state=42)
    cluster_labels = model.fit_predict(X_scaled)

    print("Siluet skoru:", round(silhouette_score(X_scaled, cluster_labels), 3))
    print("Duzeltilmis Rand indeksi:", round(adjusted_rand_score(y, cluster_labels), 3))

    print("\nKume buyuklukleri")
    for cluster_id, size in sorted(Counter(cluster_labels).items()):
        print(f"Kume {cluster_id}: {size} ornek")

    print("\nKume merkezleri (orijinal olcekte)")
    original_centers = scaler.inverse_transform(model.cluster_centers_)
    for cluster_id, center in enumerate(original_centers):
        values = ", ".join(f"{value:.2f}" for value in center)
        print(f"Kume {cluster_id}: {values}")

    print("\nGercek sinif - kume karsilastirma matrisi")
    matrix = contingency_matrix(y, cluster_labels)
    print(matrix)

    print("\nSatirlar gercek turleri gosterir:")
    for class_id, class_name in enumerate(iris.target_names):
        print(f"{class_id}: {class_name}")


if __name__ == "__main__":
    main()
