import json

import numpy as np
import treelite
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor


def process_single_tree(tree, *, n_features):
    internal_nodes = []
    leaf_nodes = []
    parent = {}

    for node_id, node in enumerate(tree):
        if "left_child" in node:
            parent[node["left_child"]] = node_id
            parent[node["right_child"]] = node_id
            internal_nodes.append(node_id)
        else:
            leaf_nodes.append(node_id)

    A = np.zeros((n_features, len(internal_nodes)), dtype=np.bool)
    B = np.zeros((len(internal_nodes),), dtype=np.float64)
    C = np.zeros((len(internal_nodes), len(leaf_nodes)), dtype=np.int32)
    D = np.zeros((len(leaf_nodes),), dtype=np.int32)
    E = np.zeros((len(leaf_nodes), 1), dtype=np.int32)
    for j, e in enumerate(internal_nodes):
        A[tree[e]["split_feature_id"], j] = 1
        B[j] = tree[e]["threshold"]

    for j, e in enumerate(leaf_nodes):
        E[j] = e
        k = e
        while k != 0:
            p = parent[k]
            if tree[p]["left_child"] == k:
                C[internal_nodes.index(p), j] = 1
                D[j] += 1
            else:
                C[internal_nodes.index(p), j] = -1
            k = p

    return A, B, C, D, E


def main():
    X, y = make_regression(n_features=5, n_informative=5, random_state=0)
    clf = RandomForestRegressor(n_estimators=1, max_depth=12, random_state=0)
    clf.fit(X, y)

    tl_model = treelite.sklearn.import_model(clf)
    obj = json.loads(tl_model.dump_as_json())

    tree = obj["trees"][0]["nodes"]
    A, B, C, D, E = process_single_tree(tree, n_features=X.shape[1])

    rng = np.random.default_rng(seed=0)
    X_test = rng.standard_normal((1000, X.shape[1]))
    T = X_test @ A
    T = T <= B
    T = T @ C
    T = T == D
    leaf_pred = T @ E
    leaf_pred = leaf_pred.flatten().tolist()
    pred = [tree[e]["leaf_value"] for e in leaf_pred]

    np.testing.assert_almost_equal(clf.predict(X_test), pred)


if __name__ == "__main__":
    main()
