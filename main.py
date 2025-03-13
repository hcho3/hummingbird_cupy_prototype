import json
import time

import cupy as cp
import numpy as np
import treelite
from cuml.experimental import ForestInference
from nvmath.bindings import cublas
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor


def process_single_tree(tree, *, n_features):
    internal_nodes = []
    leaf_nodes = []
    parent = {}

    for node_id, node in enumerate(tree["nodes"]):
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
        A[tree["nodes"][e]["split_feature_id"], j] = 1
        B[j] = tree["nodes"][e]["threshold"]

    for j, e in enumerate(leaf_nodes):
        E[j] = e
        k = e
        while k != 0:
            p = parent[k]
            if tree["nodes"][p]["left_child"] == k:
                C[internal_nodes.index(p), j] = 1
                D[j] += 1
            else:
                C[internal_nodes.index(p), j] = -1
            k = p

    return A, B, C, D, E


def process_rf(trees, *, n_features):
    A, B, C, D, E = [], [], [], [], []
    for tree in trees:
        a, b, c, d, e = process_single_tree(tree, n_features=n_features)
        A.append(a)
        B.append(b)
        C.append(c)
        D.append(d)
        E.append(e)
    # Highest number of internal nodes in all trees
    n_internal_nodes = max([a.shape[1] for a in A])
    # Highest number of leaf nodes in all trees
    n_leaf_nodes = max([c.shape[1] for c in C])
    # Apply padding
    A = [np.pad(a, ((0, 0), (0, n_internal_nodes - a.shape[1]))) for a in A]
    for a in A:
        assert a.shape == (n_features, n_internal_nodes)
    B = [np.pad(b, (0, n_internal_nodes - b.shape[0])) for b in B]
    for b in B:
        assert b.shape == (n_internal_nodes,)
    C = [
        np.pad(c, ((0, n_internal_nodes - c.shape[0]), (0, n_leaf_nodes - c.shape[1])))
        for c in C
    ]
    for c in C:
        assert c.shape == (n_internal_nodes, n_leaf_nodes)
    D = [np.pad(d, (0, n_leaf_nodes - d.shape[0])) for d in D]
    for d in D:
        assert d.shape == (n_leaf_nodes,)
    E = [np.pad(e, ((0, n_leaf_nodes - e.shape[0]), (0, 0))) for e in E]
    for e in E:
        assert e.shape == (n_leaf_nodes, 1)
    return {
        "A": cp.array(np.hstack(A), order="F"),
        "B": cp.array(np.hstack(B), order="F"),
        "C": cp.array(np.hstack(C), order="F", dtype=cp.float32),
        "D": cp.array(np.hstack(D), order="F", dtype=cp.float32),
        "E": cp.array(np.hstack(E), order="F", dtype=cp.float32),
        "n_internal_nodes": n_internal_nodes,
        "n_leaf_nodes": n_leaf_nodes,
        "n_features": n_features,
        "n_trees": len(trees),
    }


def naive_infer(X_test, *, processed_rf):
    n_trees = processed_rf["n_trees"]
    nin = processed_rf["n_internal_nodes"]
    nl = processed_rf["n_leaf_nodes"]

    T = X_test @ processed_rf["A"]
    T = T <= processed_rf["B"]

    # T = T @ C
    T = [
        T[:, (nin * i) : (nin * i + nin)]
        @ processed_rf["C"][:, (nl * i) : (nl * i + nl)]
        for i in range(n_trees)
    ]
    T = cp.hstack(T)

    # T = T == D
    T = T == processed_rf["D"]

    # leaf_pred = T @ E
    leaf_pred = [
        T[:, (nl * i) : (nl * i + nl)] @ processed_rf["E"][:, i : (i + 1)]
        for i in range(n_trees)
    ]
    leaf_pred = cp.hstack(leaf_pred)
    return leaf_pred


def infer(X_test: cp.ndarray, *, processed_rf):
    n_trees = processed_rf["n_trees"]
    nin = processed_rf["n_internal_nodes"]
    nl = processed_rf["n_leaf_nodes"]

    T = cp.matmul(X_test, processed_rf["A"], order="F")
    T = T <= processed_rf["B"]
    T = cp.asfortranarray(T.astype(cp.float32))

    # T = T @ C
    result = cp.zeros((X_test.shape[0], nl * n_trees), dtype=cp.float32, order="F")
    alpha = np.array([1], dtype=np.float32, order="F")
    beta = np.array([0], dtype=np.float32, order="F")
    cublas_handle = cublas.create()
    cublas.set_pointer_mode(cublas_handle, cublas.PointerMode.HOST)
    cublas.sgemm_strided_batched(
        cublas_handle,
        cublas.Operation.N,
        cublas.Operation.N,
        X_test.shape[0],
        nl,
        nin,
        alpha.ctypes.data,
        T.data.ptr,
        X_test.shape[0],
        X_test.shape[0] * nin,
        processed_rf["C"].data.ptr,
        nin,
        nin * nl,
        beta.ctypes.data,
        result.data.ptr,
        X_test.shape[0],
        X_test.shape[0] * nl,
        n_trees,
    )
    T = result

    # T = T == D
    T = T == processed_rf["D"]
    T = cp.asfortranarray(T.astype(cp.float32))

    # leaf_pred = T @ E
    leaf_pred = cp.zeros((X_test.shape[0], n_trees), dtype=cp.float32, order="F")
    alpha = np.array([1], dtype=np.float32, order="F")
    beta = np.array([0], dtype=np.float32, order="F")
    cublas.sgemm_strided_batched(
        cublas_handle,
        cublas.Operation.N,
        cublas.Operation.N,
        X_test.shape[0],
        1,
        nl,
        alpha.ctypes.data,
        T.data.ptr,
        X_test.shape[0],
        X_test.shape[0] * nl,
        processed_rf["E"].data.ptr,
        nl,
        nl,
        beta.ctypes.data,
        leaf_pred.data.ptr,
        X_test.shape[0],
        X_test.shape[0],
        n_trees,
    )
    return leaf_pred


def main():
    X, y = make_regression(n_features=100, n_informative=20, random_state=0)
    clf = RandomForestRegressor(n_estimators=1000, max_depth=3, random_state=0)
    clf.fit(X, y)

    tl_model = treelite.sklearn.import_model(clf)
    obj = json.loads(tl_model.dump_as_json())

    processed_rf = process_rf(obj["trees"], n_features=X.shape[1])

    rng = cp.random.default_rng(seed=0)
    X_test = rng.standard_normal((10000, X.shape[1]))
    expected_leaf_output = treelite.gtil.predict_leaf(tl_model, X_test.get())

    np.testing.assert_array_equal(
        naive_infer(X_test, processed_rf=processed_rf).get().astype("float64"),
        expected_leaf_output,
    )

    # Warm-up
    n_trials = 20
    X_test = cp.asfortranarray(X_test)
    for _ in range(n_trials):
        _ = infer(X_test, processed_rf=processed_rf).get().astype("float64")

    # Hummingbird
    tstart = time.perf_counter()
    for _ in range(n_trials):
        pred = infer(X_test, processed_rf=processed_rf).get().astype("float64")
    tend = time.perf_counter()
    print(f"Hummingbird: Time elapsed = {(tend - tstart) / n_trials} sec")
    np.testing.assert_array_equal(pred, expected_leaf_output)

    fm = ForestInference.load_from_treelite_model(tl_model)
    X_test = cp.ascontiguousarray(X_test)
    tstart = time.perf_counter()
    for _ in range(n_trials):
        pred = fm.apply(X_test).get().astype("float64")
    tend = time.perf_counter()
    print(f"FIL: Time elapsed = {(tend - tstart) / n_trials} sec")


if __name__ == "__main__":
    main()
