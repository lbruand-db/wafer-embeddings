import numpy as np

from wafer_embeddings.eval import metrics as me
from wafer_embeddings.eval import search as se


def _two_class_corpus():
    """6 items, 2 classes; within-class embeddings identical, classes orthogonal."""
    emb = np.zeros((6, 4), dtype=np.float64)
    emb[:3, 0] = 1.0  # class 0
    emb[3:, 1] = 1.0  # class 1
    labels = np.zeros((6, 2), dtype=np.float64)
    labels[:3, 0] = 1.0
    labels[3:, 1] = 1.0
    return emb, labels


def test_topk_cosine_and_exclude_diagonal():
    emb, _ = _two_class_corpus()
    idx, sims = se.topk_cosine(emb, emb, k=1, exclude_diagonal=True)
    # top neighbor (excluding self) of a class-0 row is another class-0 row.
    assert idx[0, 0] in (1, 2)
    assert np.isclose(sims[0, 0], 1.0)


def test_weighted_knn_predicts_correct_class():
    train = np.eye(2)[np.array([0, 0, 1, 1])].astype(np.float64)  # 4 x 2
    train_y = np.array([0, 0, 1, 1])
    q = np.array([[0.9, 0.1], [0.1, 0.9]])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    pred = se.weighted_knn_predict(train, train_y, q, k=2)
    assert pred.tolist() == [0, 1]


def test_pairwise_relevant_exact_and_jaccard_and_normal():
    q = np.array([[1, 0, 0], [0, 0, 0]])  # {0}, {} (Normal)
    d = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]])  # {0}, {0,1}, {}
    exact = me.pairwise_relevant(q, d, mode="exact")
    assert exact.tolist() == [[True, False, False], [False, False, True]]
    jac = me.pairwise_relevant(q, d, mode="jaccard", tau=0.5)
    assert jac[0].tolist() == [True, True, False]  # {0}vs{0,1} jaccard .5 >= .5


def test_retrieval_metrics_perfect():
    emb, labels = _two_class_corpus()
    m = me.retrieval_metrics(emb, emb, labels, labels, ks=(1, 2), mode="exact")
    assert np.isclose(m["precision@1"], 1.0)
    assert np.isclose(m["precision@2"], 1.0)
    assert np.isclose(m["recall@2"], 1.0)
    assert np.isclose(m["map@2"], 1.0)
    assert np.isclose(m["ndcg@2"], 1.0)
    assert np.isclose(m["mrr"], 1.0)


def test_retrieval_metrics_bad_ranking_below_perfect():
    # Relevant items are the LEAST similar -> mAP must drop below 1.
    emb = np.zeros((4, 2))
    emb[:, 0] = [1.0, 0.9, -0.9, -1.0]
    emb = emb / np.linalg.norm(emb + 1e-9, axis=1, keepdims=True)
    labels = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    m = me.retrieval_metrics(emb, emb, labels, labels, ks=(1,), mode="exact")
    assert m["map@1"] < 1.0


def test_clustering_metrics_perfect():
    labels = np.array([0, 0, 1, 1, 2, 2])
    emb = np.eye(3)[labels].astype(np.float64)
    m = me.clustering_metrics(labels, labels, embeddings=emb)
    assert np.isclose(m["ari"], 1.0) and np.isclose(m["nmi"], 1.0)
    assert m["silhouette"] > 0.9


def test_effective_rank_collapse_vs_isotropic():
    rng = np.random.default_rng(0)
    collapsed = np.outer(rng.standard_normal(500), rng.standard_normal(10))  # rank-1
    assert me.effective_rank(collapsed) < 1.5
    iso = rng.standard_normal((2000, 10))
    assert me.effective_rank(iso) > 7.0


def test_alignment_and_uniformity():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((50, 8))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    assert np.isclose(me.alignment(z, z), 0.0)
    assert me.alignment(z, -z) > 0.0
    # Spreading points out lowers uniformity (more negative) vs identical points.
    identical = np.ones((20, 8)) / np.sqrt(8)
    assert me.uniformity(z) < me.uniformity(identical)


def test_rankme_counts_effective_dimensions():
    rng = np.random.default_rng(0)
    iso = rng.standard_normal((4000, 16))
    assert 15.0 < me.rankme(iso) <= 16.0 + 1e-6  # isotropic -> ~D
    basis = np.linalg.qr(rng.standard_normal((16, 16)))[0][:3]
    rank3 = rng.standard_normal((4000, 3)) @ basis  # lives in a 3-d subspace
    assert 2.5 < me.rankme(rank3) < 3.2
    one = np.ones((100, 16)) * rng.standard_normal((100, 1))  # rank 1
    assert me.rankme(one) < 1.1
    assert me.rankme(np.zeros((5, 4))) == 1.0


def test_rankme_is_scale_invariant_and_bounded_by_samples():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 32))
    assert abs(me.rankme(x) - me.rankme(10.0 * x)) < 1e-6
    assert me.rankme(rng.standard_normal((8, 64))) <= 8.0 + 1e-6  # rank <= min(N, D)


def test_rankme_less_dominated_by_top_direction_than_effective_rank():
    # one strong direction + 15 weak ones: the squared spectrum hides the weak ones
    rng = np.random.default_rng(2)
    x = rng.standard_normal((5000, 16)) * np.array([10.0] + [1.0] * 15)
    assert me.rankme(x) > 2 * me.effective_rank(x)
