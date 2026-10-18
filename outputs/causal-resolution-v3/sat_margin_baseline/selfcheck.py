"""Synthetic checks only: no data outcomes, no learner fits, no neural inference."""
from common import *


def main():
    checks = 0
    pre = np.array([-1.1, -0.0, 1e-6], dtype=np.float32)
    x = design_matrix(pre, np.array([1,0,1]), np.array([[0,0,0],[9,39,39],[2,12,20]]))
    assert x.shape == (3,92) and np.array_equal(x[:,0],pre.astype(np.float64)); checks += 1
    assert np.array_equal(x[1, [1,11,51,91]], [0,1,1,1]); checks += 1
    assert metrics(np.array([1,1,0,0]), np.array([1,0,1,0]))["balanced_accuracy"] == .5; checks += 1
    assert metrics(np.array([0,0]), np.array([0,0]))["balanced_accuracy"] is None; checks += 1
    assert metrics(np.array([1,0]), np.array([np.nan,0]))["balanced_accuracy"] is None; checks += 1
    rows = [dict(assignment="A"),dict(assignment="A"),dict(assignment="B")]
    assert assignment_macro([1,0,.8], rows)["macro"] == .65; checks += 1
    assert assignment_macro([1,None,.8], rows)["macro"] is None; checks += 1
    assert interval(np.r_[np.zeros(1899),np.full(101,np.nan)])["interval_2_5_97_5"] is None; checks += 1
    assert interval(np.r_[np.zeros(1900),np.full(100,np.nan)])["interval_2_5_97_5"] == [0,0]; checks += 1
    assert interpretation(.03,[.021,.04])["category"] == "ONE_BIT_LOSES_MATERIAL_ACCURACY"; checks += 1
    assert interpretation(0,[-.02,.03])["category"] == "CONTINUOUS_MEETS_DEVELOPMENT_NONINFERIORITY"; checks += 1
    assert interpretation(-.03,[-.05,-.021])["category"] == "THIS_COMPARATOR_MISSES_DECLARED_PARITY"; checks += 1
    assert interpretation(0,[-.03,.03])["category"] == "UNRESOLVED_OR_UNSUPPORTED"; checks += 1
    assert interpretation(0,[-.02,.02])["descriptive_parity"]; checks += 1
    assert np.array_equal(np.array([.5,np.nextafter(.5,1.)]) > .5, [False,True]); checks += 1
    write_json(ROOT / "selfcheck.json", dict(status="PASS", checks=checks, data="synthetic only", fits=0, neural_forwards=0))
    print(json.dumps(dict(status="PASS", checks=checks, fits=0)))


if __name__ == "__main__": main()
