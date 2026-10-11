"""Unit tests for tiny offline reference oracle; NOT live-agent tests."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.oracle import (
    FullWorld,
    InvalidAction,
    InvalidWorld,
    Node,
    ReplayResult,
    equation1,
    nondominated,
    replay,
)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.world = FullWorld([
            Node("root", None, 0, 0),
            Node("A", "root", 1, .4, "A observed"),
            Node("A1", "A", 2, .5),
            Node("B", "root", 3, .7),
            Node("B1", "B", 4, .9),
            Node("C", "root", 5, .3),
        ])

    def test_root_children_in_creation_order(self):
        r = replay(self.world, lambda p,rho: ["root"], .5, W=2, K2=5)
        self.assertEqual(r.revealed, ("root", "A", "B", "C"))
        self.assertEqual(r.n, 3)
        self.assertEqual(r.kstar, 5)
        self.assertEqual(r.stop_reason, "K2")

    def test_progressive_deepening(self):
        def policy(prefix, rho):
            if len(prefix.nodes) == 1: return ["root"]
            if "A1" not in {n.id for n in prefix.nodes}: return ["A"]
            return []
        result = replay(self.world, policy, .5, 2, 5)
        self.assertEqual(result.revealed, ("root", "A", "A1"))
        self.assertEqual(result.kstar, 2)
        self.assertEqual(result.stop_reason, "STOP_EMPTY_BATCH")

    def test_parallel_batch(self):
        def policy(prefix, rho):
            if len(prefix.nodes) == 1: return ["root"]
            if len(prefix.nodes) == 2: return ["root", "A"]
            return []
        result = replay(self.world, policy, .5, W=2, K2=6)
        self.assertEqual(result.revealed, ("root", "A", "A1", "B"))
        self.assertEqual(result.kstar, 2)
        self.assertEqual(result.n, 3)

    def test_illegal_internal_node(self):
        def policy(prefix, rho):
            if len(prefix.nodes) == 1: return ["root"]
            if len(prefix.nodes) == 2: return ["A"]
            return ["A"] # A now internal -> illegal
        with self.assertRaises(InvalidAction):
            replay(self.world, policy, .5, W=2, K2=4)

    def test_duplicate_root_same_batch_illegal(self):
        with self.assertRaises(InvalidAction):
            replay(self.world, lambda p,rho: ["root", "root"], .5, W=2, K2=5)

    def test_w_cap(self):
        with self.assertRaises(InvalidAction):
            replay(self.world, lambda p,rho: ["root", "A"], .5, W=1, K2=1)

    def test_hidden_node_illegal(self):
        with self.assertRaises(InvalidAction):
            replay(self.world, lambda p,rho: ["B1"], .5, W=2, K2=1)

    def test_no_future_score_in_prefix(self):
        def policy(prefix, rho):
            self.assertEqual([n.id for n in prefix.nodes], ["root"])
            self.assertEqual([n.score for n in prefix.nodes], [0])
            self.assertEqual(prefix.legal, ("root",))
            return []
        replay(self.world, policy, .5, W=2, K2=1)

    def test_repeat_is_deterministic(self):
        policy = lambda p,rho: ["root"]
        a = replay(self.world, policy, .5, 2, 9)
        b = replay(self.world, policy, .5, 2, 9)
        self.assertEqual(a,b)

    def test_full_reveal_early_stop(self):
        w = FullWorld([Node("root",None,0,0), Node("A","root",1,1.)])
        r = replay(w, lambda p,rho: ["root"], .5, 1, 8)
        self.assertEqual((r.n,r.kstar,r.stop_reason), (1,1,"FULLY_REVEALED"))

    def test_nonroot_multiple_children_rejected(self):
        with self.assertRaises(InvalidWorld):
            FullWorld([Node("root",None,0,0),Node("A","root",1,.1),
                       Node("A1","A",2,.2),Node("A2","A",3,.3)])

    def test_empty_world_is_not_scoreable(self):
        w = FullWorld([Node("root",None,0,0)])
        r = replay(w,lambda p,rho:["root"],.5,1,5)
        self.assertEqual((r.n,r.kstar,r.stop_reason),(0,0,"FULLY_REVEALED"))


class EquationTests(unittest.TestCase):
    def golden_results(self):
        points = [(2,.4),(4,.4),(5,.8),(8,.9),(9,.9)]
        rounds = [2,3,4,5,6]
        return [ReplayResult(rho,("root",),tuple(),k,n,a*100,"synthetic")
                for rho,(n,a),k in zip((0,.25,.5,.75,1),points,rounds)]

    def test_frontier(self):
        self.assertEqual(nondominated([(2,.4),(4,.4),(5,.8),(8,.9),(9,.9)]),
                         ((2,.4),(5,.8),(8,.9)))

    def test_eq1_golden(self):
        r = equation1(self.golden_results(),B=10,S=100,lambda_weight=.1)
        self.assertAlmostEqual(r.hv,.54,12)
        self.assertAlmostEqual(r.penalty,(1+.75+.8+.625+6/9)/5,12)
        self.assertAlmostEqual(r.value,.46316666666666667,12)

    def test_zero_call_penalty(self):
        samples=self.golden_results()
        samples[0]=replace(samples[0],n=0,kstar=0,best_score=0)
        r=equation1(samples,B=10,S=100,lambda_weight=.1)
        self.assertGreaterEqual(r.hv,0)
        self.assertAlmostEqual(r.penalty,(0+.75+.8+.625+6/9)/5)

    def test_invalid_world_score(self):
        for B,S in ((0,100),(10,0),(10,-1)):
            with self.assertRaises(ValueError):
                equation1(self.golden_results(),B,S,.1)

    def test_incorrect_budget_sweep(self):
        with self.assertRaises(ValueError):
            equation1(self.golden_results()[:4],B=10,S=100,lambda_weight=.1)

    def test_quality_monotonicity(self):
        samples=self.golden_results()
        a=equation1(samples,10,100,0)
        samples[0]=replace(samples[0],best_score=60.)
        b=equation1(samples,10,100,0)
        self.assertGreaterEqual(b.hv,a.hv)


if __name__ == "__main__":
    unittest.main()
