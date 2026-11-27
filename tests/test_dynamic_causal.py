import random
import unittest
from alexpy.causal_engine import Graph, Node
from alexpy.dynamic_causal import DynamicCausalEngine, ROUTES


def oracle(graph, values):
    out = list(values)
    for node in graph.nodes:
        a = [out[p] for p in node.parents]
        if node.op == 'sum': v = sum(a)
        elif node.op == 'max': v = sorted(a)[-1]
        elif node.op == 'square': v = pow(a[0], 2)
        else: v = int(a[0] >= 1)
        out.append(v)
    return out


def validate(engine, graph, values, outputs):
    expected = oracle(graph, values)
    cone = set(outputs)
    for i in reversed(range(graph.inputs, len(expected))):
        if i in cone: cone.update(graph.nodes[i-graph.inputs].parents)
    active = set(range(graph.inputs, len(expected))) if engine._mode == 'eager' else cone-set(range(graph.inputs))
    assert engine.live == active
    assert engine.outputs == outputs
    assert engine.read() == tuple(expected[i] for i in outputs)
    assert all(engine._values[i] == expected[i] for i in active)
    assert engine.cached_nodes <= len(graph.nodes)
    if engine._mode == 'active_only': assert engine.cached_nodes == len(active)
    if engine._mode in ('causal', 'lazy_dirty'):
        for i in active:
            assert engine._signatures[i] == tuple(expected[p] for p in graph.nodes[i-graph.inputs].parents)
        assert not engine._dirty.intersection(active)


class DynamicTests(unittest.TestCase):
    def setUp(self):
        self.g = Graph(2, (Node('positive',(0,)),Node('sum',(2,)),Node('square',(1,)),Node('sum',(3,4))), (3,))

    def test_return_after_dormant_change(self):
        for mode in ROUTES:
            e = DynamicCausalEngine(self.g,[1,2],mode)
            e.select((4,)); e.apply({0:-1}); e.select((3,5))
            validate(e,self.g,[-1,2],(3,5))
            self.assertEqual(e.read(),(0,4))

    def test_exact_aba_and_equal_boundary(self):
        e = DynamicCausalEngine(self.g,[1,2]); e.select((4,))
        e.apply({0:-1}); e.apply({0:1})
        t=e.select((3,)); self.assertEqual(t.evaluated,()); self.assertEqual(t.reused,(2,3))
        e.select((4,)); e.apply({0:9}); t=e.select((3,))
        self.assertEqual(t.evaluated,(2,)); self.assertEqual(t.equal_boundary,(2,)); self.assertEqual(t.reused,(3,))
        validate(e,self.g,[9,2],(3,))

    def test_multi_parent_reconvergence_and_order(self):
        for mode in ROUTES:
            e=DynamicCausalEngine(self.g,[1,2],mode)
            e.select((0,)); e.apply({0:-2,1:7}); e.select((5,3,1))
            validate(e,self.g,[-2,7],(5,3,1)); self.assertEqual(e.read(),(49,0,7))
            e.select((1,3,5)); self.assertEqual(e.read(),(7,0,49))

    def test_invalid_is_atomic_and_alias(self):
        for mode in ROUTES:
            values=[1,2];e=DynamicCausalEngine(self.g,values,mode);values[0]=-1
            before=(e._values.copy(),e._signatures.copy(),e._dirty.copy(),e.outputs,e.live,e.tick,e.version,e.demand_version)
            for p in ({0:99,1:True},{0:99,2:4},[]):
                with self.assertRaises((IndexError,TypeError)):e.apply(p)
            for o in ((),(True,),(6,),(3,3),[3]):
                with self.assertRaises(ValueError):e.select(o)
            self.assertEqual(before,(e._values,e._signatures,e._dirty,e.outputs,e.live,e.tick,e.version,e.demand_version))
            self.assertEqual(e.apply({0:1}).evaluated,());self.assertEqual((e.tick,e.version),(1,0))

    def test_deep_chain(self):
        g=Graph(1,tuple(Node('positive',(i,)) for i in range(5000)),(5000,))
        for mode in ROUTES:
            e=DynamicCausalEngine(g,[1],mode);e.select((0,));e.apply({0:0});e.select((5000,));self.assertEqual(e.read(),(0,))

    def test_bigint(self):
        e=DynamicCausalEngine(self.g,[1,2**200]);e.select((4,));e.apply({1:-2**200});e.select((5,));self.assertEqual(e.read(),(2**400+1,))

    def test_random_dags(self):
        rng=random.Random(920127)
        for _ in range(12):
            nodes=[]
            for i in range(6,66):
                op=rng.choice(('sum','positive','max','square'))
                parents=(rng.randrange(6),) if op=='square' else tuple(rng.sample(range(i),1 if op=='positive' else rng.randint(1,4)))
                nodes.append(Node(op,parents))
            g=Graph(6,tuple(nodes),(65,));v=[rng.randrange(-7,8) for _ in range(6)];o=g.outputs
            engines=[DynamicCausalEngine(g,v,m) for m in ROUTES]
            for step in range(100):
                if step%3:
                    p={i:rng.randrange(-7,8) for i in rng.sample(range(6),rng.randrange(7))}
                    for i,x in p.items():v[i]=x
                    for e in engines:e.apply(p);validate(e,g,v,o)
                else:
                    o=tuple(rng.sample(range(66),rng.randrange(1,6)))
                    for e in engines:
                        old=e._signatures.copy();t=e.select(o);validate(e,g,v,o)
                        self.assertTrue(all(old[i]==e._signatures[i] for i in t.reused))

if __name__=='__main__':unittest.main()
