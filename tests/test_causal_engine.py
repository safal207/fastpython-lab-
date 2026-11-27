import random
import unittest
from alexpy.causal_engine import Graph, Node, CausalEngine, MODES


def oracle(graph, values):
    """Independent full rebuild; does not use engine evaluator or cone."""
    out = list(values)
    for node in graph.nodes:
        args = [out[p] for p in node.parents]
        if node.op == 'sum':
            result = 0
            for a in args:
                result += a
        elif node.op == 'max':
            result = sorted(args)[-1]
        elif node.op == 'square':
            result = pow(args[0], 2)
        else:
            result = 1 if args[0] >= 1 else 0
        out.append(result)
    return out


def check_transition(test, graph, engine, old, new, event):
    demanded = set(engine.live)
    if engine.mode in ('slice_full', 'slice_dirty', 'causal'):
        needed = set(graph.outputs)
        for i in range(graph.inputs+len(graph.nodes)-1, graph.inputs-1, -1):
            if i in needed:
                needed.update(graph.nodes[i-graph.inputs].parents)
        test.assertEqual(demanded, {i for i in needed if i >= graph.inputs})
    else:
        test.assertEqual(demanded, set(range(graph.inputs, graph.inputs+len(graph.nodes))))
    actual = set(event.evaluated)
    test.assertEqual(len(actual), len(event.evaluated))
    test.assertTrue(actual <= demanded)
    test.assertEqual(tuple(sorted(i for i in actual if old[i] == new[i])), event.equal_boundary)
    test.assertEqual(tuple(sorted(i for i in set(range(graph.inputs)) | demanded if old[i] != new[i])), event.changed)
    for i in demanded - actual:
        test.assertEqual(old[i], new[i])
        test.assertTrue(all(old[p] == new[p] for p in graph.nodes[i-graph.inputs].parents))
    test.assertEqual(engine.read(), tuple(new[i] for i in graph.outputs))
    test.assertTrue(all(engine._values[i] == new[i] for i in demanded | set(range(graph.inputs))))


class CausalTests(unittest.TestCase):
    def setUp(self):
        self.g = Graph(3, (Node('sum',(0,1)), Node('positive',(3,)), Node('square',(4,)), Node('square',(2,))), (5,))

    def test_exact_boundary_and_dead_branch(self):
        e = CausalEngine(self.g, [4,5,7])
        self.assertEqual(e.live, (3,4,5))
        self.assertEqual(e.apply({2:99}).evaluated, ())
        event = e.apply({0:5,1:4})
        self.assertEqual(event.evaluated, (3,))
        self.assertEqual(event.equal_boundary, (3,))
        self.assertEqual(e.read(), (1,))
        self.assertEqual(e.apply({0:-9}).evaluated, (3,4,5))
        self.assertEqual(e.read(), (0,))

    def test_invalid_is_atomic(self):
        e = CausalEngine(self.g, [1,2,3])
        for patch in ({0:99,2:True}, {0:99,3:1}, {False:1}, []):
            with self.assertRaises((IndexError,TypeError)):
                e.apply(patch)
            self.assertEqual(e.read(), (1,))
            self.assertEqual((e.tick,e.version),(0,0))
        self.assertEqual(e.apply({0:1}).changed, ())
        self.assertEqual((e.tick,e.version),(1,0))

    def test_graph_validation(self):
        for node in (Node('sum',(2,)),Node('square',(0,1)),Node('sum',(True,)),Node('random',(0,)),Node('sum',(0,0))):
            with self.assertRaises(ValueError):
                Graph(2,(node,),(2,))
        with self.assertRaises(ValueError):
            Graph(1,(),(1,))

    def test_input_output_and_alias(self):
        values = [10,20]
        e = CausalEngine(Graph(2,(),(1,0)),values)
        values[0] = 100
        self.assertEqual(e.read(), (20,10))
        e.apply({0:30})
        self.assertEqual(e.read(), (20,30))

    def test_long_chain_no_recursion(self):
        g = Graph(1,tuple(Node('positive',(i,)) for i in range(5000)),(5000,))
        e = CausalEngine(g,[1])
        self.assertEqual(e.apply({0:2}).evaluated,(1,))
        self.assertEqual(len(e.apply({0:0}).evaluated),5000)
        self.assertEqual(e.read(),(0,))

    def test_random_dags_all_modes_and_certificates(self):
        rng = random.Random(91387)
        for _ in range(15):
            nodes=[]
            for i in range(8,88):
                op=rng.choice(('sum','max','positive','square'))
                # Avoid unbounded repeated squaring while still testing big integers.
                parents=(rng.randrange(8),) if op=='square' else tuple(rng.sample(range(i),1 if op=='positive' else rng.randint(1,4)))
                nodes.append(Node(op,parents))
            g=Graph(8,tuple(nodes),tuple(rng.sample(range(88),4)))
            values=[rng.randrange(-5,6) for _ in range(8)]
            engines=[CausalEngine(g,values,m) for m in MODES]
            for _ in range(40):
                before=oracle(g,values)
                patch={i:rng.randrange(-10,11) for i in rng.sample(range(8),rng.randrange(9))}
                for i,v in patch.items(): values[i]=v
                after=oracle(g,values)
                for e in engines:
                    check_transition(self,g,e,before,after,e.apply(patch))

    def test_bigint_and_reconvergence(self):
        g=Graph(2,(Node('sum',(0,1)),Node('square',(0,)),Node('max',(2,3))), (4,2))
        e=CausalEngine(g,[2**200,-2**200])
        e.apply({0:-2**200,1:2**200})
        self.assertEqual(e.read(),(2**400,0))


if __name__ == '__main__':
    unittest.main()
