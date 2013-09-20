import ast
import unittest

import mnfy


class FunctionToLambdaTests(unittest.TestCase):

    def setUp(self):
        self.transform = mnfy.FunctionToLambda()

    def _test_failure(self, fxn_code):
        fxn_ast = ast.parse(fxn_code)
        new_ast = self.transform.visit(fxn_ast)
        new_fxn = new_ast.body[0]
        self.assertIsInstance(new_fxn, ast.FunctionDef,
            '{} not an instance of ast.FunctionDef'.format(new_ast.__class__))

    def test_decorator_fail(self):
        self._test_failure('@dec\ndef X(): return')

    def test_returns_annotation_fail(self):
        self._test_failure('def X()->None: return')

    def test_body_too_long_fail(self):
        self._test_failure('def X(): x = 2 + 3; return x')

    def test_body_not_return_fail(self):
        self._test_failure('def X(): Y()')

    def test_no_vararg_annotation_fail(self):
        self._test_failure('def X(*arg:None): return')

    def test_no_kwarg_annotation_fail(self):
        self._test_failure('def X(**kwargs:None): return')

    def test_no_arg_annotation_fail(self):
        self._test_failure('def X(a, b:None, c): return')

    def test_success(self):
        module = ast.parse('def identity(): return 42')
        fxn = module.body[0]
        new_ast = self.transform.visit(module)
        assign = new_ast.body[0]
        self.assertIsInstance(assign, ast.Assign)
        self.assertEqual(len(assign.targets), 1)
        target = assign.targets[0]
        self.assertIsInstance(target, ast.Name)
        self.assertEqual(target.id, 'identity')
        self.assertIsInstance(target.ctx, ast.Store)
        lmda = assign.value
        self.assertIsInstance(lmda, ast.Lambda)
        self.assertIs(lmda.args, fxn.args)
        self.assertIs(lmda.body, fxn.body[0].value)

    def test_return_None(self):
        # If a function has a bare return then the lambda should return None.
        module = ast.parse('def fxn(): return')
        new_ast = self.transform.visit(module)
        lambda_ = new_ast.body[0].value
        self.assertIsInstance(lambda_.body, ast.Name)
        self.assertEqual(lambda_.body.id, 'None')
        self.assertIsInstance(lambda_.body.ctx, ast.Load)

    unittest.skip('not implemented')
    def test_empty_return(self):
        pass


if __name__ == '__main__':
    unittest.main()
