import ast
import unittest

import mnfy


class TransformTest(unittest.TestCase):

    """Base class for assisting in testing AST transformations."""

    def check_transform(self, input_, expect):
        result = self.transform.visit(input_)
        if expect is None:
            if result is not None:
                self.fail('{} is not None'.formatast.dump(result, False))
        else:
            self.assertEqual(ast.dump(result, False), ast.dump(expect, False))


class CombineImportsTests(unittest.TestCase):

    def setUp(self):
        self.transform = mnfy.CombineImports()

    def test_one_Import(self):
        # Nothing should happen.
        imp = ast.Import(ast.alias('X', None))
        module = ast.Module([imp])
        new_ast = self.transform.visit(module)
        self.assertEqual(len(new_ast.body), 1)

    def test_combining_Import(self):
        # Should lead to a single Import with all of the aliases.
        to_import = ('X', None), ('Y', None)
        imports = []
        for alias in to_import:
            imports.append(ast.Import([ast.alias(*alias)]))
        module = ast.Module(imports)
        new_ast = self.transform.visit(module)
        self.assertEqual(len(new_ast.body), 1)
        new_imp = new_ast.body[0]
        self.assertIsInstance(new_imp, ast.Import)
        self.assertEqual(len(new_imp.names), 2)
        for index, (name, alias) in enumerate(to_import):
            self.assertEqual(new_imp.names[index].name, name)
            self.assertEqual(new_imp.names[index].asname, alias)

    def test_interleaved_statements(self):
        # Do not combine if something between the Import statements.
        imp1 = ast.Import([ast.alias('X', None)])
        imp2 = ast.Import([ast.alias('Y', None)])
        from_import = ast.ImportFrom('Z', [ast.alias('W', None)], 0)
        module = ast.Module([imp1, from_import, imp2])
        new_ast = self.transform.visit(module)
        self.assertEqual(len(new_ast.body), 3)
        for given, expect in zip(new_ast.body,
                (ast.Import, ast.ImportFrom, ast.Import)):
            self.assertIsInstance(given, expect)
        last_imp = new_ast.body[2]
        self.assertEqual(len(last_imp.names), 1)
        self.assertEqual(last_imp.names[0].name, 'Y')

    def test_one_ImportFrom(self):
        # Nothing changed.
        imp = ast.ImportFrom('X', [ast.alias('Y', None)], 0)
        module = ast.Module([imp])
        new_ast = self.transform.visit(module)
        self.assertEqual(len(new_ast.body), 1)

    def test_combining_ImportFrom(self):
        # Combine ImportFrom when the 'from' clause matches.
        imp1 = ast.ImportFrom('X', [ast.alias('Y', None)], 1)
        imp2 = ast.ImportFrom('X', [ast.alias('Z', None)], 1)
        module = ast.Module([imp1, imp2])
        new_ast = self.transform.visit(module)
        self.assertEqual(len(module.body), 1)
        imp = new_ast.body[0]
        self.assertEqual(len(imp.names), 2)
        for alias, (name, asname) in zip(imp.names,
                (('Y', None), ('Z', None))):
            self.assertEqual(alias.name, name)
            self.assertEqual(alias.asname, asname)

    def test_interleaved_ImportFrom(self):
        # Test prevention of statement merging.
        import_from1 = ast.ImportFrom('X', [ast.alias('Y', None)], 1)
        imp = ast.Import([ast.alias('X', None)])
        # Separate by Import
        import_from2 = ast.ImportFrom('X', [ast.alias('Z', None)], 1)
        # Different level
        import_from3 = ast.ImportFrom('X', [ast.alias('W', None)], 2)
        # Different 'from' clause
        import_from4 = ast.ImportFrom('Z', [ast.alias('Y', None)], 2)
        module = ast.Module([import_from1, imp, import_from2, import_from3,
                             import_from4])
        new_ast = self.transform.visit(module)
        self.assertEqual(len(module.body), 5)


class CombineWithStatementsTests(unittest.TestCase):

    """with A:
         with B:
            pass

       with A,B:pass
    """

    A = ast.Name('A', ast.Load())
    A_clause = ast.withitem(A, None)
    B = ast.Name('B', ast.Load())
    B_clause = ast.withitem(B, None)
    C = ast.Name('C', ast.Load())
    C_clause = ast.withitem(C, None)

    def setUp(self):
        self.transform = mnfy.CombineWithStatements()

    def test_deeply_nested(self):
        with_C = ast.With([self.C_clause], [ast.Pass()])
        with_B = ast.With([self.B_clause], [with_C])
        with_A = ast.With([self.A_clause], [with_B])
        new_ast = self.transform.visit(with_A)
        expect = ast.With([self.A_clause, self.B_clause, self.C_clause],
                          [ast.Pass()])
        self.assertEqual(ast.dump(new_ast), ast.dump(expect))

    def test_no_optimization(self):
        with_B = ast.With([self.B_clause], [ast.Pass()])
        with_A = ast.With([self.A_clause], [with_B, ast.Pass()])
        new_ast = self.transform.visit(with_A)
        self.assertEqual(new_ast, with_A)


class UnusedConstantEliminationTests(TransformTest):

    def setUp(self):
        self.transform = mnfy.EliminateUnusedConstants()

    def test_do_no_over_step_bounds(self):
        assign = ast.Assign([ast.Name('a', ast.Store())], ast.Num(42))
        self.check_transform(assign, assign)

    def test_unused_constants(self):
        number = ast.Num(1)
        string = ast.Str('A')
        bytes_ = ast.Bytes(b'A')
        module = ast.Module([ast.Expr(expr)
                                for expr in (number, string, bytes_)])
        new_ast = self.transform.visit(module)
        self.assertFalse(new_ast.body)

    def _test_empty_body(self, node):
        node.body.append(ast.Expr(ast.Str('dead code')))
        module = ast.Module([node])
        new_ast = self.transform.visit(module)
        self.assertEqual(len(new_ast.body), 1)
        block = new_ast.body[0].body
        self.assertEqual(len(block), 1)
        self.assertIsInstance(block[0], ast.Pass)
        return new_ast

    def test_empty_FunctionDef(self):
        function = ast.FunctionDef('X', ast.arguments(), [], [], None)
        self._test_empty_body(function)

    def test_empty_ClassDef(self):
        cls = ast.ClassDef('X', [], [], None, None, [], None)
        self._test_empty_body(cls)

    def test_empty_For(self):
        for_ = ast.For(ast.Name('X', ast.Store()), ast.Str('a'), [], [])
        self._test_empty_body(for_)
        # An empty 'else' clause should just go away
        for_else = ast.For(ast.Name('X', ast.Store()), ast.Str('a'),
                           [ast.Pass()], [ast.Pass()])
        expect = ast.For(ast.Name('X', ast.Store()), ast.Str('a'),
                           [ast.Pass()], [])
        self.check_transform(for_else, expect)

    def test_empty_While(self):
        while_ = ast.While(ast.Num(42), [], [])
        self._test_empty_body(while_)
        # An empty 'else' clause should be eliminated.
        while_else = ast.While(ast.Num(42), [ast.Pass()],
                               [ast.Pass()])
        expect = ast.While(ast.Num(42), [ast.Pass()], [])
        self.check_transform(while_else, expect)

    def test_empty_If(self):
        if_ = ast.If(ast.Num(2), [], [])
        self._test_empty_body(if_)
        # An empty 'else' clause should be eliminated.
        if_else = ast.If(ast.Num(42), [ast.Pass()], [ast.Pass()])
        expect = ast.If(ast.Num(42), [ast.Pass()], [])
        self.check_transform(if_else, expect)

    def test_empty_With(self):
        with_ = ast.With([ast.Name('X', ast.Load())], [])
        self._test_empty_body(with_)

    def test_empty_Try(self):
        # try/except
        exc_clause = ast.ExceptHandler(None, None,
                                        [ast.Expr(ast.Str('dead code'))])
        try_exc = ast.Try([ast.Pass()], [exc_clause], [], [])
        expect = ast.Try([ast.Pass()],
                         [ast.ExceptHandler(None, None, [ast.Pass()])], [], [])
        self.check_transform(try_exc, expect)
        # try/finally should be eliminated
        try_finally = ast.Try([ast.Pass()], [], [],
                              [ast.Expr(ast.Str('dead code'))])
        self.check_transform(try_finally, None)
        # try/else should be eliminated
        try_else = ast.Try([ast.Pass()], [], [ast.Expr(ast.Str('dead_code'))],
                           [])
        self.check_transform(try_else, None)


class IntegerToPowerTests(TransformTest):

    """100000 -> 10**5"""

    def setUp(self):
        self.transform = mnfy.IntegerToPower()

    def verify(self, base, exponent):
        num = base**exponent
        node = ast.Num(num)
        expect = ast.BinOp(ast.Num(base), ast.Pow(), ast.Num(exponent))
        self.check_transform(node, expect)

    def test_conversion(self):
        for num in (5, 12):
            self.verify(10, num)
        for num in (17, 20, 40):
            self.verify(2, num)

    def test_left_alone(self):
        for num in (10**5+1, float(10**5)):
            self.check_transform(ast.Num(num), ast.Num(num))


if __name__ == '__main__':
    unittest.main()
