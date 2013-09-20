#!/usr/bin/env python3
import ast
import inspect
import sys
import unittest

import mnfy


class SourceCodeEmissionTests(unittest.TestCase):

    """Test the emission of source code from AST nodes."""

    operators = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
                    ast.Mod: '%', ast.Pow: '**', ast.LShift: '<<',
                    ast.RShift: '>>', ast.BitOr: '|', ast.BitXor: '^',
                    ast.BitAnd: '&', ast.FloorDiv: '//'}


    def verify(self, given_ast, expect):
        visitor = mnfy.SourceCode()
        visitor.visit(given_ast)
        self.assertEqual(str(visitor), expect)

    def test_version(self):
        # Make sure that the newest AST is supported.
        self.assertGreaterEqual(int(mnfy.__ast_version__),
                                int(ast.__version__))

    def test_pop(self):
        visitor = mnfy.SourceCode()
        visitor._write('X')
        with self.assertRaises(ValueError):
            visitor._pop('sadf')
        visitor._write('X')
        self.assertEqual(visitor._pop(), 'X')

    def test_Ellipsis(self):
        self.verify(ast.Ellipsis(), '...')

    def test_Num(self):
        for num in (42, -13, 1.79769313486231e+308, 1.3, 10**12+0.5):
            self.verify(ast.Num(num), repr(num))
        for num in (10**12, -10**12):
            self.verify(ast.Num(num), hex(num))

    def test_Str(self):
        for text in ('string', '\n', r'\n'):
            self.verify(ast.Str(text), repr(text))

    def test_Bytes(self):
        for thing in (b'1', b'123'):
            self.verify(ast.Bytes(thing), repr(thing))

    def test_Dict(self):
        self.verify(ast.Dict([], []), '{}')
        self.verify(ast.Dict([ast.Num(42)], [ast.Num(42)]), '{42:42}')
        self.verify(ast.Dict([ast.Num(2), ast.Num(3)], [ast.Num(4),
            ast.Num(6)]), '{2:4,3:6}')

    def test_Set(self):
        self.verify(ast.Set([ast.Num(42)]), '{42}')
        self.verify(ast.Set([ast.Num(2), ast.Num(3)]), '{2,3}')

    def test_Tuple(self):
        self.verify(ast.Tuple([], ast.Load()), '()')
        self.verify(ast.Tuple([ast.Num(42)], ast.Load()), '(42,)')
        self.verify(ast.Tuple([ast.Num(42), ast.Num(3)], ast.Load()), '(42,3)')

    def test_List(self):
        self.verify(ast.List([], ast.Load()), '[]')
        self.verify(ast.List([ast.Num(42), ast.Num(3)], ast.Load()), '[42,3]')

    def test_comprehension(self):
        comp = ast.comprehension(ast.Name('x', ast.Store()), ast.Name('y',
                                    ast.Load()), [])
        self.verify(comp, 'for x in y')
        comp.ifs = [ast.Num(2), ast.Num(3)]
        self.verify(comp, 'for x in y if 2 if 3')

    def seq_comp_test(self, node_type, ends):
        gen = ast.comprehension(ast.Name('x', ast.Store()), ast.Name('y',
            ast.Load()), [ast.Num(2)])
        listcomp = node_type(ast.Name('w', ast.Load()), [gen])
        self.verify(listcomp, '{}w for x in y if 2{}'.format(*ends))
        gen2 = ast.comprehension(ast.Name('a', ast.Store()), ast.Name('b',
            ast.Load()), [])
        listcomp.generators.append(gen2)
        self.verify(listcomp, '{}w for x in y if 2 for a in b{}'.format(*ends))
        return listcomp

    def test_ListComp(self):
        self.seq_comp_test(ast.ListComp, '[]')

    def test_SetComp(self):
        self.seq_comp_test(ast.SetComp, '{}')

    def test_GeneratorExp(self):
        self.seq_comp_test(ast.GeneratorExp, '()')

    def test_DictComp(self):
        gen = ast.comprehension(ast.Name('x', ast.Store()), ast.Name('y',
            ast.Load()), [ast.Num(2)])
        dictcomp = ast.DictComp(ast.Name('v', ast.Load()), ast.Name('w',
            ast.Load()), [gen])
        self.verify(dictcomp, '{v:w for x in y if 2}')
        gen2 = ast.comprehension(ast.Name('a', ast.Store()), ast.Name('b',
            ast.Load()), [])
        dictcomp.generators.append(gen2)
        self.verify(dictcomp, '{v:w for x in y if 2 for a in b}')

    def test_Name(self):
        self.verify(ast.Name('X', ast.Load()), 'X')

    def test_Attribute(self):
        self.verify(ast.Attribute(ast.Name('spam', ast.Load()), 'eggs',
                        ast.Load()),
                    'spam.eggs')
        self.verify(ast.Attribute(ast.Attribute(ast.Name('spam', ast.Load()),
                        'eggs', ast.Load()), 'bacon', ast.Load()),
                    'spam.eggs.bacon')
        self.verify(ast.Attribute(ast.BinOp(ast.Name('X', ast.Load()),
                        ast.Add(), ast.Num(2)), 'Y', ast.Load()),
                    '(X+2).Y')

    def test_Assign(self):
        self.verify(ast.Assign([ast.Name('X', ast.Store())], ast.Num(42)),
                    'X=42')
        multi_assign = ast.Assign([ast.Name('X', ast.Store()), ast.Name('Y',
            ast.Store())], ast.Num(42))
        self.verify(multi_assign, 'X=Y=42')
        self.verify(ast.Assign([ast.Tuple([ast.Name('X', ast.Store()),
                        ast.Name('Y', ast.Store())], ast.Store())],
                        ast.Name('Z', ast.Load())),
                    'X,Y=Z')
        self.verify(ast.Assign([ast.Name('X', ast.Store())],
            ast.Tuple([ast.Num(1), ast.Num(2)], ast.Load())), 'X=1,2')
        self.verify(ast.Assign([ast.Name('X', ast.Store())],
                                ast.Tuple([], ast.Load())),
                    'X=()')

    def test_Delete(self):
        self.verify(ast.Delete([ast.Name('X', ast.Del()), ast.Name('Y',
                                ast.Del())]),
                    'del X,Y')

    def test_Call(self):
        name = ast.Name('spam', ast.Load())
        args = ([ast.Num(42)], '42'), ([], None)
        keywords = ([ast.keyword('X', ast.Num(42))], 'X=42'), ([], None)
        starargs = (ast.Name('args', ast.Load()), '*args'), (None, None)
        kwargs = (ast.Name('kwargs', ast.Load()), '**kwargs'), (None, None)
        for arg in args:
            for keyword in keywords:
                for stararg in starargs:
                    for kwarg in kwargs:
                        node = ast.Call(name, arg[0], keyword[0], stararg[0],
                                        kwarg[0])
                        expect = 'spam({})'.format(','.join(x for x in
                                    (arg[1], keyword[1], stararg[1], kwarg[1])
                                    if x))
                        self.verify(node, expect)
        self.verify(ast.Call(name, [ast.Num(2), ast.Num(3)], [], None, None),
                    'spam(2,3)')
        self.verify(ast.Call(name, [],
                        [ast.keyword('X', ast.Num(0)),
                            ast.keyword('Y', ast.Num(1))],
                        None, None),
                    'spam(X=0,Y=1)')
        # A single genexp doesn't need parentheses.
        genexp = self.seq_comp_test(ast.GeneratorExp, '()')
        self.verify(ast.Call(name, [genexp], [], None, None),
                    'spam(w for x in y if 2 for a in b)')
        self.verify(ast.Call(name, [genexp, genexp], [], None, None),
                    'spam((w for x in y if 2 for a in b),'
                    '(w for x in y if 2 for a in b))')

    def test_Starred(self):
        self.verify(ast.Starred(ast.Name('X', ast.Store()), ast.Store()),
                    '*X')

    def test_UnaryOp(self):
        self.verify(ast.UnaryOp(ast.Invert(), ast.Num(42)), '~42')
        self.verify(ast.UnaryOp(ast.Not(), ast.Num(42)), 'not 42')
        self.verify(ast.UnaryOp(ast.UAdd(), ast.Num(42)), '+42')
        self.verify(ast.UnaryOp(ast.USub(), ast.Num(42)), '-42')
        precedence = ast.UnaryOp(ast.Not(), ast.BoolOp(ast.Or(), [ast.Num(n=1),
                                    ast.Num(2)]))
        self.verify(precedence, 'not (1 or 2)')

    def test_BinOp(self):
        for node, op in self.operators.items():
            self.verify(ast.BinOp(ast.Num(2), node(), ast.Num(3)),
                        '2{}3'.format(op))
        # 1 + 2 * 3 = BinOp(2 + BinOp(2 * 3))
        mult = ast.BinOp(ast.Num(2), ast.Mult(), ast.Num(3))
        expr = ast.BinOp(ast.Num(1), ast.Add(), mult)
        self.verify(expr, '1+2*3')
        # (1 + 2) * 3 = BinOp(BinOp(1 + 2) * 3)
        add = ast.BinOp(ast.Num(1), ast.Add(), ast.Num(2))
        expr = ast.BinOp(add, ast.Mult(), ast.Num(3))
        self.verify(expr, '(1+2)*3')
        # 2 * 3 + 1 = BinOp(BinOp(2 * 3) + 1)
        expr = ast.BinOp(mult, ast.Add(), ast.Num(1))
        self.verify(expr, '2*3+1')
        # 3 * (1 + 2) = BinOp(3 * BinOp(1 + 2))
        expr = ast.BinOp(ast.Num(3), ast.Mult(), add)
        self.verify(expr, '3*(1+2)')
        # 3 - (1 + 2) = BinOp(3 - (BinOp1 + 2))
        expr = ast.BinOp(ast.Num(3), ast.Sub(), add)
        self.verify(expr, '3-(1+2)')
        # Deal with Pow's "special" precedence compared to unary operators.
        self.verify(ast.BinOp(ast.Num(-1), ast.Pow(), ast.Num(2)), '(-1)**2')
        self.verify(ast.UnaryOp(ast.USub(), ast.BinOp(ast.Num(1), ast.Pow(),
            ast.Num(2))), '-1**2')
        self.verify(ast.BinOp(ast.Num(1), ast.Pow(), ast.UnaryOp(ast.USub(),
                    ast.Num(2))), '1**(-2)')

    def test_Compare(self):
        comparisons = {ast.Eq: '==', ast.NotEq: '!=', ast.Lt: '<',
                        ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=',
                        ast.Is: ' is ', ast.IsNot: ' is not ', ast.In: ' in ',
                        ast.NotIn: ' not in '}
        for ast_cls, syntax in comparisons.items():
            self.verify(ast.Compare(ast.Num(3), [ast_cls()], [ast.Num(2)]),
                        '3{}2'.format(syntax))
        # 2 < 3 < 4
        three_way = ast.Compare(ast.Num(2), [ast.Lt(), ast.Lt()], [ast.Num(3),
            ast.Num(4)])
        self.verify(three_way, '2<3<4')
        # (2 < 3) < 4
        simple = ast.Compare(ast.Num(2), [ast.Lt()], [ast.Num(3)])
        left_heavy = ast.Compare(simple, [ast.Lt()], [ast.Num(4)])
        self.verify(left_heavy, '(2<3)<4')
        # 2 < (3 < 4)
        right_heavy = ast.Compare(ast.Num(2), [ast.Lt()], [simple])
        self.verify(right_heavy, '2<(2<3)')

    def test_Slice(self):
        self.verify(ast.Slice(None, None, None), ':')
        self.verify(ast.Slice(ast.Num(42), None, None), '42:')
        self.verify(ast.Slice(None, ast.Num(42), None), ':42')
        self.verify(ast.Slice(None, None, ast.Num(42)), '::42')
        self.verify(ast.Slice(ast.Num(1), ast.Num(2), None), '1:2')
        self.verify(ast.Slice(ast.Num(1), None, ast.Num(2)), '1::2')

    def test_ExtSlice(self):
        slice1 = ast.Index(ast.Num(42))
        slice2 = ast.Slice(None, None, ast.Num(6))
        self.verify(ast.ExtSlice([slice1, slice2]), '42,::6')

    def test_Subscript(self):
        sub = ast.Subscript(ast.Name('X', ast.Load()), [], ast.Load())
        # Index
        slice1 = ast.Index(ast.Num(42))
        sub.slice = slice1
        self.verify(sub, 'X[42]')
        # Slice
        slice2 = ast.Slice(None, None, ast.Num(2))
        sub.slice = slice2
        self.verify(sub, 'X[::2]')
        # ExtSlice
        sub.slice = ast.ExtSlice([slice1, slice2])
        self.verify(sub, 'X[42,::2]')

    def create_arguments(self, args=[], vararg=None, varargannotation=None,
            kwonlyargs=[], kwarg=None, kwargannotation=None, defaults=[],
            kw_defaults=[]):
        args = [ast.arg(x, None) for x in args]
        kwonlyargs = [ast.arg(x, None) for x in kwonlyargs]
        return ast.arguments(args, vararg, varargannotation, kwonlyargs,
                                kwarg, kwargannotation, defaults, kw_defaults)

    def test_arg(self):
        self.verify(ast.arg('spam', None), 'spam')
        self.verify(ast.arg('spam', ast.Num(42)), 'spam:42')

    def test_arguments(self):
        self.verify(self.create_arguments(args=['spam']), 'spam')
        self.verify(self.create_arguments(args=['spam', 'eggs']),
                    'spam,eggs')
        self.verify(self.create_arguments(args=['spam'],
                        defaults=[ast.Num(42)]),
                    'spam=42')
        self.verify(self.create_arguments(args=['spam', 'eggs'],
                        defaults=[ast.Num(3), ast.Num(2)]),
                    'spam=3,eggs=2')
        self.verify(self.create_arguments(args=['spam', 'eggs'],
                        defaults=[ast.Num(42)]),
                    'spam,eggs=42')
        self.verify(self.create_arguments(vararg='spam'), '*spam')
        self.verify(self.create_arguments(vararg='spam',
                        varargannotation=ast.Num(42)),
                    '*spam:42')
        self.verify(self.create_arguments(kwonlyargs=['spam']),
                    '*,spam')
        self.verify(self.create_arguments(kwonlyargs=['spam'],
                        kw_defaults=[ast.Num(42)]),
                    '*,spam=42')
        self.verify(self.create_arguments(args=['spam'], kwonlyargs=['eggs']),
                    'spam,*,eggs')
        self.verify(self.create_arguments(vararg='spam', kwonlyargs=['eggs']),
                    '*spam,eggs')
        self.verify(self.create_arguments(kwarg='spam'), '**spam')
        self.verify(self.create_arguments(args=['spam'], vararg='eggs'),
                    'spam,*eggs')
        self.verify(self.create_arguments(args=['spam'], kwarg='eggs'),
                    'spam,**eggs')

    def test_Lambda(self):
        self.verify(ast.Lambda(self.create_arguments(), ast.Num(42)),
                    'lambda:42')
        self.verify(ast.Lambda(self.create_arguments(['spam']), ast.Num(42)),
                    'lambda spam:42')

    def test_Pass(self):
        self.verify(ast.Pass(), 'pass')

    def test_Break(self):
        self.verify(ast.Break(), 'break')

    def test_Continue(self):
        self.verify(ast.Continue(), 'continue')

    def test_Raise(self):
        self.verify(ast.Raise(None, None), 'raise')
        self.verify(ast.Raise(ast.Name('X', ast.Load()), None), 'raise X')
        raise_ast = ast.Raise(ast.Name('X', ast.Load()),
                                ast.Name('Y', ast.Load()))
        self.verify(raise_ast, 'raise X from Y')

    def test_Return(self):
        self.verify(ast.Return(None), 'return')
        self.verify(ast.Return(ast.Num(42)), 'return 42')
        self.verify(ast.Return(ast.Tuple([ast.Num(1), ast.Num(2)], ast.Load())),
                    'return 1,2')

    def test_Yield(self):
        self.verify(ast.Yield(None), 'yield')
        self.verify(ast.Yield(ast.Num(42)), 'yield 42')
        self.verify(ast.Yield(ast.Tuple([ast.Num(1), ast.Num(2)], ast.Load())),
                    'yield 1,2')

    def test_Import(self):
        self.verify(ast.Import([ast.alias('spam', None)]), 'import spam')
        self.verify(ast.Import([ast.alias('spam', 'bacon')]),
                    'import spam as bacon')
        self.verify(ast.Import([ast.alias('spam', None),
                                ast.alias('bacon', 'bacn'),
                                ast.alias('eggs', None)]),
                    'import spam,bacon as bacn,eggs')

    def test_ImportFrom(self):
        # from X import Y
        from_X = ast.ImportFrom('X', [ast.alias('Y', None)], 0)
        self.verify(from_X, 'from X import Y')
        # from . import Y
        from_dot = ast.ImportFrom(None, [ast.alias('Y', None)], 1)
        self.verify(from_dot, 'from . import Y')
        # from .X import Y
        from_dot_X = ast.ImportFrom('X', [ast.alias('Y', None)], 1)
        self.verify(from_dot_X, 'from .X import Y')
        # from X import Y, Z
        from_X_multi = ast.ImportFrom('X', [ast.alias('Y', None),
                                        ast.alias('Z', None)], 0)
        self.verify(from_X_multi, 'from X import Y,Z')

    def test_BoolOp(self):
        and_op = ast.BoolOp(ast.And(), [ast.Num(2), ast.Num(3)])
        self.verify(and_op, '2 and 3')
        or_op = ast.BoolOp(ast.Or(), [ast.Num(2), ast.Num(3)])
        self.verify(or_op, '2 or 3')
        many_args = ast.BoolOp(ast.And(), [ast.Num(1), ast.Num(2), ast.Num(3)])
        self.verify(many_args, '1 and 2 and 3')
        no_precedence = ast.BoolOp(ast.Or(), [ast.BoolOp(ast.And(),
            [ast.Num(2), ast.Num(3)]), ast.Num(1)])
        self.verify(no_precedence, '2 and 3 or 1')
        no_precedence2 = ast.BoolOp(ast.Or(), [ast.Num(2), ast.BoolOp(ast.And(),
                                        [ast.Num(3), ast.Num(1)])])
        self.verify(no_precedence2, '2 or 3 and 1')
        precedence = ast.BoolOp(ast.And(), [ast.Num(1), ast.BoolOp(ast.Or(),
                                [ast.Num(2), ast.Num(3)])])
        self.verify(precedence, '1 and (2 or 3)')

    def test_IfExp(self):
        if_expr = ast.IfExp(ast.Num(1), ast.Num(2), ast.Num(3))
        self.verify(if_expr, '2 if 1 else 3')

    def test_If(self):
        # 'if' only
        if_ = ast.If(ast.Num(42), [ast.Pass()], [])
        self.verify(if_, 'if 42:pass')
        # if/else
        if_else = ast.If(ast.Num(42), [ast.Pass()], [ast.Pass()])
        self.verify(if_else, 'if 42:pass\nelse:pass')
        # if/elif/else
        if_elif_else = ast.If(ast.Num(6), [ast.Pass()], [if_else])
        self.verify(if_elif_else, 'if 6:pass\n'
                                  'elif 42:pass\n'
                                  'else:pass')
        # if/else w/ a leading 'if' clause + extra
        if_else_extra = ast.If(ast.Num(6), [ast.Pass()], [if_, ast.Pass()])
        self.verify(if_else_extra, 'if 6:pass\n'
                                   'else:\n'
                                   ' if 42:pass\n'
                                   ' pass')
        # 'if' w/ a leading simple stmt but another non-simple stmt
        if_fancy_body = ast.If(ast.Num(6), [ast.Pass(), if_], [])
        self.verify(if_fancy_body, 'if 6:\n pass\n if 42:pass')

    def test_For(self):
        for_ = ast.For(ast.Name('target', ast.Load()),
                        ast.Name('iter_', ast.Load()), [ast.Pass()], [])
        self.verify(for_, 'for target in iter_:pass')
        for_.orelse = [ast.Pass()]
        self.verify(for_, 'for target in iter_:pass\nelse:pass')
        for_.target = ast.Tuple([ast.Name('X', ast.Store()), ast.Name('Y',
                                    ast.Store())], ast.Store())
        for_.orelse = []
        self.verify(for_, 'for X,Y in iter_:pass')

    def test_While(self):
        while_ = ast.While(ast.Name('True', ast.Load()), [ast.Pass()], None)
        self.verify(while_, 'while True:pass')
        while_.orelse = [ast.Pass()]
        self.verify(while_, 'while True:pass\nelse:pass')

    def test_With(self):
        with_ = ast.With(ast.Name('X', ast.Load()), [], [ast.Pass()])
        self.verify(with_, 'with X:pass')
        with_.optional_vars = ast.Name('Y', ast.Store())
        self.verify(with_, 'with X as Y:pass')
        nested_with = ast.With(ast.Name('X', ast.Load()),
                                ast.Name('x', ast.Store()),
                                [ast.With(ast.Name('Y', ast.Load()),
                                    ast.Name('y', ast.Store()),
                                    [ast.Pass()])])
        self.verify(nested_with, 'with X as x,Y as y:pass')

    def test_ExceptHandler(self):
        except_ = ast.ExceptHandler(None, None, [ast.Pass()])
        self.verify(except_, 'except:pass')
        except_.type = ast.Name('Exception', ast.Load())
        self.verify(except_, 'except Exception:pass')
        except_.name = ast.Name('exc', ast.Store())
        self.verify(except_, 'except Exception as exc:pass')

    def test_TryExcept(self):
        exc_clause = ast.ExceptHandler(ast.Name('X', ast.Load()), None,
                                        [ast.Pass()])
        exc_clause_2 = ast.ExceptHandler(None, None, [ast.Pass()])
        try_ = ast.TryExcept([ast.Pass()], [exc_clause, exc_clause_2], None)
        self.verify(try_, 'try:pass\nexcept X:pass\nexcept:pass')
        try_.orelse = [ast.Pass()]
        self.verify(try_, 'try:pass\nexcept X:pass\nexcept:pass\nelse:pass')

    def test_TryFinally(self):
        try_finally = ast.TryFinally([ast.Pass()], [ast.Pass()])
        self.verify(try_finally, 'try:pass\nfinally:pass')
        exc_clause = ast.ExceptHandler(None, None, [ast.Pass()])
        try_except = ast.TryExcept([ast.Pass()], [exc_clause], None)
        try_finally.body = [try_except]
        self.verify(try_finally, 'try:pass\nexcept:pass\nfinally:pass')
        try_finally.body = [try_except, ast.Raise(None, None)]
        expect = 'try:\n try:pass\n except:pass\n raise\nfinally:pass'
        self.verify(try_finally, expect)

    def test_AugAssign(self):
        for cls, op in self.operators.items():
            aug_assign = ast.AugAssign(ast.Name('X', ast.Store()), cls(),
                    ast.Num(1))
            self.verify(aug_assign, 'X{}=1'.format(op))
        self.verify(ast.AugAssign(ast.Name('X', ast.Store()), ast.Add(),
                        ast.Tuple([ast.Num(1), ast.Num(2)], ast.Load())),
                    'X+=1,2')

    def test_Assert(self):
        assert_ = ast.Assert(ast.Num(42), None)
        self.verify(assert_, 'assert 42')
        assert_msg = ast.Assert(ast.Num(42), ast.Num(6))
        self.verify(assert_msg, 'assert 42,6')

    def test_Global(self):
        glbl = ast.Global(['x'])
        self.verify(glbl, 'global x')
        many_glbl = ast.Global(['x', 'y'])
        self.verify(many_glbl, 'global x,y')

    def test_Nonlocal(self):
        nonlocal_ = ast.Nonlocal(['X'])
        self.verify(nonlocal_, 'nonlocal X')
        many_nonlocal = ast.Nonlocal(['X', 'Y'])
        self.verify(many_nonlocal, 'nonlocal X,Y')

    def test_FunctionDef(self):
        # Arguments
        with_args = ast.FunctionDef('X', self.create_arguments(args='Y'),
                                    [ast.Pass()], [], None)
        self.verify(with_args, 'def X(Y):pass')
        # Decorators
        decorated = ast.FunctionDef('X', self.create_arguments(), [ast.Pass()],
                [ast.Name('dec1', ast.Load()), ast.Name('dec2', ast.Load()),
                    ast.Name('dec3', ast.Load())],
                None)
        self.verify(decorated, '@dec1\n@dec2\n@dec3\ndef X():pass')
        # Return annotation
        annotated = ast.FunctionDef('X', self.create_arguments(), [ast.Pass()],
                [], ast.Num(42))
        self.verify(annotated, 'def X()->42:pass')

    def test_ClassDef(self):
        # class X: pass
        cls = ast.ClassDef('X', [], [], None, None, [ast.Pass()], [])
        self.verify(cls, 'class X:pass')
        # class X(Y): pass
        cls = ast.ClassDef('X', [ast.Name('Y', ast.Load())], [], None, None,
                [ast.Pass()], [])
        self.verify(cls, 'class X(Y):pass')
        # class X(Y=42): pass
        cls = ast.ClassDef('X', [], [ast.keyword('Y', ast.Num(42))], None,
                None, [ast.Pass()], [])
        self.verify(cls, 'class X(Y=42):pass')
        # class X(Z, Y=42): pass
        cls.bases.append(ast.Name('Z', ast.Load()))
        self.verify(cls, 'class X(Z,Y=42):pass')
        # class X(*args): pass
        cls = ast.ClassDef('X', [], [], ast.Name('args', ast.Load()), None,
                [ast.Pass()], [])
        self.verify(cls, 'class X(*args):pass')
        # class X(Y, *args): pass
        cls.bases.append(ast.Name('Y', ast.Load()))
        self.verify(cls, 'class X(Y,*args):pass')
        # class X(**kwargs): pass
        cls = ast.ClassDef('X', [], [], None, ast.Name('kwargs', ast.Load()),
                [ast.Pass()], [])
        self.verify(cls, 'class X(**kwargs):pass')
        # class X(Y, **kwargs): pass
        cls.bases.append(ast.Name('Y', ast.Load()))
        self.verify(cls, 'class X(Y,**kwargs):pass')
        # Decorators
        cls = ast.ClassDef('X', [], [], None, None, [ast.Pass()],
                            [ast.Name('dec1', ast.Load()),
                                ast.Name('dec2', ast.Load())])
        self.verify(cls, '@dec1\n@dec2\nclass X:pass')

    def test_simple_statements(self):
        # Simple statements can be put on a single line as long as the scope
        # has not changed.
        for body, expect in [(ast.Expr(ast.Num(42)), '42'),
                    (ast.Import([ast.alias('a', None)]), 'import a'),
                    (ast.ImportFrom('b', [ast.alias('a', None)], 1),
                        'from .b import a'),
                    (ast.Break(), 'break'),
                    (ast.Continue(), 'continue'),
                    (ast.Pass(), 'pass'),
                    (ast.Assign([ast.Name('X', ast.Store())], ast.Num(42)),
                        'X=42'),
                    (ast.Delete([ast.Name('X', ast.Del())]), 'del X'),
                    (ast.Raise(None, None), 'raise'),
                    (ast.Return(None), 'return'),
                    (ast.AugAssign(ast.Name('X', ast.Store()), ast.Add(),
                        ast.Num(42)), 'X+=42'),
                    (ast.Assert(ast.Num(42), None), 'assert 42'),
                    (ast.Global(['x']), 'global x'),
                    (ast.Nonlocal(['x']), 'nonlocal x'),
                ]:
            if_simple = ast.If(ast.Num(42), [body], None)
            self.verify(if_simple, 'if 42:{}'.format(expect))

        if_multiple_simples = ast.If(ast.Num(42), [ast.Pass(), ast.Pass()],
                                        None)
        self.verify(if_multiple_simples, 'if 42:pass;pass')
        inner_if = ast.If(ast.Num(6), [ast.Pass()], None)
        funky_if = ast.If(ast.Num(42), [ast.Break(), ast.Continue(), inner_if,
                                        ast.Break(), ast.Continue()],
                            None)
        self.verify(funky_if,
                    'if 42:\n break;continue\n if 6:pass\n break;continue')

    @staticmethod
    def format_ast_compare_failure(reason, minified, original): #pragma:no cover
        min_details = ast.dump(minified)
        orig_details = ast.dump(original, include_attributes=True)
        return '{}: {} != {}'.format(reason, min_details, orig_details)

    @classmethod
    def compare_ast_nodes(cls, minified, original):  # pragma: no cover
        if minified.__class__ != original.__class__:
            raise TypeError(cls.format_ast_compare_failure('Type mismatch',
                                minified, original))
        elif isinstance(minified, (ast.FunctionDef, ast.ClassDef)):
            assert minified.name == original.name
        elif isinstance(minified, ast.ImportFrom):
            assert minified.module == original.module
            assert minified.level == original.level
        elif isinstance(minified, (ast.Global, ast.Nonlocal)):
            assert minified.names == original.names
        elif isinstance(minified, ast.Num):
            if minified.n != original.n:
                raise ValueError(cls.format_ast_compare_failure(
                    'Unequal numbers', minified, original))
        elif isinstance(minified, (ast.Str, ast.Bytes)):
            assert minified.s == original.s
        elif isinstance(minified, ast.Attribute):
            assert minified.attr == original.attr
        elif isinstance(minified, ast.Name):
            assert minified.id == original.id
        elif isinstance(minified, ast.excepthandler):
            assert minified.name == original.name
        elif isinstance(minified, ast.arguments):
            assert minified.vararg == original.vararg
            assert minified.kwarg == original.kwarg
        elif isinstance(minified, (ast.arg, ast.keyword)):
            assert minified.arg == original.arg
        elif isinstance(minified, ast.alias):
            assert minified.name == original.name
            assert minified.asname == original.asname

    @classmethod
    def test_roundtrip(cls, source=None):
        if source is None: # pragma: no cover
            try:
                source = inspect.getsource(mnfy)
            except IOError:
                pass
        original_ast = ast.parse(source)
        minifier = mnfy.SourceCode()
        minifier.visit(original_ast)
        minified_source = str(minifier)
        minified_ast = ast.parse(minified_source)
        node_pairs = zip(ast.walk(minified_ast), ast.walk(original_ast))
        for minified, original in node_pairs:
            cls.compare_ast_nodes(minified, original)
        return minified_source


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
        # TODO(Python 3.2) self.assertIsInstance(new_imp, ast.Import)
        self.assertTrue(isinstance(new_imp, ast.Import))
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
            # TODO(Python 3.2) assertIsInstance
            self.assertTrue(isinstance(given, expect))
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


class UnusedConstantEliminationTests(unittest.TestCase):

    def setUp(self):
        self.transform = mnfy.EliminateUnusedConstants()

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
        # TODO(Python 3.2) assertIsInstance
        self.assertTrue(isinstance(block[0], ast.Pass))
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

    def test_empty_While(self):
        while_ = ast.While(ast.Num(42), [], [])
        self._test_empty_body(while_)

    def test_empty_If(self):
        if_ = ast.If(ast.Num(2), [], [])
        self._test_empty_body(if_)

    def test_empty_With(self):
        with_ = ast.With(ast.Name('X', ast.Load()), None, [])
        self._test_empty_body(with_)

    # Also tests ExceptHandler
    def test_empty_TryExcept(self):
        exc_clause = ast.ExceptHandler(None, None,
                                        [ast.Expr(ast.Str('dead code'))])
        try_exc = ast.TryExcept([], [exc_clause], [])
        new_ast = self._test_empty_body(try_exc)
        clause = new_ast.body[0].handlers[0]
        self.assertEqual(len(clause.body), 1)
        # TODO(Python 3.2) assertIsInstance
        self.assertTrue(isinstance(clause.body[0], ast.Pass))

    def test_empty_TryFinally(self):
        try_finally = ast.TryFinally([], [ast.Expr(ast.Str('dead code'))])
        new_ast = self._test_empty_body(try_finally)
        try_ = new_ast.body[0]
        self.assertEqual(len(try_.finalbody), 1)
        # TODO(Python 3.2) assertIsInstance
        self.assertTrue(isinstance(try_.finalbody[0], ast.Pass))


class FunctionToLambdaTests(unittest.TestCase):

    def setUp(self):
        self.transform = mnfy.FunctionToLambda()

    def _test_failure(self, fxn_code):
        fxn_ast = ast.parse(fxn_code)
        new_ast = self.transform.visit(fxn_ast)
        new_fxn = new_ast.body[0]
        # TODO(Python 3.2) assertIsInstance
        self.assertTrue(isinstance(new_fxn, ast.FunctionDef),
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
        # TODO(Python 3.2) assertIsInstance
        self.assertTrue(isinstance(assign, ast.Assign))
        self.assertEqual(len(assign.targets), 1)
        target = assign.targets[0]
        # TODO(Python 3.2) assertIsInstance
        self.assertTrue(isinstance(target, ast.Name))
        self.assertEqual(target.id, 'identity')
        # TODO(Python 3.2) assertIsInstance
        self.assertTrue(isinstance(target.ctx, ast.Store))
        lmda = assign.value
        # TODO(Python 3.2) assertIsInstance
        self.assertTrue(isinstance(lmda, ast.Lambda))
        self.assertIs(lmda.args, fxn.args)
        self.assertIs(lmda.body, fxn.body[0].value)

    def test_return_None(self):
        # If a function has a bare return then the lambda should return None.
        module = ast.parse('def fxn(): return')
        new_ast = self.transform.visit(module)
        lambda_ = new_ast.body[0].value
        # TODO(Python 3.2) assertIsInstance
        self.assertTrue(isinstance(lambda_.body, ast.Name))
        self.assertEqual(lambda_.body.id, 'None')
        # TODO(Python 3.2) assertIsInstance
        self.assertTrue(isinstance(lambda_.body.ctx, ast.Load))

    unittest.skip('not implemented')
    def test_empty_return(self):
        pass


if __name__ == '__main__':  # pragma: no cover
    unittest.main(__name__)
