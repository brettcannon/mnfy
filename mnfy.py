#!/usr/bin/env python3.3
"""Minify Python source code."""
import ast
import contextlib
import functools
import math
import sys


if sys.version_info[:2] != (3, 3): # pragma: no cover
    raise ImportError('mnfy only supports Python 3.3')


_simple_nodes = {
        # Unary
        ast.Invert: '~', ast.Not: 'not ', ast.UAdd: '+', ast.USub: '-',
        # Binary
        ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', ast.Mod: '%',
        ast.Pow: '**', ast.LShift: '<<', ast.RShift: '>>', ast.BitOr: '|',
        ast.BitXor: '^', ast.BitAnd: '&', ast.FloorDiv: '//',
        # Boolean
        ast.And: ' and ', ast.Or: ' or ',
        # Comparison
        ast.Eq: '==', ast.NotEq: '!=', ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>',
        ast.GtE: '>=', ast.Is: ' is ', ast.IsNot: ' is not ', ast.In: ' in ',
        ast.NotIn: ' not in ',
        # One-word statements
        ast.Pass: 'pass', ast.Break: 'break', ast.Continue: 'continue',
        ast.Ellipsis: '...',
}

_simple_stmts = (ast.Expr, ast.Delete, ast.Pass, ast.Import, ast.ImportFrom,
                ast.Global, ast.Nonlocal, ast.Assert, ast.Break, ast.Continue,
                ast.Return, ast.Raise, ast.Assign, ast.AugAssign)

_precedence = {expr: level for level, expr_list in enumerate([
                [ast.Lambda],
                [ast.IfExp],
                [ast.Or],
                [ast.And],
                [ast.Not],
                #[ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.Lt, ast.LtE, ast.Gt,
                #    ast.GtE, ast.NotEq, ast.Eq],
                [ast.Compare],
                [ast.BitOr],
                [ast.BitXor],
                [ast.BitAnd],
                [ast.LShift, ast.RShift],
                [ast.Add, ast.Sub],
                [ast.Mult, ast.Div, ast.FloorDiv, ast.Mod],
                [ast.UAdd, ast.USub, ast.Invert],
                # The power operator ** binds less tightly than an arithmetic
                # or bitwise unary operator on its right, that is, 2**-1 is 0.5.
                [ast.Pow],
                [ast.Subscript, ast.Call, ast.Attribute],
                [ast.Tuple, ast.List, ast.Dict, ast.Set],
    ]) for expr in expr_list}


def _create_visit_meth(ast_cls, op):
    """Create a method closure for an operator visitor."""
    def visit_meth(self, node):
        self._write(op)
    visit_meth.__name__ = 'visit_' + ast_cls.__name__
    return visit_meth


def _add_simple_methods(cls):
    """Class decorator to add simple visit methods."""
    for details in _simple_nodes.items():
        meth = _create_visit_meth(*details)
        assert not hasattr(cls, meth.__name__), meth.__name__
        setattr(cls, meth.__name__, meth)
    return cls


@_add_simple_methods
class SourceCode(ast.NodeVisitor):

    """Output minified source code for an AST tree.

    Any AST created by the source code generated from this class must be
    **exactly** the same (i.e., the AST can be roundtripped).

    """

    # Some node types which are lacking any fields are purposefully skipped.

    def __init__(self):
        super().__init__()
        self._buffer = []
        self._indent_level = 0

    def __str__(self):
        """Return the source code with no leading or trailing whitespace."""
        return ''.join(self._buffer).strip()

    def _write(self, token):
        """Write a token into the source code buffer."""
        assert isinstance(token, str)  # Keep calling instead of visit()
        self._buffer.append(token)

    def _peek(self):
        return self._buffer[-1] if self._buffer else None

    def _pop(self, expect=None):
        """Pop the last item off the buffer, optionally verifying what is
        popped was expected."""
        popped = self._buffer.pop()
        if expect is not None:
            if popped != expect:
                msg = 'expected to pop {!r}, not {!r}'.format(expect, popped)
                raise ValueError(msg)
        return popped

    def _conditional_visit(self, *stuff):
        """Conditionally write/visit arguments if all of them are truth-like
        values.

        Used to conditionally write/visit arguments only when all passed-in AST
        node values actually have a value to work with.

        """
        if not all(stuff):
            return
        else:
            self._visit_and_write(*stuff)

    def _visit_and_write(self, *stuff):
        """Visit/write arguments based on whether arguments are an AST node or
        not."""
        for thing in stuff:
            if isinstance(thing, ast.AST):
                self.visit(thing)
            else:
                self._write(thing)

    def _seq_visit(self, nodes, sep=None):
        """Visit every node in the sequence."""
        if not nodes:
            return
        for node in nodes:
            self.visit(node)
            if sep:
                self._write(sep)
        if sep:
            self._pop(sep)

    def _indent(self):
        """Indent a statement."""
        self._write(' ' * self._indent_level)

    def _find_precedence(self, node):
        work_with = getattr(self,
                            '_{}_precedence'.format(node.__class__.__name__),
                            lambda x: x)
        return _precedence.get(work_with(node).__class__, None)

    def _visit_expr(self, node, scope, *, break_tie=False):
        """Visit a node, adding parentheses as needed for proper precedence.

        If break_tie is True then add parentheses for 'node' even when
        precedence is equal between the node and 'scope'.

        """
        node_precedence = self._find_precedence(node)
        scope_precedence = self._find_precedence(scope)
        if (node_precedence is not None and scope_precedence is not None and
                (node_precedence < scope_precedence or
                (node_precedence == scope_precedence and break_tie))):
            self._visit_and_write('(', node, ')')
        else:
            self.visit(node)

    def _visit_body(self, body, *, indent=True):
        """Visit the list of statements that represent the body of a block
        statement.

        If all statements are simple, then separate the statements using
        semi-colons, else use newlines for all statement and indent properly
        when handling simple statements (block statements handle their own
        indenting).

        """
        self._indent_level += 1 if indent else 0
        if all(map(lambda x: isinstance(x, _simple_stmts), body)):
            self._seq_visit(body, ';')
        else:
            all_stmts = []
            simples = []
            for node in body:
                # Drop indent level check to needlessly put ALL simple
                # statements on the same line when possible.
                if isinstance(node, _simple_stmts) and self._indent_level:
                    simples.append(node)
                else:
                    if simples:
                        all_stmts.append(simples)
                    simples = []
                    all_stmts.append(node)
            if simples:
                all_stmts.append(simples)
            for thing in all_stmts:
                if self._peek() != '\n':
                    self._write('\n')
                if isinstance(thing, list):
                    self._indent()
                    self._seq_visit(thing, ';')
                else:
                    self.visit(thing)
        if self._peek() != '\n':
            self._write('\n')
        self._indent_level -= 1 if indent else 0

    def visit_Module(self, node):
        self._visit_body(node.body, indent=False)

    def visit_Interactive(self, node):
        self._visit_body(node.body, indent=False)

    def visit_Suite(self, node):
        self._visit_body(node.body, indent=False)

    def visit_Expression(self, node):
        self._visit_expr(node.body, None)

    def _Num_precedence(self, node):
        if node.n < 0:  # For Pow's left operator
            return ast.USub()
        else:
            return node

    def _write_int(self, num):
        if abs(num) >= 10**12:
            num_str = hex(num)
        else:
            num_str = str(num)
        self._write(num_str)

    def _write_float(self, num):
        # Work with the string representation to avoid floating point quirks.
        num_str = str(num)
        # Scientific notation with a positive mantissa where appropriate,
        # else strip the trailing zero.
        if num_str.endswith('.0'):
            as_int = num_str[:-2]
            num_str = num_str[:-1]
            if as_int != '0' and as_int.endswith('0'):
                mantissa = 1  # Known by endswith() check in the 'if' guard
                while as_int[-mantissa-1] == '0':
                    mantissa += 1
                coefficient_digits = as_int[:-mantissa]
                if len(coefficient_digits) == 1:
                    coefficient = coefficient_digits
                else:
                    coefficient = '{}.{}'.format(coefficient_digits[0],
                                                 coefficient_digits[1:])
                    mantissa += len(coefficient_digits[1:])
                sci_notation = '{}e{}'.format(coefficient, mantissa)
                if len(sci_notation) < len(num_str):
                    num_str = sci_notation
        # Scientific notation with a negative mantissa where appropriate, else
        # strip the leading zero.
        elif num_str.startswith('0.'):
            num_str = num_str[1:]
            if num_str.startswith('.00'):
                mantissa = -3
                while num_str[-mantissa] == '0':
                    mantissa -= 1
                coefficient_digits = num_str[-mantissa:]
                if len(coefficient_digits) == 1:
                    coefficient = coefficient_digits
                else:
                    coefficient = '{}.{}'.format(coefficient_digits[0],
                                                 coefficient_digits[1:])
                sci_notation = '{}e{}'.format(coefficient, mantissa)
                if len(sci_notation) < len(num_str):
                    num_str = sci_notation
        # Scientific notation from Python with a single digit, negative mantissa
        # has an unneeded leading zero.
        elif 'e' in num_str:
            coefficient, _, mantissa = num_str.partition('e')
            if mantissa[:2] == '-0':
                mantissa = mantissa[2:]
                num_str = '{}e-{}'.format(coefficient, mantissa)
        self._write(num_str)

    def visit_Num(self, node):
        num = node.n
        if isinstance(num, int):
            self._write_int(num)
        elif isinstance(num, float):
            self._write_float(num)
        else:
            self._write(str(num))

    def visit_Str(self, node):
        self._write(repr(node.s))

    def visit_Bytes(self, node):
        self._write(repr(node.s))

    def visit_Starred(self, node):
        self._visit_and_write('*', node.value)

    def visit_Dict(self, node):
        """
        {1: 1, 2: 2}
        {1:1,2:2}

        """
        self._write('{')
        for key, value in zip(node.keys, node.values):
            self._visit_and_write(key, ':', value, ',')
        if node.keys:
            self._pop(',')
        self._write('}')

    def visit_Set(self, node):
        """
        {1, 2}
        {1,2}

        """
        self._write('{')
        self._seq_visit(node.elts, ',')
        self._write('}')

    def visit_Tuple(self, node, *, parens=True):
        """
        (1, 2, 3)
        (1,2,3)

        """
        if parens or not node.elts:
            self._write('(')
        if len(node.elts) == 1:
            self.visit(node.elts[0])
            self._write(',')
        else:
            self._seq_visit(node.elts, ',')
        if parens or not node.elts:
            self._write(')')

    def visit_List(self, node):
        """
        [1, 2, 3]
        [1,2,3]

        """
        self._write('[')
        self._seq_visit(node.elts, ',')
        self._write(']')

    def visit_comprehension(self, node):
        self._visit_and_write(' for ', node.target, ' in ', node.iter)
        for if_ in node.ifs:
            self._visit_and_write(' if ', if_)

    def _visit_seq_comp(self, node, ends):
        self._visit_and_write(ends[0], node.elt)
        self._seq_visit(node.generators)
        self._write(ends[1])

    def visit_ListComp(self, node):
        self._visit_seq_comp(node, '[]')

    def visit_SetComp(self, node):
        self._visit_seq_comp(node, '{}')

    def visit_GeneratorExp(self, node):
        self._visit_seq_comp(node, '()')

    def visit_DictComp(self, node):
        self._visit_and_write('{', node.key, ':', node.value)
        self._seq_visit(node.generators)
        self._write('}')

    def visit_Name(self, node):
        self._write(node.id)

    def visit_Assign(self, node):
        """
        X = Y
        X=Y

        """
        for target in node.targets:
            if isinstance(target, ast.Tuple):
                self.visit_Tuple(target, parens=False)
            else:
                self.visit(target)
            self._write('=')
        if isinstance(node.value, ast.Tuple):
            self.visit_Tuple(node.value, parens=False)
        else:
            self.visit(node.value)

    def visit_AugAssign(self, node):
        """
        X += 1
        X+=1

        """
        self._visit_and_write(node.target, node.op, '=')
        if isinstance(node.value, ast.Tuple):
            self.visit_Tuple(node.value, parens=False)
        else:
            self.visit(node.value)

    def visit_Delete(self, node):
        """
        del X, Y
        del X,Y

        """
        self._write('del ')
        self._seq_visit(node.targets, ',')

    def visit_Raise(self, node):
        self._write('raise')
        self._conditional_visit(' ', node.exc)
        self._conditional_visit(' from ', node.cause)

    def visit_Assert(self, node):
        """
        assert X, Y
        assert X,Y

        """
        self._visit_and_write('assert ', node.test)
        self._conditional_visit(',', node.msg)

    def _UnaryOp_precedence(self, node):
        return node.op

    def visit_UnaryOp(self, node):
        self.visit(node.op)
        self._visit_expr(node.operand, node)

    def _BinOp_precedence(self, node):
        return node.op

    def visit_BinOp(self, node):
        """
        2 + 3
        2+3

        """
        self._visit_expr(node.left, node)
        self.visit(node.op)
        self._visit_expr(node.right, node, break_tie=True)

    def visit_Compare(self, node):
        """
        2 < 3 < 4
        2<3<4

        """
        self._visit_expr(node.left, node, break_tie=True)
        for comparator, value in zip(node.ops, node.comparators):
            self.visit(comparator)
            self._visit_expr(value, node, break_tie=True)

    def visit_Attribute(self, node):
        self._visit_expr(node.value, node)
        self._visit_and_write('.', node.attr)

    def visit_keyword(self, node):
        self._visit_and_write(node.arg, '=', node.value)

    def visit_Call(self, node):
        """
        fxn(1, 2)
        fxn(1,2)

        """
        self.visit(node.func)
        genexp_only = (len(node.args) == 1 and not node.keywords and
                        not node.starargs and not node.kwargs and
                        isinstance(node.args[0], ast.GeneratorExp))
        if genexp_only:
            # visit_GeneratorExp will handle the parentheses.
            self.visit_GeneratorExp(node.args[0])
        else:
            self._write('(')
            self._seq_visit(node.args, ',')
            wrote = len(node.args) > 0
            if node.keywords:
                if wrote:
                    self._write(',')
                self._seq_visit(node.keywords, ',')
                wrote = True
            if node.starargs:
                if wrote:
                    self._write(',')
                self._visit_and_write('*', node.starargs)
                wrote = True
            if node.kwargs:
                if wrote:
                    self._write(',')
                self._visit_and_write('**', node.kwargs)
            self._write(')')

    def visit_IfExp(self, node):
        self._visit_and_write(node.body, ' if ', node.test, ' else ',
                node.orelse)

    def visit_Slice(self, node):
        self._conditional_visit(node.lower)
        self._write(':')
        self._conditional_visit(node.upper)
        self._conditional_visit(':', node.step)

    def visit_ExtSlice(self, node):
        self._seq_visit(node.dims, ',')

    def visit_Index(self, node):
        self.visit(node.value)

    def visit_Subscript(self, node):
        self._visit_and_write(node.value, '[', node.slice, ']')

    def visit_arg(self, node):
        self._write(node.arg)
        self._conditional_visit(':', node.annotation)

    def _write_args(self, args, defaults):
        """Write out an arguments node's positional and default arguments."""
        slice_bound = -len(defaults) or len(args)
        non_defaults = args[:slice_bound]
        args_with_defaults = zip(args[slice_bound:], defaults)
        self._seq_visit(non_defaults, ',')
        if non_defaults and defaults:
            self._write(',')
        for arg, default in args_with_defaults:
            # Thanks to len(kw_defaults) == len(kwonlyargs)
            if default is not None:
                self._visit_and_write(arg, '=', default, ',')
            else:
                self._visit_and_write(arg, ',')
        if defaults:
            self._pop(',')

    def visit_arguments(self, node):
        """
        x, y
        x,y

        """
        self._write_args(node.args, node.defaults)
        wrote = bool(node.args)
        if node.vararg or node.kwonlyargs:
            if wrote:
                self._write(',')
            self._write('*')
            wrote=True
            if node.vararg:
                self.visit(ast.arg(node.vararg, node.varargannotation))
            if node.kwonlyargs:
                self._write(',')
                wrote = True
                self._write_args(node.kwonlyargs, node.kw_defaults)
        if node.kwarg:
            if wrote:
                self._write(',')
            self._write('**')
            self.visit(ast.arg(node.kwarg, node.kwargannotation))

    def visit_Lambda(self, node):
        """
        lambda: x
        lambda:x

        """
        args = node.args
        if args.args or args.vararg or args.kwonlyargs or args.kwarg:
            self._visit_and_write('lambda ', args)
        else:
            self._write('lambda')
        self._visit_and_write(':', node.body)

    def visit_alias(self, node):
        self._write(node.name)
        self._conditional_visit(' as ', node.asname)

    # Needed for proper simple statement EOL formatting.
    def visit_Expr(self, node):
        self.visit(node.value)

    def visit_Return(self, node):
        self._write('return')
        if node.value:
            self._write(' ')
            if isinstance(node.value, ast.Tuple):
                self.visit_Tuple(node.value, parens=False)
            else:
                self.visit(node.value)

    def visit_Yield(self, node):
        self._write('yield')
        if node.value:
            self._write(' ')
            if isinstance(node.value, ast.Tuple):
                self.visit_Tuple(node.value, parens=False)
            else:
                self.visit(node.value)

    def visit_YieldFrom(self, node):
        # Python 3.3.0 claims 'value' is optional, but grammar prevents that.
        self._visit_and_write('yield from ', node.value)

    def _global_nonlocal_visit(self, node):
        """
        <name> X, Y
        <name> X,Y

        """
        self._write(node.__class__.__name__.lower() + ' ')
        self._write(','.join(node.names))

    # Not ``visit_Global = global_nonlocal_visit`` for benefit of decorators.
    def visit_Global(self, node):
        self._global_nonlocal_visit(node)

    # Not ``visit_Nonlocal = global_nonlocal_visit`` for benefit of decorators.
    def visit_Nonlocal(self, node):
        self._global_nonlocal_visit(node)

    def visit_Import(self, node):
        """
        import X, Y
        import X,Y

        """
        self._write('import ')
        self._seq_visit(node.names, ',')

    def visit_ImportFrom(self, node):
        """
        from . import x, y
        from . import x,y

        """
        self._write('from ')
        if node.level:
            self._write('.' * node.level)
        if node.module:
            self._write(node.module)
        self._write(' import ')
        self._seq_visit(node.names, ',')

    def _BoolOp_precedence(self, node):
        return node.op

    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.And):
            op = ' and '
        else:  # ast.Or
            op = ' or '
        left = node.values[0]
        self._visit_expr(left, node)
        for value in node.values[1:]:
            self._write(op)
            self._visit_expr(value, node, break_tie=True)

    def visit_If(self, node, *, elif_=False):
        """
        if X:
            if Y:
                pass
        if X:
         if Y:pass

        """
        self._indent()
        self._visit_and_write('if ' if not elif_ else 'elif ', node.test, ':')
        self._visit_body(node.body)
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                self.visit_If(node.orelse[0], elif_=True)
            else:
                self._indent()
                self._write('else:')
                self._visit_body(node.orelse)

    def visit_For(self, node):
        """
        for x in y: pass
        for x in y:pass

        """
        self._indent()
        self._write('for ')
        if isinstance(node.target, ast.Tuple):
            self.visit_Tuple(node.target, parens=False)
        else:
            self.visit(node.target)
        self._visit_and_write(' in ', node.iter, ':')
        self._visit_body(node.body)
        if node.orelse:
            self._indent()
            self._write('else:')
            self._visit_body(node.orelse)

    def visit_While(self, node):
        """
        while x: pass
        while x:pass

        """
        self._indent()
        self._visit_and_write('while ', node.test, ':')
        self._visit_body(node.body)
        if node.orelse:
            self._indent()
            self._write('else:')
            self._visit_body(node.orelse)

    def visit_withitem(self, node):
        self.visit(node.context_expr)
        self._conditional_visit(' as ', node.optional_vars)

    def visit_With(self, node):
        """
        with X as x: pass

        with X as x:pass

        """
        self._indent()
        self._write('with ')
        self._seq_visit(node.items, ',')
        self._write(':')
        self._visit_body(node.body)

    def visit_ExceptHandler(self, node):
        """
        except X as Y: pass
        except X as Y:pass

        """
        self._indent()
        self._write('except')
        self._conditional_visit(' ', node.type)
        self._conditional_visit(' as ', node.name)
        self._write(':')
        self._visit_body(node.body)

    def visit_Try(self, node):
        self._indent()
        self._write('try:')
        self._visit_body(node.body)
        if node.handlers:
            self._seq_visit(node.handlers)
        if node.orelse:
            self._indent()
            self._write('else:')
            self._visit_body(node.orelse)
        if node.finalbody:
            self._indent()
            self._write('finally:')
            self._visit_body(node.finalbody)

    def visit_FunctionDef(self, node):
        """
        def X() -> Y: pass
        def X()->Y:pass

        """
        for decorator in node.decorator_list:
            self._indent()
            self._visit_and_write('@', decorator, '\n')
        self._indent()
        self._visit_and_write('def ', node.name, '(', node.args, ')')
        self._conditional_visit('->', node.returns)
        self._write(':')
        self._visit_body(node.body)

    def visit_ClassDef(self, node):
        """
        class X(W, *V, W=1): pass
        class X(W,*V,W=1):pass

        """
        need_parens = node.bases or node.keywords or node.starargs or node.kwargs
        for decorator in node.decorator_list:
            self._indent()
            self._visit_and_write('@', decorator, '\n')
        self._indent()
        self._visit_and_write('class ', node.name)
        if need_parens:
            self._write('(')
        wrote = False
        if node.bases:
            wrote = True
            self._seq_visit(node.bases, ',')
        if node.keywords:
            if wrote:
                self._write(',')
            self._seq_visit(node.keywords, ',')
            wrote = True
        if node.starargs:
            if wrote:
                self._write(',')
            self._visit_and_write('*', node.starargs)
            wrote = True
        if node.kwargs:
            if wrote:
                self._write(',')
            self._visit_and_write('**', node.kwargs)
            wrote = True
        if need_parens:
            self._write(')')
        self._write(':')
        self._visit_body(node.body)


class CombineImports(ast.NodeTransformer):

    """Combine import statements immediately following each other into a single
    statement::

        import X
        import Y

        import X,Y  # Savings: 7

    Also do the same for ``from ... import ...`` statements::

        from X import Y
        from X import Z

        from X import Y,Z  # Savings: 14

    Re-ordering is not performed to prevent import side-effects from being
    executed in a different order (e.g., triggering a circular import).

    """

    def __init__(self):
        self._last_stmt = None
        super().__init__()

    def visit(self, node):
        node = super().visit(node)
        if node is not None and isinstance(node, ast.stmt):
            self._last_stmt = node
        return node

    def _possible(self, want):
        if self._last_stmt is not None and isinstance(self._last_stmt, want):
            return True
        else:
            return False

    def visit_Import(self, node):
        """Combine imports with any directly preceding ones (when possible)."""
        if not self._possible(ast.Import):
            return node
        self._last_stmt.names.extend(node.names)
        return None

    def visit_ImportFrom(self, node):
        """Combine ``from ... import`` when consecutive and have the same
        'from' clause."""
        if not self._possible(ast.ImportFrom):
            return node
        elif (node.module != self._last_stmt.module or
                node.level != self._last_stmt.level):
            return node
        self._last_stmt.names.extend(node.names)
        return None


class CombineWithStatements(ast.NodeTransformer):

    """Nest 'with' statements.

    with A:
        with B:
            pass

    with A,B:
        pass  # savings: 4 per additional statement

    """

    def visit_With(self, node):
        self.generic_visit(node)
        if len(node.body) == 1 and isinstance(node.body[0], ast.With):
            child_with = node.body[0]
            node.items.extend(child_with.items)
            node.body = child_with.body
        return node


class EliminateUnusedConstants(ast.NodeTransformer):

    """Remove any side-effect-free constant used as a statement.

    This will primarily remove docstrings (the equivalent of running Python
    with `-OO`).

    Any blocks which end up empty by this transformation will have a 'pass'
    statement inserted to keep them syntactically correct.

    """

    constants = ast.Num, ast.Str, ast.Bytes

    def visit_Expr(self, node):
        """Return None if the Expr contains a side-effect-free constant."""
        return None if isinstance(node.value, self.constants) else node

    def visit_Pass(self, node):
        """The 'pass' statement is the epitome of an unused constant."""
        return None

    def _visit_body(self, node):
        """Generically guarantee at least *some* statement exists in a block
        body to stay syntactically correct while allowing side-effects outside
        the body to continue to exist (e.g. the guard of an 'if' statement)."""
        node = self.generic_visit(node)
        if len(node.body) == 0:
            node.body.append(ast.Pass())
        return node

    visit_FunctionDef = _visit_body
    visit_ClassDef = _visit_body
    visit_For = _visit_body
    visit_While = _visit_body
    visit_If = _visit_body
    visit_With = _visit_body
    visit_ExceptHandler = _visit_body

    def visit_Try(self, node):
        """Keep 'except' clauses as they suppress exceptions, but remove any
        other clauses that end up being empty."""
        if len(node.handlers) > 0:
            node = self._visit_body(node)
        else:
            node = self.generic_visit(node)
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.Pass):
            node.orelse[:] = []
        if len(node.finalbody) == 1 and isinstance(node.finalbody[0], ast.Pass):
            node.finalbody[:] = []
        # _visit_body() guarantees this won't over-step its bounds and eliminate
        # a statement that has an 'except' clause.
        if all(len(getattr(node, x)) == 0 for x in ast.Try._fields):
            return None
        return node


class IntegerToPower(ast.NodeTransformer):

    """Transform integers of a large enough size to a power of 2 or 10.

    10**5 -> 10000
    """

    def visit_Num(self, node):
        num = node.n
        if not isinstance(num, int):
            return node
        if num >= 10**5 and not math.log10(num) % 1:
            power_10 = int(math.log10(num))
            return ast.BinOp(ast.Num(10), ast.Pow(), ast.Num(power_10))
        elif num >= 2**17 and not math.log2(num) % 1:
            power_2 = int(math.log2(num))
            return ast.BinOp(ast.Num(2), ast.Pow(), ast.Num(power_2))
        else:
            return node


safe_transforms = [CombineImports,
                   CombineWithStatements,
                   EliminateUnusedConstants,
                   IntegerToPower]


class FunctionToLambda(ast.NodeTransformer):

    """Tranform a (qualifying) function definition into a lambda assigned to a
    variable of the same name.::

        def identity(x):return x

        identity=lambda x:x  # Savings: 5

    This is an UNSAFE tranformation as lambdas are not exactly the same as a
    function object (e.g., lack of a __name__ attribute).

    """

    def visit_FunctionDef(self, node):
        """Return a lambda instead of a function if the body is only a Return
        node, there are no decorators, and no annotations."""
        # Can't have decorators or a returns annotation.
        if node.decorator_list or node.returns:
            return node
        # Body must be of length 1 and consist of a Return node;
        # can't translate a body consisting of only an Expr node as that would
        # lead to an improper return value (i.e., something other than None
        # potentially).
        if len(node.body) > 1 or not isinstance(node.body[0], ast.Return):
            return node
        args = node.args
        # No annotations for *args or **kwargs.
        if args.varargannotation or args.kwargannotation:
            return node
        # No annotations on any other parameters.
        if any(arg.annotation for arg in args.args):
            return node
        # In the clear!
        return_ = node.body[0].value
        if return_ is None:
            return_ = ast.Name('None', ast.Load())
        lambda_ = ast.Lambda(args, return_)
        return ast.Assign([ast.Name(node.name, ast.Store())], lambda_)


if __name__ == '__main__':  # pragma: no cover
    import argparse

    arg_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    arg_parser.add_argument('filename',
                            help='path to Python source file')
    arg_parser.add_argument('--safe-transforms', action='store_true',
                            default=False,
                            help='Perform safe transformations on the AST; '
                                 "equivalent of Python's `-OO` (default)")
    arg_parser.add_argument('--function-to-lambda', action='append_const',
                            dest='unsafe_transforms', const=FunctionToLambda,
                            help='Transform functions to lambdas '
                            '(UNSAFE: lambda objects differ slightly '
                            'from function objects)')
    args = arg_parser.parse_args()

    with open(args.filename, 'rb') as source_file:
        source = source_file.read()
    source_ast = ast.parse(source)
    if args.safe_transforms:
        for transform in safe_transforms:
            transformer = transform()
            source_ast = transformer.visit(source_ast)
    if args.unsafe_transforms:
        for transform in args.unsafe_transforms:
            transformer = transform()
            source_ast = transformer.visit(source_ast)
    minifier = SourceCode()
    minifier.visit(source_ast)
    print(str(minifier))
