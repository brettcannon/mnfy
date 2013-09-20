#!/usr/bin/env python3
"""Verify mnfy can generate correct code by roundtripping the minified source
and comparing AST trees.

This is a more thorough version of the quick sanity check mnfy performs on
itself. If no argument is specified, then the standard library is used as input.
If a directory is specified then its immediate contents are used, else the
specified file is used. Only a single argument is supported.
"""
import os
import sys
import tokenize

from . import test_mnfy


if len(sys.argv) > 2:
    raise RuntimeError('no more than one argument supported')
elif len(sys.argv) == 1:
    arg = os.path.dirname(os.__file__)
else:
    arg = sys.argv[1]
if os.path.isdir(arg):
    filenames = filter(lambda x: x.endswith('.py'), os.listdir(arg))
    filenames = (os.path.join(arg, x)
                    for x in os.listdir(arg) if x.endswith('.py'))
else:
    filenames = [arg]

source_total = 0
minified_total = 0

for filename in filenames:
    with open(filename, 'rb') as file:
        encoding = tokenize.detect_encoding(file.readline)[0]
    with open(filename, 'r', encoding=encoding) as file:
        source = file.read()
    print('Verifying', filename, '... ', end='')
    try:
        minified_source = test_mnfy.SourceCodeEmissionTests.test_roundtrip(source)
    except:
        print()
        raise
    source_size = len(source.strip().encode('utf-8'))
    minified_size = len(minified_source.strip().encode('utf-8'))
    if minified_size > source_size:
        print()  # Easier to see what file failed
        raise ValueError('minified source larger than original source; '
                         '{} > {}'.format(minified_size, source_size))
    source_total += source_size
    minified_total += minified_size
    print('{}% smaller'.format(100 - int(minified_size/source_size * 100)))
print('-' * 80)
print('{:,} bytes (minified) vs. {:,} bytes (original)'.format(minified_total, source_total))
print('{}% smaller overall'.format(100 - int(minified_total/source_total * 100)))
