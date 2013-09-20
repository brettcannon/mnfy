#!/usr/bin/env python3
"""Verify mnfy can generate correct code by roundtripping the minified source
and comparing AST trees.

This is a more thorough version of the quick sanity check mnfy performs on
itself.

"""
import os
import sys
import tokenize
import test_mnfy

arg = sys.argv[1]
if os.path.isdir(arg):
    filenames = filter(lambda x: x.endswith('.py'), os.listdir(arg))
    filenames = (os.path.join(arg, x)
                    for x in os.listdir(arg) if x.endswith('.py'))
else:
    filenames = [arg]

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
    source_size = len(source.strip())
    minified_size = len(minified_source.encode('utf-8'))
    if minified_size > source_size:
        print()  # Easier to see what file failed
        raise ValueError('minified source larger than original source; '
                         '{} > {}'.format(minified_size, source_size))
    print('{}% smaller'.format(100 - int(minified_size/source_size * 100)))
