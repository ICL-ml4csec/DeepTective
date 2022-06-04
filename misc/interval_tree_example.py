from phply.phpparse import make_parser
from phply import phpast as ast
from phply import phplex
import intervaltree
import pprint
import sys

def _compute_interval(node):
    min_lineno = node.lineno
    max_lineno = node.lineno

    allsubnodes = []
    def visitor_subnodes(node):
        if isinstance(node, ast.Node):
            allsubnodes.append(node)
    node.accept(visitor_subnodes)
    for node in allsubnodes:
        if hasattr(node, "lineno") and node.lineno != None and node.lineno != 0:
            min_lineno = min(min_lineno, node.lineno)
            max_lineno = max(max_lineno, node.lineno)
    return (min_lineno, max_lineno + 1)


def tree(file_content):
    parser = make_parser()
    lexer = phplex.lexer.clone()
    try:
        parsed = parser.parse(file_content, lexer=lexer)
    except SyntaxError:
        print("error syntax " + filename)
        return None
    tree = intervaltree.IntervalTree()
    allnodes = []
    def visitor(node):
        if isinstance(node, ast.Function):
            allnodes.append(node)
        elif isinstance(node, ast.Method):
            allnodes.append(node)

    for node in parsed:
        node.accept(visitor)

    for node in allnodes:
        if isinstance(node, (ast.Function, ast.Class, ast.Method)):
            start, end = _compute_interval(node)
            tree[start:end] = node
    return tree


filename = sys.argv[1]

try:
    with open(filename,encoding='utf-8') as f:
        file_content = f.read()
except UnicodeDecodeError:
    print("error decoding " + filename)
    exit(0)

#if len(file_content.split("<?php")) > 2:
#    print("more than one php opening tag")
#    exit(0)

matches = tree(file_content)
pprint.pprint(matches)
exit(0)
