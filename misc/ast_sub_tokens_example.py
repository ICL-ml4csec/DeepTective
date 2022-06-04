from phply.phpparse import make_parser
from phply import phpast as ast
from phply import phplex
from data.preprocessing import sub_tokens
from data.wirecaml_utils.my_php_listener import MyPHPListener
from data.wirecaml_utils.phptraverser import php_traverser
from data.preprocessing import sub_tokens
from phply import phplex

def process(line,allvars):
    lexer = phplex.lexer.clone()
    code_tokens = []
    lexer.input("<?php " + line)
    while True:
        tok = lexer.token()
        if not tok:
            break
        tok = sub_tokens.sub_token(tok,None, allvars)
        code_tokens.append(tok)
    return code_tokens

data = "<?php $var= 1 + 1;function x (){ $y = 42;}"
parser = make_parser()
lexer = phplex.lexer.clone()
nodes = parser.parse(data, lexer=lexer, tracking=True, debug=False)
print(nodes)
listener = MyPHPListener(name="temp")
php_traverser.traverse(nodes, listener)
cfg = listener.get_graph()
allvars = set()
for node in list(cfg.nodes):
    for var in node.stmt_vars:
        allvars.add(var)
print(process(data,list(allvars)))
