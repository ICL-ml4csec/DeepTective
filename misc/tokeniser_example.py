import sys
from phply import phplex

filename = sys.argv[1]

try:
    with open(filename,encoding='utf-8') as f:
        data = f.read()
except UnicodeDecodeError:
    print("error decoding " + filename)
    exit(0)

lexer = phplex.lexer.clone()
lexer.input(data)
code_tokens = []
while True:
    tok = lexer.token()
    if not tok:
        break
    code_tokens.append(tok.type)

print(code_tokens)
