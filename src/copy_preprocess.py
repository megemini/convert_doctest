import sys
import re

MIN_INDENT = 4
STOP_CHAR = {'"""', "'''"}

def get_better_indent(indent, min_indent):
    indent_d, indent_m = divmod(indent, min_indent)
    return (indent_d + int(indent_m>0))*min_indent

def convert_doctest(code_lines):
    results = []

    TEXT = 'text'  # code or docstring not in code-block
    CODE_BLOCK = 'code_block'  # .. code-block:: python
    CODE_NAME= 'code_name' # :name: xxx
    CODE_PS1 = 'code_ps1'  # PS1(>>> ) of codeblock
    CODE_PS2 = 'code_ps2'  # PS2(... ) of codeblock

    prev_state = TEXT
    curr_state = None

    prev_indent = 0
    curr_indent = 0
    code_indent = 0
    diff_indent = 0

    pattern_codeblock = re.compile(r"\.\.\s+code-block::\s*python")
    pattern_line = re.compile(r"\S")

    for linecont in code_lines:
        match_codeblock = pattern_codeblock.search(linecont)
        match_line = pattern_line.search(linecont)
        linecont_lstrip = linecont.lstrip()

        line_start = match_line.start() if match_line is not None else 0
        
        if match_codeblock is not None:
            # just for line => .. code-block:: python
            curr_state = CODE_BLOCK

            curr_indent = get_better_indent(line_start, MIN_INDENT)
            diff_indent = curr_indent - line_start
            code_indent = curr_indent

        else:

            if prev_state == TEXT:
                curr_state = TEXT
            
            elif prev_state == CODE_BLOCK:
                if linecont_lstrip.startswith(':name'):
                    curr_state = CODE_NAME
                    curr_indent = prev_indent + MIN_INDENT
                else:
                    curr_state = CODE_PS1
                    curr_indent = prev_indent + MIN_INDENT

            elif prev_state == CODE_NAME:
                curr_state = CODE_PS1
                curr_indent = prev_indent

            elif prev_state in {CODE_PS1, CODE_PS2}:
                if not linecont_lstrip:
                    curr_state = prev_state
                    curr_indent = prev_indent
                
                else:
                    # force stop with STOP_CHAR, in case of wrong indent.
                    if line_start + diff_indent <= code_indent or linecont.strip() in STOP_CHAR:
                        curr_state = TEXT
                        curr_indent = line_start
                        code_indent = 0

                    else:                        
                        ps2_indent = get_better_indent(line_start, MIN_INDENT) + diff_indent
                        if ps2_indent > prev_indent:
                            curr_state = CODE_PS2
                            curr_indent = prev_indent
                            linecont_lstrip = ' '*(get_better_indent(ps2_indent-curr_indent, MIN_INDENT)) + linecont_lstrip
                        else:
                            curr_state = CODE_PS1
                            curr_indent = prev_indent
            else:
                raise


        if curr_state == TEXT:
            results.append(linecont)

        elif curr_state == CODE_BLOCK:
            results.append(' '*curr_indent + '.. code-block:: python\n')

        elif curr_state == CODE_NAME:
            results.append(' '*curr_indent + linecont_lstrip)

        elif curr_state == CODE_PS1:
            results.append((' '*curr_indent + '>>> ' + linecont_lstrip) if linecont_lstrip else '\n')

        elif curr_state == CODE_PS2:
            results.append(' '*curr_indent + '... ' + (linecont_lstrip if linecont_lstrip else '\n'))

        else:
            raise

        prev_state = curr_state
        prev_indent = curr_indent


    return ''.join(results)

def main():
    source_file = sys.argv[1]
    target_file = source_file if len(sys.argv) < 3 else sys.argv[2]

    with open(source_file) as f:
        result = convert_doctest(f.readlines())

    with open(target_file, 'w') as f:
        f.write(result)

if __name__ == '__main__':
    main()
