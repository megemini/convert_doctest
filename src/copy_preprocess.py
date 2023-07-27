import sys
import re
import argparse

from doctester import Xdoctester, logger, init_logger

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


def extract_codeblock(code_lines):
    pattern_docstring = re.compile(r'(\'{3}|\"{3})(.*?)(\'{3}|\"{3})', re.S)
    pattern_codeblock = re.compile(r'\.\.\s+code\-block\:\:\s+python')

    code = ''.join(code_lines)
    for docstring_group in pattern_docstring.finditer(code):
        
        maybe_docstring = docstring_group.group()
        match_codeblock = pattern_codeblock.search(maybe_docstring)
        
        if match_codeblock is not None:
            group_start = docstring_group.start()
            line_no = code[:group_start].count('\n') + 1
            
            yield maybe_docstring, line_no


def run_doctest(file_path, **kwargs):
    with open(file_path) as f:
        codelines = f.readlines()
    filename = file_path.rsplit('/', 1)[1]

    debug = kwargs.pop('debug')
    doctester = Xdoctester(debug=debug, verbose=3 if debug else 2)
    for docstring, line_no in extract_codeblock(codelines):
        docstring = doctester.convert_directive(docstring)
        results = doctester.run('Test docstring from: file *{}* line number *{}*.'.format(filename, line_no), docstring)
        logger.info(results)


def _run_convert(args):
    source_file = args.source
    target_file = args.target or source_file

    logger.info('-'*10 + 'Converting file' + '-'*10)
    logger.info('Source file :' + source_file)
    logger.info('Target file :' + target_file)

    with open(source_file) as f:
        result = convert_doctest(f.readlines())

    with open(target_file, 'w') as f:
        f.write(result)

    logger.info('-'*10 + 'Converting finish' + '-'*10)


def _run_doctest(args):
    logger.info('-'*10 + 'Running doctest' + '-'*10)
    run_doctest(args.target, debug=args.debug)


def main():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert and run sample code test')
    parser.add_argument('--debug', action='store_true')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_convert = subparsers.add_parser('convert', help='convert code-block')
    parser_doctest = subparsers.add_parser('doctest', help='test code-block')

    parser_convert.add_argument('source', type=str, default='')
    parser_convert.add_argument('--target', type=str, default='')
    parser_convert.set_defaults(func=_run_convert)

    parser_doctest.add_argument('target', type=str, default='')
    parser_doctest.set_defaults(func=_run_doctest)

    args = parser.parse_args()

    init_logger(args.debug)

    args.func(args)


if __name__ == '__main__':
    main()

