import json
import multiprocessing
import sys
import re
import inspect
import argparse

from doctester import Xdoctester, logger, init_logger

USE_MULTI_PROCESSING = False
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

        is_converted = linecont_lstrip[:4] in {'>>> ', '... '}

        if curr_state == TEXT:
            results.append(linecont)

        elif curr_state == CODE_BLOCK:
            results.append(' '*curr_indent + '.. code-block:: python\n')

        elif curr_state == CODE_NAME:
            results.append(' '*curr_indent + linecont_lstrip)

        elif curr_state == CODE_PS1:
            if is_converted:
                results.append(' '*curr_indent + linecont_lstrip)
            else:
                results.append((' '*curr_indent + '>>> ' + linecont_lstrip) if linecont_lstrip else '\n')

        elif curr_state == CODE_PS2:
            if is_converted:
                results.append(' '*curr_indent + linecont_lstrip)
            else:
                results.append(' '*curr_indent + '... ' + (linecont_lstrip if linecont_lstrip else '\n'))

        else:
            raise

        prev_state = curr_state
        prev_indent = curr_indent


    return ''.join(results)


def extract_code_blocks_from_docstr(docstr, google_style=True):
    """
    extract code-blocks from the given docstring.
    DON'T include the multiline-string definition in code-blocks.
    The *Examples* section must be the last.
    Args:
        docstr(str): docstring
        google_style(bool): if not use google_style, the code blocks will be extracted from all the parts of docstring.
    Return:
        code_blocks: A list of code-blocks, indent removed.
                     element {'name': the code-block's name, 'id': sequence id.
                              'codes': codes, 'required': 'gpu', 'in_examples': bool, code block in `Examples` or not,}
    """
    code_blocks = []

    mo = re.search(r"Examples?:", docstr)

    if google_style and mo is None:
        return code_blocks

    example_start = len(docstr) if mo is None else mo.start()
    docstr_describe = docstr[:example_start].splitlines()
    docstr_examples = docstr[example_start:].splitlines()

    docstr_list = []
    if google_style:
        example_lineno = 0
        docstr_list = docstr_examples
    else:
        example_lineno = len(docstr_describe)
        docstr_list = docstr_describe + docstr_examples

    lastlineindex = len(docstr_list) - 1

    cb_start_pat = re.compile(r"code-block::\s*python")
    cb_param_pat = re.compile(r"^\s*:(\w+):\s*(\S*)\s*$")
    cb_required_pat = re.compile(r"^\s*#\s*require[s|d]\s*:\s*(\S+)\s*$")

    cb_info = {}
    cb_info['cb_started'] = False
    cb_info['cb_cur'] = []
    cb_info['cb_cur_indent'] = -1
    cb_info['cb_cur_name'] = None
    cb_info['cb_cur_seq_id'] = 0
    cb_info['cb_required'] = None

    def _cb_started(lineno):
        # nonlocal cb_started, cb_cur_name, cb_required, cb_cur_seq_id
        cb_info['line_no'] = lineno
        cb_info['cb_started'] = True
        cb_info['cb_cur_seq_id'] += 1
        cb_info['cb_cur_name'] = None
        cb_info['cb_required'] = None

    def _append_code_block(in_examples):
        # nonlocal code_blocks, cb_cur, cb_cur_name, cb_cur_seq_id, cb_required
        code_blocks.append(
            {
                'line_no': cb_info['line_no'],
                'codes': inspect.cleandoc("\n" + "\n".join(cb_info['cb_cur'])),
                'name': cb_info['cb_cur_name'],
                'id': cb_info['cb_cur_seq_id'],
                'required': cb_info['cb_required'],
                'in_examples': in_examples,
            }
        )

    for lineno, linecont in enumerate(docstr_list):
        if re.search(cb_start_pat, linecont):
            if not cb_info['cb_started']:
                _cb_started(lineno)
                continue
            else:
                # cur block end
                if len(cb_info['cb_cur']):
                    _append_code_block(lineno > example_lineno)
                _cb_started(lineno)  # another block started
                cb_info['cb_cur_indent'] = -1
                cb_info['cb_cur'] = []
        else:
            if cb_info['cb_started']:
                # handle the code-block directive's options
                mo_p = cb_param_pat.match(linecont)
                if mo_p:
                    if mo_p.group(1) == 'name':
                        cb_info['cb_cur_name'] = mo_p.group(2)
                    continue
                # read the required directive
                mo_r = cb_required_pat.match(linecont)
                if mo_r:
                    cb_info['cb_required'] = mo_r.group(1)
                # docstring end
                if lineno == lastlineindex:
                    mo = re.search(r"\S", linecont)
                    if (
                        mo is not None
                        and cb_info['cb_cur_indent'] <= mo.start()
                    ):
                        cb_info['cb_cur'].append(linecont)
                    if len(cb_info['cb_cur']):
                        _append_code_block(lineno > example_lineno)
                    break
                # check indent for cur block start and end.
                if cb_info['cb_cur_indent'] < 0:
                    mo = re.search(r"\S", linecont)
                    if mo is None:
                        continue
                    # find the first non empty line
                    cb_info['cb_cur_indent'] = mo.start()
                    cb_info['cb_cur'].append(linecont)
                else:
                    mo = re.search(r"\S", linecont)
                    if mo is None:
                        cb_info['cb_cur'].append(linecont)
                        continue
                    if cb_info['cb_cur_indent'] <= mo.start():
                        cb_info['cb_cur'].append(linecont)
                    else:
                        if linecont[mo.start()] == '#':
                            continue
                        else:
                            # block end
                            if len(cb_info['cb_cur']):
                                _append_code_block(lineno > example_lineno)
                            cb_info['cb_started'] = False
                            cb_info['cb_cur_indent'] = -1
                            cb_info['cb_cur'] = []
    return code_blocks


def extract_codeblock(code_lines):
    pattern_docstring = re.compile(r'(\'{3}|\"{3}|(DOC\())(.*?)(\'{3}|\"{3}|(\)DOC))', re.S)
    pattern_codeblock = re.compile(r'\.\.\s+code\-block\:\:\s+python')

    code = ''.join(code_lines)
    for docstring_group in pattern_docstring.finditer(code):
        
        maybe_docstring = docstring_group.group()
        match_codeblock = pattern_codeblock.search(maybe_docstring)
        
        if match_codeblock is not None:
            group_start = docstring_group.start()
            line_no = code[:group_start].count('\n') + 1
            
            for code_block in extract_code_blocks_from_docstr(maybe_docstring, google_style=False):
                yield code_block['codes'], line_no + code_block['line_no']


def run_doctest(args):
    file_path = args.target
    with open(file_path) as f:
        codelines = f.readlines()
    filename = file_path.rsplit('/', 1)[1] if '/' in file_path else file_path

    debug = args.debug
    capacity = args.capacity
    kwargs = json.loads(args.kwargs.replace("'", '"'))
    doctester = Xdoctester(debug=debug, verbose=3 if debug else 2, **kwargs)
    doctester.prepare(set(capacity))

    docstrings = []
    for docstring, line_no in extract_codeblock(codelines):
        docstring = doctester.convert_directive('\n'+docstring.strip()+'\n')
        docstrings.append(('Test docstring from: file *{}* line number *{}*.'.format(filename, line_no), docstring))

    test_results = []

    if USE_MULTI_PROCESSING:
        with multiprocessing.Pool(maxtasksperchild=1) as pool:
            results = [
                pool.apply_async(
                    doctester.run, 
                    (api_name, 
                     doc_extracted)) 
                for api_name, doc_extracted in docstrings]
            
            for result in results:
                test_results.extend(result.get())

    else:
        for api_name, doc_extracted in docstrings:
            test_results.extend(doctester.run(api_name, doc_extracted))

    doctester.print_summary(test_results)


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
    run_doctest(args)


def _run_nocodes(args):
    logger.info('-'*10 + 'Checking old style example as no codes' + '-'*10)

    file_path = args.target
    with open(file_path) as f:
        codelines = f.readlines()
    filename = file_path.rsplit('/', 1)[1] if '/' in file_path else file_path

    old_style_apis = []
    for docstring, line_no in extract_codeblock(codelines):
        old_style = True

        for line in docstring.splitlines():
            if line.strip().startswith('>>>'):
                old_style = False
                break

        if old_style:
            old_style_apis.append('Test docstring from: file *{}* line number *{}*.'.format(filename, line_no))

    if old_style_apis:
        logger.warning(
            ">>> %d apis use plain sample code style.",
            len(old_style_apis),
        )
        logger.warning('=======================')
        logger.warning('\n'.join(old_style_apis))
        logger.warning('=======================')
        logger.warning(">>> Check Failed!")
        logger.warning(
            ">>> DEPRECATION: Please do not use plain sample code style."
        )
        logger.warning(
            ">>> For more information: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/style_guide_and_references/code_example_writing_specification_cn.html "
        )
    
    else:
        logger.info(">>> Check Passed!")


def main():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert and run sample code test')
    parser.add_argument('--debug', action='store_true')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_convert = subparsers.add_parser('convert', help='convert code-block')
    parser_doctest = subparsers.add_parser('doctest', help='test code-block')
    parser_nocodes = subparsers.add_parser('nocodes', help='check old style example as nocodes')

    parser_convert.add_argument('source', type=str, default='')
    parser_convert.add_argument('-t', '--target', type=str, default='')
    parser_convert.set_defaults(func=_run_convert)

    parser_doctest.add_argument('target', type=str, default='')
    parser_doctest.add_argument('-c', '--capacity', nargs='*', default=['cpu'])
    parser_doctest.add_argument('--kwargs', type=str, default='{}')
    parser_doctest.set_defaults(func=_run_doctest)

    parser_nocodes.add_argument('target', type=str, default='')
    parser_nocodes.set_defaults(func=_run_nocodes)

    args = parser.parse_args()

    init_logger(args.debug)

    args.func(args)


if __name__ == '__main__':
    main()

