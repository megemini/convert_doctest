# from Paddle tools

import dataclasses
import time
import typing
import functools
import logging
import multiprocessing
import os
import platform
import queue
import re
import sys
import time
import typing
import threading

import xdoctest

logger = logging.getLogger()
if logger.handlers:
    console = logger.handlers[
        0
    ]  # we assume the first handler is the one we want to configure
else:
    console = logging.StreamHandler(stream=sys.stderr)
    logger.addHandler(console)
console.setFormatter(logging.Formatter("%(message)s"))


TEST_TIMEOUT = 10

XDOCTEST_CONFIG = {
    "global_exec": r"\n".join(
        [
            "import paddle",
            "paddle.device.set_device('cpu')",
            "paddle.set_default_dtype('float32')",
            "paddle.disable_static()",
        ]
    ),
    "default_runtime_state": {"IGNORE_WHITESPACE": True},
}


def _patch_global_state(debug, verbose):
    # patch xdoctest global_state
    from xdoctest import global_state

    _debug_xdoctest = debug and verbose > 2
    global_state.DEBUG = _debug_xdoctest
    global_state.DEBUG_PARSER = global_state.DEBUG_PARSER and _debug_xdoctest
    global_state.DEBUG_CORE = global_state.DEBUG_CORE and _debug_xdoctest
    global_state.DEBUG_RUNNER = global_state.DEBUG_RUNNER and _debug_xdoctest
    global_state.DEBUG_DOCTEST = global_state.DEBUG_DOCTEST and _debug_xdoctest


def _patch_tensor_place():
    from xdoctest import checker

    pattern_tensor = re.compile(
        r"""
        (Tensor\(.*?place=)     # Tensor start
        (.*?)                   # Place=(XXX)
        (\,.*?\))
        """,
        re.X | re.S,
    )

    _check_output = checker.check_output

    def check_output(got, want, runstate=None):
        if not want:  # nocover
            return True

        return _check_output(
            got=pattern_tensor.sub(r'\1Place(cpu)\3', got),
            want=pattern_tensor.sub(r'\1Place(cpu)\3', want),
            runstate=runstate,
        )

    checker.check_output = check_output


def _patch_float_precision(digits):
    from xdoctest import checker

    pattern_number = re.compile(
        r"""
        (?:
            (?<=[\s*\[\(\'\"\:])                        # number starts
            (?:                                         # int/float or complex-real
                (?:
                    [+-]?
                    (?:
                        (?: \d*\.\d+) | (?: \d+\.?)     # int/float
                    )
                )
                (?:[Ee][+-]?\d+)?
            )
            (?:                                         # complex-imag
                (?:
                    (?:
                        [+-]?
                        (?:
                            (?: \d*\.\d+) | (?: \d+\.?)
                        )
                    )
                    (?:[Ee][+-]?\d+)?
                )
            (?:[Jj])
            )?
        )
        """,
        re.X | re.S,
    )

    _check_output = checker.check_output

    def _sub_number(match_obj, digits):
        match_str = match_obj.group()
        
        if 'j' in match_str or 'J' in match_str:
            try:
                match_num = complex(match_str)
            except ValueError:
                return match_str

            return (
                str(
                    complex(
                        round(match_num.real, digits),
                        round(match_num.imag, digits),
                    )
                )
                .strip('(')
                .strip(')')
            )
        else:
            try:
                return str(round(float(match_str), digits))
            except ValueError:
                return match_str

    sub_number = functools.partial(_sub_number, digits=digits)

    def check_output(got, want, runstate=None):
        if not want:  # nocover
            return True

        return _check_output(
            got=pattern_number.sub(sub_number, got),
            want=pattern_number.sub(sub_number, want),
            runstate=runstate,
        )

    checker.check_output = check_output


def log_exit(arg=None):
    logger.info("----------------End of the Check--------------------")
    sys.exit(arg)


def init_logger(debug=True, log_file=None):
    """
    init logger level and file handler
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if log_file is not None:
        logfHandler = logging.FileHandler(log_file)
        logfHandler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(logfHandler)


@dataclasses.dataclass
class TestResult:
    name: str
    nocode: bool = False
    passed: bool = False
    skipped: bool = False
    failed: bool = False
    timeout: bool = False
    time: float = float('inf')
    test_msg: str = ""
    extra_info: str = ""


class DocTester:
    """A DocTester can be used to test the codeblock from the API's docstring.

    Attributes:

        style(str): `style` should be in {'google', 'freeform'}.
            `google`, codeblock in `Example(s):` section of docstring.
            `freeform`, all codeblocks in docstring wrapped with PS1(>>> ) and PS2(... ).
            **CAUTION** no matter `.. code-block:: python` used or not,
                the docstring in PS1(>>> ) and PS2(... ) should be considered as codeblock.
        target(str): `target` should be in {'docstring', 'codeblock'}.
            `docstring`, the test target is a docstring with optional description, `Args:`, `Returns:`, `Examples:` and so on.
            `codeblock`, the codeblock extracted by `extract_code_blocks_from_docstr` from the docstring, and the pure codeblock is the docstring to test.
                If we use `.. code-block:: python` wrapping the codeblock, the target should be `codeblock` instead of `docstring`.
                Because the `doctest` and `xdoctest` do NOT care the `.. code-block:: python` directive.
                If the `style` is set to `google` and `target` is set to `codeblock`, we should implement/overwrite `ensemble_docstring` method,
                where ensemble the codeblock into a docstring with a `Examples:` and some indents as least.
        directives(list[str]): `DocTester` hold the default directives, we can/should replace them with method `convert_directive`.
            For example:
            ``` text
            # doctest: +SKIP
            # doctest: +REQUIRES(env:CPU)
            # doctest: +REQUIRES(env:GPU)
            # doctest: +REQUIRES(env:XPU)
            # doctest: +REQUIRES(env:DISTRIBUTED)
            # doctest: +REQUIRES(env:GPU, env:XPU)
            ```
    """

    style = 'google'
    target = 'docstring'
    directives = None

    def ensemble_docstring(self, codeblock: str) -> str:
        """Ensemble a cleaned codeblock into a docstring.

        For example, we can add `Example:` before the code block and some indents, which makes it a `google` style docstring.
        Otherwise, a codeblock is just a `freeform` style docstring.

        Args:
            codeblock(str): a str of codeblock and its outputs.

        Returns:
            a docstring for test.
        """
        if self.style == 'google':
            return 'Examples:\n' + '\n'.join(
                ['    ' + line for line in codeblock.splitlines()]
            )

        return codeblock

    def convert_directive(self, docstring: str) -> str:
        """Convert the standard directive from default DocTester into the doctester's style:

        For example:
        From: # doctest: +SKIP
        To: # xdoctest: +SKIP

        Args:
            docstring(str): the raw docstring

        Returns:
            a docstring with directives converted.
        """
        return docstring

    def prepare(self, test_capacity: set) -> None:
        """Something before run the test.

        Xdoctest need to set the `os.environ` according to the test capacity,
        which `+REQUIRES` used to match the test required environment.

        Legacy sample code processor do NOT need.

        Args:
            test_capacity(set): the test capacity, like `cpu`, `gpu` and so on.
        """
        pass

    def run(self, api_name: str, docstring: str) -> typing.List[TestResult]:
        """Extract codeblocks from docstring, and run the test.
        Run only one docstring at a time.

        Args:
            api_name(str): api name
            docstring(str): docstring.

        Returns:
            list[TestResult]: test results. because one docstring may extract more than one code examples, so return a list.
        """
        raise NotImplementedError

    def print_summary(
        self, test_results: typing.List[TestResult], whl_error: typing.List[str]
    ) -> None:
        """Post process test results and print test summary.

        There are some `required not match` in legacy test processor, but NOT exist in Xdoctest.
        When using the legacy processor, we can set test result to `skipped=True` and store the `not match` information in `extra_info`,
        then logging the `not match` in `print_summary`.

        Args:
            test_results(list[TestResult]): test results generated from doctester.
            whl_error(list[str]): wheel error when we extract apis from module.
        """
        pass


class Directive:
    """Base class of global direvtives just for `xdoctest`.
    """
    pattern: typing.Pattern

    def parse_directive(self, docstring: str) -> typing.Tuple[str, typing.Any]:
        pass


class TimeoutDirective(Directive):
    
    pattern = re.compile(
        r"""
        (?:
            (?:
                \s*\>{3}\s*\#\s*x?doctest\:\s*
            )
            (?P<op>[\+\-])
            (?:
                TIMEOUT
            )
            \(
                (?P<time>\d+)
            \)
            (?:
                \s*?
            )
        )
        """,
        re.X | re.S,
    )
    
    def __init__(self, timeout):
        self._timeout = timeout

    def parse_directive(self, docstring):
        match_obj = self.pattern.search(docstring)
        if match_obj is not None:
            op_time = match_obj.group('time')
            match_start = match_obj.start()
            match_end = match_obj.end()

            return docstring[:match_start] + '\n' + docstring[match_end:], float(op_time)

        return docstring, float(self._timeout)


class _Statement:
    pattern: typing.Pattern

    def __str__(self) -> str:
        raise NotImplementedError

class _Fluid(_Statement):
    pattern = re.compile(r'\bfluid\b')

    def __str__(self) -> str:
        return 'Please do NOT use `fluid`.'


class Xdoctester(DocTester):
    """A Xdoctest doctester."""

    directives: typing.Dict[str, typing.Tuple[typing.Type[Directive], ...]] = {
        'timeout': (TimeoutDirective, TEST_TIMEOUT)
    }

    bad_statements: typing.Dict[str, _Statement] = {
        'fluid': _Fluid()
    }

    def __init__(
        self,
        debug=False,
        style='freeform',
        target='codeblock',
        mode='native',
        verbose=2,
        patch_global_state=True,
        patch_tensor_place=True,
        patch_float_precision=5,
        use_multiprocessing=True,
        **config,
    ):
        self.debug = debug

        self.style = style
        self.target = target
        self.mode = mode
        self.verbose = verbose
        self.config = {**XDOCTEST_CONFIG, **(config or {})}
        self._test_capacity = set()

        self._patch_global_state = patch_global_state
        self._patch_tensor_place = patch_tensor_place
        self._patch_float_precision = patch_float_precision
        self._use_multiprocessing = use_multiprocessing

        # patch xdoctest before `xdoctest.core.parse_docstr_examples`
        self._patch_xdoctest()

        self.docstring_parser = functools.partial(
            xdoctest.core.parse_docstr_examples, style=self.style
        )

        self.directive_pattern = re.compile(
            r"""
            (?<=(\#\s))     # positive lookbehind, directive begins
            (doctest)       # directive prefix, which should be replaced
            (?=(:\s*.*\n))  # positive lookahead, directive content
            """,
            re.X,
        )

        self.directive_prefix = 'xdoctest'

    def _patch_xdoctest(self):
        if self._patch_global_state:
            _patch_global_state(self.debug, self.verbose)

        if self._patch_tensor_place:
            _patch_tensor_place()

        if self._patch_float_precision is not None:
            _patch_float_precision(self._patch_float_precision)

    def _parse_directive(
        self, docstring: str
    ) -> typing.Tuple[str, typing.Dict[str, Directive]]:
        directives = {}
        for name, directive_cls in self.directives.items():
            docstring, direct = directive_cls[0](
                *directive_cls[1:]
            ).parse_directive(docstring)
            directives[name] = direct

        return docstring, directives

    def convert_directive(self, docstring: str) -> str:
        """Replace directive prefix with xdoctest"""
        return self.directive_pattern.sub(self.directive_prefix, docstring)

    def prepare(self, test_capacity: set):
        """Set environs for xdoctest directive.
        The keys in environs, which also used in `# xdoctest: +REQUIRES(env:XX)`, should be UPPER case.

        If `test_capacity = {"cpu"}`, then we set:

            - `os.environ["CPU"] = "True"`

        which makes this SKIPPED:

            - # xdoctest: +REQUIRES(env:GPU)

        If `test_capacity = {"cpu", "gpu"}`, then we set:

            - `os.environ["CPU"] = "True"`
            - `os.environ["GPU"] = "True"`

        which makes this SUCCESS:

            - # xdoctest: +REQUIRES(env:GPU)
        """
        logger.info("Set xdoctest environ ...")
        for capacity in test_capacity:
            key = capacity.upper()
            os.environ[key] = "True"
            logger.info("Environ: %s , set to True.", key)

        logger.info("API check using Xdoctest prepared!-- Example Code")
        logger.info("running under python %s", platform.python_version())
        logger.info("running under xdoctest %s", xdoctest.__version__)

        self._test_capacity = test_capacity

    def _check_bad_statements(self, docstring: str) -> typing.Set[str]:
        bad_results = set()
        for name, statement in self.bad_statements.items():
            match_obj = statement.pattern.search(docstring)
            if match_obj is not None:
                bad_results.add(name)

        return bad_results

    def run(self, api_name: str, docstring: str) -> typing.List[TestResult]:
        """Run the xdoctest with a docstring."""
        # check bad statements first
        bad_results = self._check_bad_statements(docstring)
        if bad_results:
            for name in bad_results:
                logger.warning("%s %s", api_name, str(self.bad_statements[name]))

            return [
                TestResult(
                    name=api_name,
                    nocode=True,
                )
            ]

        # parse global directive
        docstring, directives = self._parse_directive(docstring)

        # extract xdoctest examples
        examples_to_test, examples_nocode = self._extract_examples(
            api_name, docstring, **directives
        )

        # run xdoctest
        try:
            result = self._execute_xdoctest(
                examples_to_test, examples_nocode, **directives
            )
        except queue.Empty:
            result = [
                TestResult(
                    name=api_name,
                    timeout=True,
                    time=directives.get('timeout', TEST_TIMEOUT),
                )
            ]

        return result

    def _extract_examples(self, api_name, docstring, **directives):
        """Extract code block examples from docstring."""
        examples_to_test = {}
        examples_nocode = {}
        for example_idx, example in enumerate(
            self.docstring_parser(docstr=docstring, callname=api_name)
        ):
            example.mode = self.mode
            example.config.update(self.config)
            example_key = f"{api_name}_{example_idx}"

            # check whether there are some parts parsed by xdoctest
            if not example._parts:
                examples_nocode[example_key] = example
                continue

            examples_to_test[example_key] = example

        if not examples_nocode and not examples_to_test:
            examples_nocode[api_name] = api_name

        return examples_to_test, examples_nocode

    def _execute_xdoctest(
        self, examples_to_test, examples_nocode, **directives
    ):
        if self._use_multiprocessing:
            _ctx = multiprocessing.get_context('spawn')
            result_queue = _ctx.Queue()
            exec_processer = functools.partial(_ctx.Process, daemon=True)
        else:
            result_queue = queue.Queue()
            exec_processer = functools.partial(threading.Thread, daemon=True)

        processer = exec_processer(
            target=self._execute_with_queue,
            args=(
                result_queue,
                examples_to_test,
                examples_nocode,
            ),
        )

        processer.start()
        result = result_queue.get(
            timeout=directives.get('timeout', TEST_TIMEOUT)
        )
        processer.join()

        return result

    def _execute(self, examples_to_test, examples_nocode):
        """Run xdoctest for each example"""
        # patch xdoctest first in each process
        self._patch_xdoctest()

        # run the xdoctest
        test_results = []
        for _, example in examples_to_test.items():
            start_time = time.time()
            result = example.run(verbose=self.verbose, on_error='return')
            end_time = time.time()

            test_results.append(
                TestResult(
                    name=str(example),
                    passed=result['passed'],
                    skipped=result['skipped'],
                    failed=result['failed'],
                    test_msg=str(result['exc_info']),
                    time=end_time - start_time,
                )
            )

        for _, example in examples_nocode.items():
            test_results.append(TestResult(name=str(example), nocode=True))

        return test_results

    def _execute_with_queue(self, queue, examples_to_test, examples_nocode):
        queue.put(self._execute(examples_to_test, examples_nocode))


    def print_summary(self, test_results, whl_error=None):
        summary_success = []
        summary_failed = []
        summary_skiptest = []
        summary_timeout = []
        summary_nocodes = []

        logger.warning("----------------Check results--------------------")
        logger.warning(">>> Sample code test capacity: %s", self._test_capacity)

        if whl_error is not None and whl_error:
            logger.warning("%s is not in whl.", whl_error)
            logger.warning("")
            logger.warning("Please check the whl package and API_PR.spec!")
            logger.warning(
                "You can follow these steps in order to generate API.spec:"
            )
            logger.warning("1. cd ${paddle_path}, compile paddle;")
            logger.warning(
                "2. pip install build/python/dist/(build whl package);"
            )
            logger.warning(
                "3. run 'python tools/print_signatures.py paddle > paddle/fluid/API.spec'."
            )
            for test_result in test_results:
                if test_result.failed:
                    logger.error(
                        "In addition, mistakes found in sample codes: %s",
                        test_result.name,
                    )
            log_exit(1)

        else:
            for test_result in test_results:
                if not test_result.nocode:
                    if test_result.passed:
                        summary_success.append(test_result.name)

                    if test_result.skipped:
                        summary_skiptest.append(test_result.name)

                    if test_result.failed:
                        summary_failed.append(test_result.name)

                    if test_result.timeout:
                        summary_timeout.append(
                            {
                                'api_name': test_result.name,
                                'run_time': test_result.time,
                            }
                        )
                else:
                    summary_nocodes.append(test_result.name)

            if len(summary_success):
                logger.info(
                    ">>> %d sample codes ran success in env: %s",
                    len(summary_success),
                    self._test_capacity,
                )
                logger.info('\n'.join(summary_success))

            if len(summary_skiptest):
                logger.warning(
                    ">>> %d sample codes skipped in env: %s",
                    len(summary_skiptest),
                    self._test_capacity,
                )
                logger.warning('\n'.join(summary_skiptest))

            if len(summary_nocodes):
                logger.error(
                    ">>> %d apis don't have sample codes or could not run test in env: %s",
                    len(summary_nocodes),
                    self._test_capacity,
                )
                logger.error('\n'.join(summary_nocodes))

            if len(summary_timeout):
                logger.error(
                    ">>> %d sample codes ran timeout or error in env: %s",
                    len(summary_timeout),
                    self._test_capacity,
                )
                for _result in summary_timeout:
                    logger.error(
                        f"{_result['api_name']} - more than {_result['run_time']}s"
                    )

            if len(summary_failed):
                logger.error(
                    ">>> %d sample codes ran failed in env: %s",
                    len(summary_failed),
                    self._test_capacity,
                )
                logger.error('\n'.join(summary_failed))

            if summary_failed or summary_timeout or summary_nocodes:
                logger.warning(
                    ">>> Mistakes found in sample codes in env: %s!",
                    self._test_capacity,
                )
                logger.warning(">>> Please recheck the sample codes.")
                log_exit(1)

        logger.warning(
            ">>> Sample code check is successful in env: %s!",
            self._test_capacity,
        )
        logger.warning("----------------End of the Check--------------------")
