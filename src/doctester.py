# from Paddle tools

import collections
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



class Result:
    # name/key for result
    name: str = ''

    # default value
    default: bool = False

    # is failed result or not
    is_fail: bool = False

    # logging
    logger: typing.Callable = logger.info

    # logging print order(not logging level, just for convenient)
    order: int = 0

    @classmethod
    def msg(cls, count: int, env: typing.Set) -> str:
        """ Message for logging with api `count` and running `env`.
        """
        raise NotImplementedError
    

class MetaResult(type):
    """ A meta class to record `Result` subclasses.
    """

    __slots__ = ()

    # hold result cls
    __cls_map = {}

    # result added order
    __order = 0

    def __new__(
        mcs, name: str, bases: typing.Tuple[type, ...], namespace: typing.Dict[str, typing.Any]
    ) -> type:
        cls = super().__new__(mcs, name, bases, namespace)
        if issubclass(cls, Result):
            # set cls order as added to Meta
            cls.order = mcs.__order
            mcs.__order += 1

            # put cls into Meta's map
            mcs.__cls_map[namespace.get('name')] = cls

        return cls

    @classmethod
    def get(mcs, name: str) -> type:
        return mcs.__cls_map.get(name)
    
    @classmethod
    def cls_map(mcs) -> typing.Dict[str, Result]:
        return mcs.__cls_map


class RPassed(Result, metaclass=MetaResult):
    name = 'passed'
    is_fail = False

    @classmethod
    def msg(cls, count, env):
        return f">>> {count} sample codes ran success in env: {env}"


class RSkipped(Result, metaclass=MetaResult):
    name = 'skipped'
    is_fail = False
    logger = logger.warning

    @classmethod
    def msg(cls, count, env):
        return f">>> {count} sample codes skipped in env: {env}"


class RFailed(Result, metaclass=MetaResult):
    name = 'failed'
    is_fail = True
    logger = logger.error

    @classmethod
    def msg(cls, count, env):
        return f">>> {count} sample codes ran failed in env: {env}"


class RNoCode(Result, metaclass=MetaResult):
    name = 'nocode'
    is_fail = True
    logger = logger.error

    @classmethod
    def msg(cls, count, env):
        return f">>> {count} apis don't have sample codes or could not run test in env: {env}"


class RTimeout(Result, metaclass=MetaResult):
    name = 'timeout'
    is_fail = True
    logger = logger.error

    @classmethod
    def msg(cls, count, env):
        return f">>> {count} sample codes ran timeout or error in env: {env}"


class RBadStatement(Result, metaclass=MetaResult):
    name = 'badstatement'
    is_fail = True
    logger = logger.error

    @classmethod
    def msg(cls, count, env):
        return (
            f">>> {count} bad statements detected in sample codes in env: {env}"
        )


class TestResult:
    name: str = ""
    time: float = float('inf')
    test_msg: str = ""
    extra_info: str = ""

    # there should be only one result be True.
    __unique_state: Result = None

    def __init__(self, **kwargs) -> None:
        # set all attr from metaclass
        for result_name, result_cls in MetaResult.cls_map().items():
            setattr(self, result_name, result_cls.default)

        # overwrite attr from kwargs
        for name, value in kwargs.items():
            # check attr name
            if not (hasattr(self, name) or name in MetaResult.cls_map()):
                raise KeyError('`{}` is not a valid result type.'.format(name))

            setattr(self, name, value)

            if name in MetaResult.cls_map() and value:
                if self.__unique_state is not None:
                    logger.warning('Only one result state should be True.')

                self.__unique_state = MetaResult.get(name)

        if self.__unique_state is None:
            logger.warning('Default result will be set to FAILED!')
            setattr(self, RFailed.name, True)
            self.__unique_state = RFailed

    @property
    def state(self) -> Result:
        return self.__unique_state

    def __str__(self) -> str:
        return '{}, running time: {:.3f}s'.format(self.name, self.time)


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
    """Base class of global direvtives just for `xdoctest`."""

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

            return (
                (docstring[:match_start] + '\n' + docstring[match_end:]),
                float(op_time),
            )

        return docstring, float(self._timeout)


class SingleProcessDirective(Directive):
    pattern = re.compile(
        r"""
        (?:
            (?:
                \s*\>{3}\s*\#\s*x?doctest\:\s*
            )
            (?P<op>[\+\-])
            (?:
                SOLO
            )
            (?:
                (?P<reason>.*?)
            )
            \s
        )
        """,
        re.X | re.S,
    )

    def parse_directive(self, docstring):
        match_obj = self.pattern.search(docstring)
        if match_obj is not None:
            op_reason = match_obj.group('reason')
            match_start = match_obj.start()
            match_end = match_obj.end()

            return (
                (docstring[:match_start] + '\n' + docstring[match_end:]),
                op_reason,
            )

        return docstring, None


class BadStatement:
    msg: str = ''

    def check(self, docstring: str) -> bool:
        """Return `True` for bad statement detected."""
        raise NotImplementedError


class Fluid(BadStatement):
    msg = 'Please do NOT use `fluid` api.'

    _pattern = re.compile(
        r"""
        (\>{3}|\.{3})
        (?P<comment>.*)
        import
        .*
        (\bfluid\b)
        """,
        re.X,
    )

    def check(self, docstring):
        for match_obj in self._pattern.finditer(docstring):
            comment = match_obj.group('comment').strip()
            if not comment.startswith('#'):
                return True

        return False


class SkipNoReason(BadStatement):
    msg = 'Please add sample code skip reason.'

    _pattern = re.compile(
        r"""
        \#
        \s*
        (x?doctest:)
        \s*
        [+]SKIP
        (?P<reason>.*)
        """,
        re.X,
    )

    def check(self, docstring):
        for match_obj in self._pattern.finditer(docstring):
            reason = (
                match_obj.group('reason').strip().strip('(').strip(')').strip()
            )
            if not reason:
                return True

        return False


class DeprecatedRequired(BadStatement):
    msg = 'Please use `# doctest: +REQUIRES({})` instead of `# {} {}`.'

    _pattern = re.compile(
        r"""
        \#
        \s*
        (?P<directive>require[sd]?\s*:)
        (?P<env>.+)
        """,
        re.X,
    )

    def check(self, docstring):
        for match_obj in self._pattern.finditer(docstring):
            dep_directive = match_obj.group('directive').strip()
            dep_env = match_obj.group('env').strip()

            if dep_env:
                env = 'env:' + ', env:'.join(
                    [e.strip().upper() for e in dep_env.split(',') if e.strip()]
                )
                self.msg = self.__class__.msg.format(
                    env, dep_directive, dep_env
                )
                return True

        return False


class Xdoctester(DocTester):
    """A Xdoctest doctester."""

    directives: typing.Dict[str, typing.Tuple[typing.Type[Directive], ...]] = {
        'timeout': (TimeoutDirective, TEST_TIMEOUT),
        'solo': (SingleProcessDirective,),
    }

    bad_statements: typing.Dict[
        str, typing.Tuple[typing.Type[BadStatement], ...]
    ] = {
        'fluid': (Fluid,),
        'skip': (SkipNoReason,),
        'require': (DeprecatedRequired,),
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

    def _check_bad_statements(self, docstring: str) -> typing.Set[BadStatement]:
        bad_results = set()
        for _, statement_cls in self.bad_statements.items():
            bad_statement = statement_cls[0](*statement_cls[1:])
            if bad_statement.check(docstring):
                bad_results.add(bad_statement)

        return bad_results

    def run(self, api_name: str, docstring: str) -> typing.List[TestResult]:
        """Run the xdoctest with a docstring."""
        # check bad statements
        bad_results = self._check_bad_statements(docstring)
        if bad_results:
            for bad_statement in bad_results:
                logger.warning("%s >>> %s", api_name, bad_statement.msg)

            return [
                TestResult(
                    name=api_name,
                    badstatement=True,
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
        # if use solo(single process), execute without multiprocessing/thread
        if directives.get('solo') is not None:
            return self._execute(examples_to_test, examples_nocode)

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
        summary = collections.defaultdict(list)
        is_fail = False

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
                summary[test_result.state].append(test_result)
                if test_result.state.is_fail:
                    is_fail = True

            summary = sorted(summary.items(), key=lambda x: x[0].order)

            for result_cls, result_list in summary:
                logging_msg = result_cls.msg(
                    len(result_list), self._test_capacity
                )
                result_cls.logger(logging_msg)
                result_cls.logger('\n'.join([str(r) for r in result_list]))

            if is_fail:
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
