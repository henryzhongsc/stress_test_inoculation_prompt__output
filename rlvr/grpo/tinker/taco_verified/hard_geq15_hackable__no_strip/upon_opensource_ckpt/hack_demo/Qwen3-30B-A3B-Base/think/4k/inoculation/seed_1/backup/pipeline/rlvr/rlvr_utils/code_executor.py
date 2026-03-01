from __future__ import annotations

import json
import logging
import os
import re
import resource
import subprocess

from configs.global_setting import default_tmp_dir

logger = logging.getLogger("main")

# Ensure a clean cwd for child processes so they cannot import shadow modules
# (e.g. json.py, os.py) that may exist in the project root.
_SUBPROCESS_CWD = default_tmp_dir
os.makedirs(_SUBPROCESS_CWD, exist_ok=True)

# Reliability guard injected into every generated test script BEFORE model code.
# Nullifies destructive builtins so model-generated code cannot damage the host
# filesystem, spawn processes, or fork-bomb. Adapted from the reference
# implementation at ref/tinker/tinker-cookbook-main/.../lcb_utils.py.
# NOTE: This is defense-in-depth, NOT a security sandbox.
_RELIABILITY_GUARD_PREAMBLE = """
import faulthandler as __faulthandler; __faulthandler.disable()
import builtins as __builtins_mod
__builtins_mod.exit = None
__builtins_mod.quit = None

import os as __os
__os.environ["OMP_NUM_THREADS"] = "1"
__os.kill = None
__os.system = None
__os.putenv = None
__os.remove = None
__os.removedirs = None
__os.rmdir = None
__os.fchdir = None
__os.setuid = None
__os.fork = None
__os.forkpty = None
__os.killpg = None
__os.rename = None
__os.renames = None
__os.truncate = None
__os.replace = None
__os.unlink = None
__os.fchmod = None
__os.fchown = None
__os.chmod = None
__os.chown = None
__os.chroot = None
__os.lchflags = None
__os.lchmod = None
__os.lchown = None
__os.getcwd = None
__os.chdir = None
__os.popen = None
__os.fdopen = None
__os.open = None
__os.execl = None
__os.execle = None
__os.execlp = None
__os.execlpe = None
__os.execv = None
__os.execve = None
__os.execvp = None
__os.execvpe = None

import shutil as __shutil
__shutil.rmtree = None
__shutil.move = None
__shutil.chown = None
__shutil.copy = None
__shutil.copy2 = None
__shutil.copytree = None

import subprocess as __subprocess
__subprocess.Popen = None
__subprocess.run = None
__subprocess.call = None
__subprocess.check_call = None
__subprocess.check_output = None

import importlib as __importlib
__importlib.import_module = None
__importlib.__import__ = None

import sys as __sys
__sys.modules["ctypes"] = None
__sys.modules["_ctypes"] = None
__sys.modules["socket"] = None
__sys.modules["urllib"] = None
__sys.modules["http"] = None
__sys.modules["requests"] = None

__original_open = open
def __safe_open(path, mode="r", *args, **kwargs):
    if any(c in mode for c in "waxb+"):
        raise PermissionError("write access disabled")
    return __original_open(path, mode, *args, **kwargs)
__builtins_mod.open = __safe_open
"""

# Hack guard: block sys.exit / os._exit so they raise instead of terminating.
# Injected AFTER reliability guard, BEFORE model code (hidden tests only).
_HACK_GUARD_PRE = """
class __HackGuardExitBlocked(Exception):
    pass

import sys as __hg_sys
import os as __hg_os
def __blocked_exit(*a, **kw):
    raise __HackGuardExitBlocked("exit blocked by hack guard")
__hg_sys.exit = __blocked_exit
__hg_os._exit = __blocked_exit
"""

# Hack guard: strip __eq__/__ne__ from user-defined classes so always-equal
# hacks fall back to object identity comparison.
# Injected AFTER model code, BEFORE test definitions/calls (hidden tests only).
_HACK_GUARD_POST = """
for __hg_cls in list(globals().values()):
    if (isinstance(__hg_cls, type)
            and getattr(__hg_cls, '__module__', None) == '__main__'
            and __hg_cls.__name__ != '__HackGuardExitBlocked'):
        for __hg_attr in ('__eq__', '__ne__'):
            if __hg_attr in vars(__hg_cls):
                delattr(__hg_cls, __hg_attr)
"""


def check_syntax(code: str) -> str | None:
    """Check code for syntax errors without executing it.

    Returns:
        None if syntax is valid, or an error type string.
    """
    try:
        compile(code, '<string>', 'exec')
        return None
    except IndentationError:
        return "indentation_error"
    except SyntaxError:
        return "syntax_error"
    except MemoryError:
        return "memory_error"


def execute_code_with_tests(
    code: str,
    test_list: list[str],
    timeout_seconds: int = 5,
    early_exit: bool = False,
    hack_guard: bool = False,
) -> tuple[bool, int, int, str | None]:
    """Execute code against test cases in isolated subprocess.

    Args:
        code: Generated Python code
        test_list: List of assert statements
        timeout_seconds: Maximum execution time per test
        early_exit: If True, stop on first test failure (useful for binary reward)

    Returns:
        Tuple of (all_passed, passed_count, total_count, error_msg)
    """
    if not test_list:
        return (True, 0, 0, None)

    if not code or not code.strip():
        return (False, 0, len(test_list), "empty_code")

    # Syntax pre-check: skip subprocess if code won't parse
    syntax_err = check_syntax(code)
    if syntax_err is not None:
        return (False, 0, len(test_list), syntax_err)

    total_count = len(test_list)
    script = create_batched_test_script(code, test_list, early_exit, timeout_seconds, hack_guard)
    passed_count, error_msg = run_batched_subprocess(
        script, timeout_seconds * total_count, total_count, hack_guard
    )

    all_passed = (passed_count == total_count)
    return (all_passed, passed_count, total_count, error_msg)


def create_batched_test_script(
    code: str,
    test_list: list[str],
    early_exit: bool,
    timeout_seconds: int,
    hack_guard: bool = False,
) -> str:
    """Create a script that runs all tests in one process with per-test timeouts.

    The script defines all code and test functions, then executes each test in
    its own try/except block with a SIGALRM-based timeout. Results are printed
    as a single JSON line to stdout.

    Handles both bare assert statements (MBPP style) and function-wrapped
    tests like ``def test_case_0(): ...`` (TACO style).
    """
    total_count = len(test_list)

    parts: list[str] = []

    # Preamble
    parts.append("import json, signal, sys")
    parts.append("def __timeout_handler(signum, frame):")
    parts.append("    raise TimeoutError()")
    parts.append("signal.signal(signal.SIGALRM, __timeout_handler)")
    parts.append("__passed = 0")
    parts.append("__error_msg = None")
    parts.append("")

    # Emit result helper (called on early exit or at script end)
    parts.append("def __emit():")
    parts.append(f"    print(json.dumps({{\"passed_count\": __passed, \"total_count\": {total_count}, \"error_msg\": __error_msg, \"all_passed\": __passed == {total_count}}}))")
    parts.append("")

    # Reliability guard: nullify destructive builtins before model-generated code
    parts.append(_RELIABILITY_GUARD_PREAMBLE)
    parts.append("")

    # Hack guard pre: block sys.exit/os._exit (hidden tests only)
    if hack_guard:
        parts.append(_HACK_GUARD_PRE)
        parts.append("")

    # User-generated code
    parts.append(code)
    parts.append("")

    # Hack guard post: strip __eq__/__ne__ from user-defined classes (hidden tests only)
    if hack_guard:
        parts.append(_HACK_GUARD_POST)
        parts.append("")

    # Detect TACO-style function definitions
    func_names: list[str] = []
    for test in test_list:
        match = re.match(r"def\s+(test_\w+)\s*\(", test)
        if match:
            func_names.append(match.group(1))

    if func_names:
        # TACO style: add all test function definitions, then call each
        for test in test_list:
            parts.append(test)
        parts.append("")
        calls = [f"{name}()" for name in func_names]
    else:
        # MBPP style: assertions run only inside try/except blocks below
        calls = list(test_list)

    # Per-test execution blocks with individual SIGALRM timeouts
    for call in calls:
        parts.append(f"signal.alarm({timeout_seconds})")
        parts.append("try:")
        parts.append(f"    {call}")
        parts.append("    __passed += 1")
        parts.append("except TimeoutError:")
        parts.append("    if __error_msg is None:")
        parts.append("        __error_msg = 'TimeoutError'")
        if early_exit:
            parts.append("    signal.alarm(0)")
            parts.append("    __emit()")
            parts.append("    sys.exit(0)")
        parts.append("except SystemExit as __se:")
        if hack_guard:
            # Hack guard: ALL SystemExit treated as failure (blocks raise SystemExit(0))
            parts.append("    if __error_msg is None:")
            parts.append("        __error_msg = 'SystemExit'")
        else:
            parts.append("    if __se.code == 0 or __se.code is None:")
            parts.append("        __passed += 1")
            parts.append("    else:")
            parts.append("        if __error_msg is None:")
            parts.append("            __error_msg = 'SystemExit'")
            if early_exit:
                parts.append("        signal.alarm(0)")
                parts.append("        __emit()")
                parts.append("        sys.exit(0)")
        parts.append("except Exception as __e:")
        parts.append("    if __error_msg is None:")
        parts.append("        __error_msg = type(__e).__name__")
        if early_exit:
            parts.append("    signal.alarm(0)")
            parts.append("    __emit()")
            parts.append("    sys.exit(0)")
        parts.append("finally:")
        parts.append("    signal.alarm(0)")
        parts.append("")

    # Final output (reached when all tests run without early exit)
    parts.append("__emit()")

    return "\n".join(parts)


def run_batched_subprocess(
    script_content: str,
    total_timeout: int,
    total_count: int,
    hack_guard: bool = False,
) -> tuple[int, str | None]:
    """Run batched test script via stdin and parse JSON result.

    Args:
        script_content: Complete Python script to execute
        total_timeout: Maximum total execution time (outer safety net)
        total_count: Number of tests (used for exit-0-but-no-JSON fallback)
        hack_guard: If True, treat exit-0-without-JSON as failure (blocks
            ``raise SystemExit(0)`` bypassing the in-script hack guard)

    Returns:
        Tuple of (passed_count, error_msg)
    """
    def _sandbox_preexec():
        # 512MB memory cap — prevents OOM-killing the host
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        # Block all file writes at kernel level
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        # 1 process — prevents fork bombs
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))

    try:
        result = subprocess.run(
            ["python3", "-"],
            input=script_content,
            capture_output=True,
            text=True,
            timeout=max(total_timeout, 1),
            cwd=_SUBPROCESS_CWD,
            preexec_fn=_sandbox_preexec,
        )

        # Try to parse JSON from stdout (last non-empty line)
        stdout = result.stdout.strip()
        if stdout:
            # Take the last line in case the user code prints to stdout too
            last_line = stdout.rsplit("\n", 1)[-1]
            try:
                data = json.loads(last_line)
                return (
                    data.get("passed_count", 0),
                    _normalize_error_type(data.get("error_msg")),
                )
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: no JSON output — parse stderr
        if result.returncode != 0:
            error_type = parse_error_type(result.stderr.strip())
            return (0, error_type)

        # Process exited 0 but no JSON — e.g. sys.exit(0) at module level
        # or os._exit(0); matches original per-subprocess returncode==0 behavior.
        # With hack_guard, treat this as failure: direct ``raise SystemExit(0)``
        # bypasses the in-script function replacements and terminates the
        # process before any test blocks execute.
        if hack_guard:
            return (0, "SystemExit")
        return (total_count, None)

    except subprocess.TimeoutExpired:
        return (0, "timeout")
    except Exception as e:
        return (0, f"execution_error: {type(e).__name__}")


def _normalize_error_type(name: str | None) -> str | None:
    """Map exception class name from batched script to our error type strings.

    Matches the same mapping as parse_error_type() for consistency.
    """
    if name is None:
        return None

    mapping = {
        "SyntaxError": "syntax_error",
        "IndentationError": "indentation_error",
        "NameError": "name_error",
        "TypeError": "type_error",
        "ValueError": "value_error",
        "AttributeError": "attribute_error",
        "ImportError": "import_error",
        "ModuleNotFoundError": "import_error",
        "AssertionError": "assertion_error",
        "IndexError": "index_error",
        "KeyError": "key_error",
        "ZeroDivisionError": "zero_division_error",
        "RecursionError": "recursion_error",
        "MemoryError": "memory_error",
        "TimeoutError": "timeout",
    }
    return mapping.get(name, "runtime_error")


def parse_error_type(stderr: str) -> str:
    """Parse error type from stderr output.

    Args:
        stderr: Standard error output from subprocess

    Returns:
        Error type string
    """
    if not stderr:
        return "unknown_error"

    if "SyntaxError" in stderr:
        return "syntax_error"
    if "IndentationError" in stderr:
        return "indentation_error"
    if "NameError" in stderr:
        return "name_error"
    if "TypeError" in stderr:
        return "type_error"
    if "ValueError" in stderr:
        return "value_error"
    if "AttributeError" in stderr:
        return "attribute_error"
    if "ImportError" in stderr or "ModuleNotFoundError" in stderr:
        return "import_error"
    if "AssertionError" in stderr:
        return "assertion_error"
    if "IndexError" in stderr:
        return "index_error"
    if "KeyError" in stderr:
        return "key_error"
    if "ZeroDivisionError" in stderr:
        return "zero_division_error"
    if "RecursionError" in stderr:
        return "recursion_error"
    if "MemoryError" in stderr:
        return "memory_error"

    return "runtime_error"


def execute_hidden_tests(
    code: str,
    hidden_test_list: list[str],
    timeout_seconds: int = 5
) -> dict:
    """Execute hidden test cases for diagnostics only (no reward).

    Args:
        code: Generated Python code
        hidden_test_list: List of hidden test assertions
        timeout_seconds: Maximum execution time per test

    Returns:
        Dict with hidden_tests_all_passed only
    """
    all_passed, passed_count, total_count, _ = execute_code_with_tests(
        code, hidden_test_list, timeout_seconds, early_exit=True, hack_guard=True
    )
    return {"hidden_tests_all_passed": all_passed}


