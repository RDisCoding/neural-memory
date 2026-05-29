"""
sandbox.py
Secure code execution sandbox with full HumanEval canonical test parser.

HumanEval test format:
    def check(candidate):
        assert candidate(arg) == expected
        assert candidate(arg2) == expected2

We extract each assert, run it with the generated code, and count
passed assertions for partial-credit reward computation.
"""

import ast
import os
import re
import subprocess
import tempfile
import textwrap
from typing import Tuple, List, Dict


# ─────────────────────────────────────────────────────────────
# FULL HUMANEVAL CANONICAL TEST PARSER
# ─────────────────────────────────────────────────────────────

def parse_humaneval_asserts(check_fn_src: str) -> List[str]:
    """
    Extract individual assert statements from a HumanEval check() function.

    Input (raw test field from HumanEval dataset):
        def check(candidate):
            assert candidate([1,2,3]) == [3,2,1]
            assert candidate([]) == []

    Returns list of assert expressions with 'candidate' substituted
    by the placeholder '__fn__':
        ["assert __fn__([1,2,3]) == [3,2,1]",
         "assert __fn__([]) == []"]
    """
    try:
        tree = ast.parse(textwrap.dedent(check_fn_src))
    except SyntaxError:
        return []

    asserts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assert):
                    # Unparse back to source (Python 3.9+)
                    try:
                        src = ast.unparse(stmt)
                    except AttributeError:
                        # Fallback for older Python: extract from raw source
                        src = _extract_assert_from_source(
                            check_fn_src, stmt.lineno
                        )
                    # Rename 'candidate' → '__fn__'
                    src = src.replace("candidate", "__fn__")
                    asserts.append(src)
    return asserts


def _extract_assert_from_source(src: str, lineno: int) -> str:
    """Fallback: grab the assert line directly from source."""
    lines = src.splitlines()
    if 0 < lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def build_test_script(generated_code: str,
                      entry_point: str,
                      asserts: List[str]) -> str:
    """
    Build a self-contained Python script that:
    1. Defines the generated function
    2. Runs each assert individually
    3. Prints PASS/FAIL for each assert
    4. Exits with the pass count
    """
    lines = [
        "import sys, traceback",
        "",
        "# ── Generated code ──",
        generated_code,
        "",
        f"__fn__ = {entry_point}",
        "",
        "results = []",
    ]

    for i, assert_stmt in enumerate(asserts):
        safe = assert_stmt.replace("\\", "\\\\").replace('"', '\\"')
        lines += [
            f"try:",
            f"    {assert_stmt}",
            f"    results.append(('PASS', {i}))",
            f"except Exception as e:",
            f"    results.append(('FAIL', {i}))",
        ]

    lines += [
        "",
        "passed = sum(1 for r in results if r[0] == 'PASS')",
        "total  = len(results)",
        "for status, idx in results:",
        "    print(f'  Assert {idx+1}/{total}: {status}')",
        "print(f'__RESULT__ {passed} {total}')",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# MAIN SANDBOX ENTRY POINT
# ─────────────────────────────────────────────────────────────

def execute_humaneval(generated_code: str,
                      problem: Dict,
                      timeout: int = 10) -> Tuple[float, str]:
    """
    Full HumanEval execution with canonical check() parser.

    Args:
        generated_code: Python code string produced by the model
        problem: dict with keys 'test' (check fn src) and 'entry_point'
        timeout: seconds before TIMEOUT penalty

    Returns:
        (reward: float, feedback: str)
        reward ∈ [-0.1, 1.0]
            -0.1  → timeout (infinite loop penalty)
             0.0  → syntax error or all tests fail
             0.x  → partial credit (x/total tests pass)
             1.0  → all tests pass
    """
    # Step 1: Syntax check before running subprocess
    try:
        ast.parse(generated_code)
    except SyntaxError as e:
        return 0.0, f"SyntaxError: {e}"

    # Step 2: Parse asserts from check() function
    asserts = parse_humaneval_asserts(problem.get("test", ""))
    if not asserts:
        # Fallback: run entire check block as a single test
        return _execute_full_check_block(generated_code, problem, timeout)

    entry_point = problem.get("entry_point", "solution")

    # Step 3: Build and write the test script
    script = build_test_script(generated_code, entry_point, asserts)

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "solution.py")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(script)

        try:
            result = subprocess.run(
                ["python", fpath],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir
            )
        except subprocess.TimeoutExpired:
            return -0.1, f"TIMEOUT after {timeout}s (possible infinite loop)"
        except Exception as e:
            return 0.0, f"Subprocess error: {e}"

    stdout = result.stdout
    stderr = result.stderr

    # Step 4: Parse __RESULT__ line
    match = re.search(r"__RESULT__ (\d+) (\d+)", stdout)
    if not match:
        # Runtime error — no __RESULT__ line
        err_preview = stderr[:400] if stderr else "(no stderr)"
        return 0.0, f"RuntimeError:\n{err_preview}"

    passed = int(match.group(1))
    total  = int(match.group(2))
    reward = passed / total if total > 0 else 0.0

    feedback_lines = [
        line for line in stdout.splitlines()
        if not line.startswith("__RESULT__")
    ]
    feedback = "\n".join(feedback_lines)
    feedback += f"\n→ {passed}/{total} assertions passed (reward={reward:.2f})"

    return reward, feedback


def _execute_full_check_block(generated_code: str,
                               problem: Dict,
                               timeout: int) -> Tuple[float, str]:
    """
    Fallback: run the entire check() function as-is.
    Used when assert parsing returns nothing (malformed test field).
    Returns 1.0 if no exception, 0.0 if any exception.
    """
    entry_point = problem.get("entry_point", "solution")
    check_src   = problem.get("test", "")
    script = (
        f"{generated_code}\n\n"
        f"{check_src}\n\n"
        f"check({entry_point})\n"
        f"print('__RESULT__ 1 1')\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "solution.py")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(script)
        try:
            r = subprocess.run(
                ["python", fpath],
                capture_output=True, text=True,
                timeout=timeout, cwd=tmpdir
            )
        except subprocess.TimeoutExpired:
            return -0.1, "TIMEOUT"
        except Exception as e:
            return 0.0, str(e)

    if "__RESULT__ 1 1" in r.stdout:
        return 1.0, "Full check() passed"
    return 0.0, f"check() failed:\n{r.stderr[:300]}"


def execute_code_safely(code: str,
                        test_cases: list,
                        timeout: int = 10) -> Tuple[float, str]:
    """
    Generic sandbox for non-HumanEval test cases.
    test_cases: list of {"input": "fn(args)", "expected": "repr(value)"}
    Returns (reward ∈ [-0.1, 1.0], feedback string)
    """
    try:
        ast.parse(code)
    except SyntaxError as e:
        return 0.0, f"SyntaxError: {e}"

    passed = 0
    feedback = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, test in enumerate(test_cases):
            src = (
                f"{code}\n\n"
                f"_result = {test['input']}\n"
                f"print(repr(_result))\n"
            )
            fpath = os.path.join(tmpdir, f"sol_{i}.py")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(src)

            try:
                r = subprocess.run(
                    ["python", fpath],
                    capture_output=True, text=True,
                    timeout=timeout, cwd=tmpdir
                )
                actual = r.stdout.strip()
                if actual == test["expected"].strip():
                    passed += 1
                    feedback.append(f"Test {i+1}: PASS")
                else:
                    feedback.append(
                        f"Test {i+1}: FAIL  got={actual!r}  "
                        f"want={test['expected']!r}"
                    )
            except subprocess.TimeoutExpired:
                feedback.append(f"Test {i+1}: TIMEOUT")
                return -0.1, "\n".join(feedback)
            except Exception as e:
                feedback.append(f"Test {i+1}: ERROR {e}")

    reward = passed / len(test_cases) if test_cases else 0.0
    return reward, "\n".join(feedback)


# ─────────────────────────────────────────────────────────────
# SMOKE TESTS
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running sandbox smoke tests...\n")

    # Test 1: correct code
    good = "def reverse_string(s):\n    return s[::-1]"
    tests = [{"input": "reverse_string('hello')", "expected": "'olleh'"}]
    r, fb = execute_code_safely(good, tests)
    assert r == 1.0, f"Expected 1.0, got {r}"
    print(f"  PASS correct code: reward={r}")

    # Test 2: wrong code
    wrong = "def reverse_string(s):\n    return s"
    r2, _ = execute_code_safely(wrong, tests)
    assert r2 == 0.0, f"Expected 0.0, got {r2}"
    print(f"  PASS wrong code:   reward={r2}")

    # Test 3: timeout
    loop = "def reverse_string(s):\n    while True: pass"
    r3, _ = execute_code_safely(loop, tests, timeout=2)
    assert r3 == -0.1, f"Expected -0.1, got {r3}"
    print(f"  PASS timeout:      reward={r3}")

    # Test 4: HumanEval parser
    fake_problem = {
        "entry_point": "add",
        "test": (
            "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
            "    assert candidate(0, 0) == 0\n"
            "    assert candidate(-1, 1) == 0\n"
        )
    }
    add_code = "def add(a, b):\n    return a + b"
    r4, fb4 = execute_humaneval(add_code, fake_problem)
    assert r4 == 1.0, f"Expected 1.0, got {r4}\n{fb4}"
    print(f"  PASS HumanEval full parser: reward={r4}")

    # Test 5: partial credit — passes assert(0,0)==0 but fails the other two
    add_broken = "def add(a, b):\n    return a * b"  # 1*2=2!=3 FAIL, 0*0=0 PASS, -1*1=-1!=0 FAIL
    r5, fb5 = execute_humaneval(add_broken, fake_problem)
    assert 0.0 < r5 < 1.0, f"Expected partial credit, got {r5}\n{fb5}"
    print(f"  PASS partial credit: reward={r5:.2f}")

    print("\nAll sandbox tests passed.")
