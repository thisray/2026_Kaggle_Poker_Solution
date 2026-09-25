import importlib
import inspect
from pathlib import Path
import sys
import time
import traceback


def main():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    modules = [
        path.stem
        for path in sorted(root.glob("test_*.py"))
        if path.stem != "test_runner"
    ]
    failures = []
    passed = 0
    started = time.monotonic()
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            failures.append(f"{module_name} [import]")
            print(f"FAIL {module_name} [import]")
            traceback.print_exc()
            continue
        tests = [
            (name, function)
            for name, function in inspect.getmembers(module, inspect.isfunction)
            if name.startswith("test_") and function.__module__ == module.__name__
        ]
        for name, function in tests:
            label = f"{module_name}.{name}"
            try:
                function()
            except Exception:
                failures.append(label)
                print(f"FAIL {label}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"PASS {label}")
    elapsed = time.monotonic() - started
    print(f"RESULT {passed} passed, {len(failures)} failed in {elapsed:.2f}s")
    if failures:
        print("FAILED TESTS")
        for label in failures:
            print(f"- {label}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
