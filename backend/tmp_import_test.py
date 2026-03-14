import importlib, traceback

try:
    importlib.import_module('ai.v2.test_explainability')
    print('import OK')
except Exception:
    traceback.print_exc()
    print('IMPORT_FAILED')
