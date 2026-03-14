import os, sys, importlib.util, traceback
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print('PROJECT_ROOT=', PROJECT_ROOT)
try:
    from ai.v2 import test_explainability as te
    print('normal import OK')
except Exception as e:
    print('normal import failed:', e)
    te_path = os.path.join(PROJECT_ROOT, 'ai', 'v2', 'test_explainability.py')
    print('te_path exists?', os.path.exists(te_path), te_path)
    if os.path.exists(te_path):
        spec = importlib.util.spec_from_file_location('ai_v2_test_explainability', te_path)
        te_mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(te_mod)
            print('file import OK')
            print('has HCCTModel?', hasattr(te_mod, 'HCCTModel'))
        except Exception:
            traceback.print_exc()
            print('file import failed')
    else:
        print('file not found')
