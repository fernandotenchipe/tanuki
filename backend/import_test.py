import importlib, traceback, sys
try:
    importlib.import_module('app')
    print('IMPORT_OK')
except Exception:
    traceback.print_exc()
    sys.exit(1)
