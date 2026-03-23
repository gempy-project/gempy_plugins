from gempy_plugins.optional_dependencies import require_flopy
flopy = require_flopy()
print(f"FloPy version: {flopy.__version__}")