import os

# litellm fetches its model price map from GitHub at import time; use the bundled copy instead (offline and
# reproducible).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
