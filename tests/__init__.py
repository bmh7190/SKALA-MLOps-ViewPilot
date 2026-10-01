"""테스트는 운영 DB·모델을 건드리지 않는 임시 경로에서 실행한다."""
import atexit
import os
import tempfile

_runtime = tempfile.TemporaryDirectory(prefix="viewpilot-tests-")
os.environ["VIEWPILOT_RUNTIME_DIR"] = _runtime.name
os.environ["AUTO_RETRAIN"] = "false"
atexit.register(_runtime.cleanup)
