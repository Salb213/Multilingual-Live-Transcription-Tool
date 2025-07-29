import os
from app.gui.main_window import run

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")

if __name__ == "__main__":
    run()