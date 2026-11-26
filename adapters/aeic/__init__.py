"""AEIC adapter. Importing it puts AEIC's code (third_party/aeic, unchanged) on sys.path: AEIC's files import each
other both as `src.codec...` and as `codec...` / `base_model` / `my_utils...`."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AEIC_ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "third_party", "aeic")


def add_third_party_paths():
    for p in (os.path.join(AEIC_ROOT, "src"), AEIC_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)


add_third_party_paths()
