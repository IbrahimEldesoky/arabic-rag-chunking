"""Shared locations. Every script imports this via the bootstrap line at its top."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA = os.path.join(ROOT, 'data')
P_ARCD = os.path.join(DATA, 'arcd_rows.json')
P_TYDI = os.path.join(DATA, 'tydi_rows.json')
STAGE3 = os.path.join(ROOT, 'stage3_dense_fusion')
