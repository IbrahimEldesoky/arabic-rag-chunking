"""Merge the pickles written by run_experiments.py into res_all.pkl (input of analysis.py).
Usage: python merge_results.py res_norm_stop,norm_stop_stem.pkl res_raw,norm,norm_stem,char4.pkl"""
import sys, os, pickle
os.chdir(os.path.dirname(os.path.abspath(__file__)))
R = {}
for f in sys.argv[1:]: R.update(pickle.load(open(f, 'rb')))
pickle.dump(R, open('res_all.pkl', 'wb')); print(len(R), 'entries -> res_all.pkl')
