"""Step 0: read ARCD parquet files (pure-python reader) and build arcd_rows.json. Usage: python make_rows.py <dir with train/validation parquet>"""
import sys, os
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, '..', 'common')); os.chdir(_H)   # run from anywhere
from paths import P_ARCD, P_TYDI, STAGE3
import sys,json; from pq import read_parquet
d=sys.argv[1] if len(sys.argv)>1 else '.'; rows=[]
for sp in ['train','validation']:
    c=read_parquet(f'{d}/{sp}-00000-of-00001.parquet')
    for i in range(len(c['id'])):
        rows.append(dict(split=sp,id=c['id'][i],title=c['title'][i],context=c['context'][i],question=c['question'][i].strip(),ans=c['answers.text'][i][0],start=c['answers.answer_start'][i][0]))
json.dump(rows,open(P_ARCD,'w'),ensure_ascii=False); print(len(rows),'rows')
