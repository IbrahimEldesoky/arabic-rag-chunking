import re
DIAC=re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]')
TOK=re.compile(r'[\u0621-\u064A\u0660-\u0669a-zA-Z0-9]+')
def normalize(s):
    s=DIAC.sub('',s).replace('\u0640','')
    s=re.sub('[إأآٱ]','ا',s); s=s.replace('ى','ي').replace('ة','ه').replace('ؤ','و').replace('ئ','ي')
    s=s.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩','0123456789'))
    return s.lower()
_STOP_RAW="""في من على الى إلى عن مع هذا هذه ذلك تلك هذان هؤلاء الذي التي الذين اللذان اللتان ما ماذا من متى اين أين كيف لماذا كم هل اي أي ايضا أيضا
و ثم او أو لكن بل لا لم لن ليس كان كانت يكون تكون كانوا كل بعض غير بين حتى عند عندما اذا إذا قد لقد انه إنه انها إنها ان أن إن كما حيث بعد قبل خلال منذ دون هو هي هم هن نحن انا أنا انت أنت هما فيه فيها منه منها به بها له لها لهم عنه عنها وقد كذلك التى الذى مما فيما عما بما لما اما أما اذ إذ ذات ذو يا بان بأن لأن لان ولا ولم ولن وما ومن وهو وهي وهذا وفي ومع وعلى والى وان وأن وإن
 اي أي اين أين بماذا لماذا اى كيف متي اسم ماهو ماهي"""
STOP=set(normalize(w) for w in _STOP_RAW.split())
PRE=['وال','بال','كال','فال','لل','ال','و']      # longest first
SUF=['ات','ان','ين','ون','ها','هم','هن','كم','نا','ه','ة','ي','ك']
def light_stem(w):
    if len(w)<4: return w
    for p in PRE:
        if w.startswith(p) and len(w)-len(p)>=3: w=w[len(p):]; break
    for s in SUF:
        if w.endswith(s) and len(w)-len(s)>=3: w=w[:-len(s)]; break
    return w
def tokens(text,mode):
    """mode: raw | norm | norm_stop | norm_stem | norm_stop_stem | char4"""
    if mode=='raw': return [t.lower() for t in TOK.findall(text)]
    t=TOK.findall(normalize(text))
    if mode=='norm': return t
    if 'stop' in mode: t=[x for x in t if x not in STOP]
    if 'stem' in mode: t=[light_stem(x) for x in t]
    if mode=='char4':
        out=[]
        for x in TOK.findall(normalize(text)):
            y='#'+x+'#'
            out+= [y[i:i+4] for i in range(max(1,len(y)-3))]
        return out
    return t
