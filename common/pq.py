"""Minimal pure-python Parquet reader (SNAPPY/UNCOMPRESSED, PLAIN + RLE_DICTIONARY, nested lists)."""
import struct
from thrift_compact import R

def snappy(b):
    r=R(b); n=r.varint(); out=bytearray(); p=r.p
    while p<len(b):
        t=b[p]; p+=1; k=t&3
        if k==0:
            l=t>>2
            if l>=60:
                nb=l-59; l=int.from_bytes(b[p:p+nb],'little'); p+=nb
            l+=1; out+=b[p:p+l]; p+=l
        else:
            if k==1: l=((t>>2)&7)+4; o=((t>>5)<<8)|b[p]; p+=1
            elif k==2: l=(t>>2)+1; o=int.from_bytes(b[p:p+2],'little'); p+=2
            else: l=(t>>2)+1; o=int.from_bytes(b[p:p+4],'little'); p+=4
            for _ in range(l): out.append(out[-o])
    assert len(out)==n,(len(out),n)
    return bytes(out)

def hybrid(b,p,w,count):
    out=[]; r=R(b,p)
    while len(out)<count:
        h=r.varint()
        if h&1:
            g=h>>1; nb=g*w; chunk=int.from_bytes(b[r.p:r.p+nb],'little'); r.p+=nb
            m=(1<<w)-1
            for i in range(g*8): out.append((chunk>>(i*w))&m)
        else:
            c=h>>1; bw=(w+7)//8; v=int.from_bytes(b[r.p:r.p+bw],'little') if w else 0; r.p+=bw
            out+= [v]*c
    return out[:count], r.p

def plain_ba(b,n):
    out=[];p=0
    for _ in range(n):
        l=struct.unpack('<I',b[p:p+4])[0]; p+=4; out.append(b[p:p+l].decode('utf8')); p+=l
    return out

def plain_i32(b,n): return list(struct.unpack('<%di'%n,b[:4*n]))
def plain_i64(b,n): return list(struct.unpack('<%dq'%n,b[:8*n]))

def levels(schema):
    # returns {path_tuple:(max_def,max_rep,type)}
    res={}
    def walk(i,path,d,r):
        e=schema[i]; name=e[4].decode(); nc=e.get(5,0); rt=e.get(3,0)
        if i>0:
            path=path+(name,)
            if rt==1: d+=1
            elif rt==2: d+=1; r+=1
        if nc==0 and i>0: res[path]=(d,r,e.get(1)); return i+1
        j=i+1
        for _ in range(nc): j=walk(j,path,d,r)
        return j
    walk(0,(),0,0); return res

def read_column(b,cc,lv):
    md=cc[3]; path=tuple(x.decode() for x in md[3]); maxd,maxr,typ=lv[path]; codec=md[4]; nvals=md[5]
    start=md.get(11) or md[9]; if_dict=None; p=start; dictv=None; vals=[];defs=[];reps=[]
    total=0
    while total<nvals:
        r=R(b,p); h=r.struct(); p=r.p; csz=h[3]; usz=h[2]
        raw=b[p:p+csz]; p+=csz
        data=snappy(raw) if codec==1 else raw
        if h[1]==2:
            n=h[7][1]
            dictv=plain_ba(data,n) if typ==6 else (plain_i32(data,n) if typ==1 else plain_i64(data,n))
            continue
        dh=h[5]; n=dh[1]; enc=dh[2]; q=0
        rl=[0]*n; dl=[maxd]*n
        if maxr>0:
            ln=struct.unpack('<I',data[q:q+4])[0]; q+=4
            rl,_=hybrid(data,q,maxr.bit_length(),n); q+=ln
        if maxd>0:
            ln=struct.unpack('<I',data[q:q+4])[0]; q+=4
            dl,_=hybrid(data,q,maxd.bit_length(),n); q+=ln
        nn=sum(1 for x in dl if x==maxd)
        body=data[q:]
        if enc in(2,8):
            w=body[0]; idx,_=hybrid(body,1,w,nn); v=[dictv[i] for i in idx]
        elif enc==0:
            v=plain_ba(body,nn) if typ==6 else (plain_i32(body,nn) if typ==1 else plain_i64(body,nn))
        else: raise NotImplementedError(enc)
        vals+=v; defs+=dl; reps+=rl; total+=n
    return path,maxd,maxr,vals,defs,reps

def read_parquet(path):
    b=open(path,'rb').read(); n=struct.unpack('<I',b[-8:-4])[0]
    m=R(b,len(b)-8-n).struct(); lv=levels(m[2]); cols={}
    for rg in m[4]:
        for cc in rg[1]:
            pth,maxd,maxr,vals,defs,reps=read_column(b,cc,lv)
            name=pth[0]
            if maxr==0:
                it=iter(vals); col=[next(it) if d==maxd else None for d in defs]
            else:
                col=[]; it=iter(vals)
                for d,r in zip(defs,reps):
                    v=next(it) if d==maxd else None
                    if r==0: col.append([v])
                    else: col[-1].append(v)
            cols.setdefault(name,[]).extend(col) if name not in cols or maxr==0 or True else None
            if name=='answers': cols[name+'.'+pth[1]]=cols.pop(name)
    return cols
