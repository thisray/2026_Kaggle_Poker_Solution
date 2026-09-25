"""Read the supplied flat, dictionary/Snappy Parquet without extra installs.
Only the encodings in this research attachment are supported; production uses
pandas.read_parquet with pyarrow. All unsupported constructs fail explicitly.
"""
from pathlib import Path
import struct, ctypes, ctypes.util
import numpy as np
from thrift.protocol import TCompactProtocol
from thrift.transport import TTransport
from thrift.Thrift import TType

def thrift_obj(data):
    tr=TTransport.TMemoryBuffer(data); p=TCompactProtocol.TCompactProtocol(tr)
    def value(t):
        if t==TType.STRUCT:
            r={};p.readStructBegin()
            while True:
                _,tt,i=p.readFieldBegin()
                if tt==TType.STOP: break
                r[i]=value(tt);p.readFieldEnd()
            p.readStructEnd();return r
        if t==TType.LIST:
            tt,n=p.readListBegin();r=[value(tt) for _ in range(n)];p.readListEnd();return r
        if t==TType.MAP:
            kt,vt,n=p.readMapBegin();r={value(kt):value(vt) for _ in range(n)};p.readMapEnd();return r
        return {TType.BOOL:p.readBool,TType.BYTE:p.readByte,TType.I16:p.readI16,TType.I32:p.readI32,TType.I64:p.readI64,TType.DOUBLE:p.readDouble,TType.STRING:p.readBinary}[t]()
    r=value(TType.STRUCT);return r,tr._buffer.tell()

def varint(b,pos):
    x=0;s=0
    while True:
        v=b[pos];pos+=1;x|=(v&127)<<s
        if v<128:return x,pos
        s+=7

def rle(b,bits,n):
    out=[];pos=0;mask=(1<<bits)-1
    while len(out)<n:
        head,pos=varint(b,pos)
        if head&1:
            count=(head>>1)*8;size=count*bits//8
            x=int.from_bytes(b[pos:pos+size],'little');pos+=size
            out.extend((x>>(j*bits))&mask for j in range(count))
        else:
            count=head>>1;size=(bits+7)//8
            x=int.from_bytes(b[pos:pos+size],'little');pos+=size
            out.extend([x]*count)
    return np.array(out[:n],dtype=np.int64)

def plain(b,typ,n):
    dtypes={1:'<i4',2:'<i8',4:'<f4',5:'<f8'}
    if typ in dtypes:return np.frombuffer(b,dtype=dtypes[typ],count=n).copy()
    if typ==0:return np.unpackbits(np.frombuffer(b,dtype=np.uint8),bitorder='little')[:n].astype(bool)
    if typ==6:
        pos=0;r=[]
        for _ in range(n):
            size=struct.unpack_from('<I',b,pos)[0];pos+=4;r.append(b[pos:pos+size].decode());pos+=size
        return np.array(r,object)
    raise NotImplementedError(('plain type',typ))

def read_flat(path):
    b=Path(path).read_bytes();assert b[:4]==b[-4:]==b'PAR1'
    size=struct.unpack('<I',b[-8:-4])[0];meta,_=thrift_obj(b[-8-size:-8])
    snappy=ctypes.CDLL(ctypes.util.find_library('snappy'))
    snappy.snappy_uncompress.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t)]
    def decomp(raw,codec,size):
        if codec==0:return raw
        if codec!=1:raise NotImplementedError(('codec',codec))
        out=ctypes.create_string_buffer(size);length=ctypes.c_size_t(size)
        assert snappy.snappy_uncompress(raw,len(raw),out,ctypes.byref(length))==0
        assert length.value==size;return out.raw[:size]
    schemas={x[4].decode():x for x in meta[2][1:]};cols={}
    for rg in meta[4]:
        for c in rg[1]:
            m=c[3];name=m[3][0].decode();typ=m[1];n=m[5];codec=m[4]
            pos=min(m.get(11,m[9]),m[9]);got=[];dictionary=None
            while sum(len(x) for x in got)<n:
                h,hl=thrift_obj(b[pos:]);pos+=hl
                raw=decomp(b[pos:pos+h[3]],codec,h[2]);pos+=h[3]
                if h[1]==2:
                    dh=h[7];assert dh[2]==0;dictionary=plain(raw,typ,dh[1]);continue
                if h[1]!=0:raise NotImplementedError(('page',h))
                dh=h[5];nn=dh[1];enc=dh[2];offset=0
                assert schemas[name].get(3)==1,('not optional flat',schemas[name])
                dlen=struct.unpack_from('<I',raw,offset)[0];offset+=4
                defs=rle(raw[offset:offset+dlen],1,nn);offset+=dlen
                nv=int(defs.sum())
                if enc in (2,8):
                    assert dictionary is not None;bits=raw[offset];v=dictionary[rle(raw[offset+1:],bits,nv)]
                elif enc==0:v=plain(raw[offset:],typ,nv)
                else:raise NotImplementedError(('encoding',enc))
                if nv<nn:
                    arr=np.full(nn,np.nan,dtype=object if typ==6 else float);arr[defs==1]=v;v=arr
                got.append(v)
            cols.setdefault(name,[]).append(np.concatenate(got))
    import pandas as pd
    df=pd.DataFrame({k:np.concatenate(v) for k,v in cols.items()});assert len(df)==meta[3]
    return df
if __name__=='__main__':
    import sys
    d=read_flat(sys.argv[1]);print(d.shape);print(d.dtypes);print(d.head().to_string())
    if len(sys.argv)>2:d.to_csv(sys.argv[2],index=False)
