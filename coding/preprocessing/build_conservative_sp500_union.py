#!/usr/bin/env python3
"""Build a conservative TAQ-symbol union for S&P 500 preprocessing."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

REF=Path("reference/index_membership")
OUT=REF/"sp500_2018_2019_conservative_union_taq_symbols.txt"
META=REF/"sp500_2018_2019_conservative_union_metadata.json"
EVENTS=REF/"sp500_ticker_events_2018_2019.csv"
ALIASES={"BRK/B":"BRK B","BRK.B":"BRK B","BRK_B":"BRK B","BF/B":"BF B","BF.B":"BF B","BF_B":"BF B","GOOGL":"GOOG L"}
EVENT_ROWS=[
("PCLN","BKNG","2018-02-27","ticker_change","Priceline renamed Booking Holdings"),
("HCN","WELL","2018-02-28","ticker_change","Welltower ticker change"),
("CBG","CBRE","2018-03-20","ticker_change","CBRE Group ticker change"),
("LUK","JEF","2018-05-24","ticker_change","Leucadia renamed Jefferies Financial Group"),
("WR","EVRG","2018-06-05","merger_ticker_change","Westar/Evergy mapping"),
("DWDP","DD","2019-06-03","spin_or_rename","DowDuPont/DuPont mapping"),
("BMS","AMCR","2019-06-11","replacement_or_mapping","Bemis/Amcor mapping"),
("HRS","LHX","2019-07-01","merger_ticker_change","Harris/L3Harris mapping"),
("CBS","VIAC","2019-12-05","merger_ticker_change","CBS/ViacomCBS mapping"),
("BBT","TFC","2019-12-09","merger_ticker_change","BB&T/Truist mapping"),
("BCR","BDX","2018-01-02","acquisition_mapping","Bard/Becton Dickinson boundary"),
("TWX","FLT","2018-06-20","index_replacement_boundary","H1 public change log replacement"),
]
COLS={"symbol","source_symbol","ticker","source_ticker","taq_symbol","addition_symbol","removal_symbol","old_symbol","new_symbol"}

def norm(x:Any)->str:
    s=str(x or "").strip().upper()
    if not s or s in {"NAN","NONE","NULL"}: return ""
    s=ALIASES.get(s,s).replace("/"," ").replace("."," ").replace("_"," ")
    s=" ".join(s.split())
    return ALIASES.get(s,s)

def plausible(s:str)->bool:
    c=s.replace(" ","")
    return bool(c) and c.isalnum() and not any(ch.isdigit() for ch in c) and 1<=len(c)<=6

def add(symbols,sources,x,source):
    s=norm(x)
    if plausible(s): symbols.add(s); sources[s].add(source)

def ensure_events(path:Path):
    if path.exists(): return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.writer(h); w.writerow(["old_symbol","new_symbol","effective_date","event_type","source","source_note"])
        for old,new,d,t,note in EVENT_ROWS: w.writerow([old,new,d,t,"conservative_manual_seed",note])

def read_txt(path,symbols,sources):
    for line in path.read_text(encoding="utf-8-sig",errors="replace").splitlines():
        line=line.strip()
        if line and not line.startswith("#"): add(symbols,sources,line,path.as_posix())

def read_csv(path,symbols,sources):
    with path.open("r",encoding="utf-8-sig",errors="replace",newline="") as h:
        r=csv.DictReader(h); cols=[c for c in (r.fieldnames or []) if c.strip().lower() in COLS]
        for row in r:
            for c in cols: add(symbols,sources,row.get(c),f"{path.as_posix()}:{c}")

def in_window(obj,start,end):
    v=obj.get("effectiveDate") or obj.get("effective_date") or obj.get("date")
    if not v: return True
    try: d=date.fromisoformat(str(v)[:10])
    except ValueError: return True
    return start<=d<=end

def walk(x):
    if isinstance(x,dict):
        yield x
        for v in x.values(): yield from walk(v)
    elif isinstance(x,list):
        for v in x: yield from walk(v)

def read_json(path,symbols,sources,start,end):
    data=json.loads(path.read_text(encoding="utf-8-sig",errors="replace"))
    for obj in walk(data):
        if in_window(obj,start,end):
            for k in ("ticker","symbol","source_symbol"):
                if k in obj: add(symbols,sources,obj[k],f"{path.as_posix()}:{k}")

def build(reference_dir:Path,extra:list[Path],start:date,end:date):
    symbols=set(); sources=defaultdict(set); ensure_events(reference_dir/"sp500_ticker_events_2018_2019.csv")
    paths=[]
    for pat in ("public_sp500*.txt","public_sp500*.csv","sp500_*symbols*.txt","sp500_membership_intervals.csv","sp500_ticker_events_2018_2019.csv","sp500_ric_to_taq_crosswalk.csv"):
        paths+=sorted(reference_dir.glob(pat))
    sp=Path("SP500")
    if sp.exists(): paths+=sorted(sp.glob("*.csv"))+sorted(sp.glob("*.json"))
    paths+=extra; seen=[]
    for path in paths:
        path=path.resolve()
        if path in seen or not path.exists(): continue
        seen.append(path)
        if path.suffix.lower()==".txt": read_txt(path,symbols,sources)
        elif path.suffix.lower()==".csv": read_csv(path,symbols,sources)
        elif path.suffix.lower()==".json": read_json(path,symbols,sources,start,end)
    for old,new,*_ in EVENT_ROWS:
        add(symbols,sources,old,"conservative_manual_seed"); add(symbols,sources,new,"conservative_manual_seed")
    ordered=sorted(symbols)
    meta={"policy":"conservative_union_v1","index_id":"sp500","start":start.isoformat(),"end":end.isoformat(),"symbol_count":len(ordered),"over_inclusion_intentional":True,"usage":"Preprocessing and volume availability only; point-in-time membership controls evaluation eligibility.","source_files":[str(p) for p in seen],"manual_ticker_pairs":[{"old_symbol":a,"new_symbol":b,"effective_date":c,"event_type":d,"note":e} for a,b,c,d,e in EVENT_ROWS],"symbol_sources":{s:sorted(sources[s]) for s in ordered}}
    return ordered,meta

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--reference-dir",type=Path,default=REF); ap.add_argument("--extra-source",type=Path,action="append",default=[]); ap.add_argument("--start",default="2018-01-02"); ap.add_argument("--end",default="2019-12-31"); ap.add_argument("--out",type=Path,default=OUT); ap.add_argument("--metadata",type=Path,default=META)
    args=ap.parse_args(); symbols,meta=build(args.reference_dir,args.extra_source,date.fromisoformat(args.start),date.fromisoformat(args.end))
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text("\n".join(symbols)+"\n",encoding="utf-8"); args.metadata.write_text(json.dumps(meta,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({"out":str(args.out),"metadata":str(args.metadata),"symbol_count":len(symbols)},indent=2))
if __name__=="__main__": main()
