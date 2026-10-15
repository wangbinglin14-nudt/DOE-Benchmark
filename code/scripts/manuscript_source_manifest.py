"""List current manuscript dependencies; optionally build a journal-upload ZIP.

Default is read-only inventory generation, with no duplicate manuscript tree.
The optional ZIP contains source, included tables, number macros and six PDF
figures. It does not claim to include the research data/code release.
"""
from pathlib import Path
import argparse, hashlib, json, re, zipfile

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'manuscripts'

def dependencies():
    found={}
    pending=[PAPER/'nature_cs_main.tex',PAPER/'supplementary.tex']
    while pending:
        p=pending.pop().resolve()
        p.relative_to(PAPER.resolve())
        if p in found:
            continue
        assert p.is_file(), str(p)
        found[p]=True
        if p.suffix=='.tex':
            text=p.read_text(encoding='utf-8')
            for target in re.findall(r'\\input\{([^}]+)\}|\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):
                pending.append(PAPER/(target[0] or target[1]))
    return sorted(found)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--zip-output',type=Path,help='Optional upload ZIP; refuses to overwrite.')
    args=parser.parse_args()
    files=dependencies()
    rows=[{'path':p.relative_to(ROOT).as_posix(),'archive_path':p.relative_to(PAPER).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    dest=ROOT/'project_docs/submission_readiness_20260927/MANUSCRIPT_SOURCE_MANIFEST.json'
    dest.write_text(json.dumps({'scope':'Manuscript sources only; not a data/code release','files':rows},indent=2),encoding='utf-8')
    if args.zip_output:
        with zipfile.ZipFile(args.zip_output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for p in files:
                archive.write(p,p.relative_to(PAPER).as_posix())
        with zipfile.ZipFile(args.zip_output) as archive:
            assert archive.testzip() is None
    print(f'{len(files)} source dependencies verified; ZIP '+('created.' if args.zip_output else 'not requested.'))

if __name__=='__main__': main()
