"""Offline structural smoke tests for the supplied educational project."""
import ast
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
EXPECTED={'Activity_01':1, 'Activity_02':1, 'Lab3':6, 'Lab5':6, 'Lab6':9, 'RAG-2':1}

def main():
    count=0
    for folder, target in EXPECTED.items():
        notebooks=list((ROOT/folder).glob('*.ipynb'))
        assert len(notebooks)==target, f'{folder}: expected {target}, got {len(notebooks)}'
        for path in notebooks:
            nb=json.loads(path.read_text(encoding='utf-8'))
            assert nb['nbformat']==4, f'{path}: invalid nbformat'
            source='\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
            ast.parse(source,filename=str(path))
            py=ROOT/'scripts'/path.relative_to(ROOT).with_suffix('.py')
            assert py.exists(),f'Missing script {py}'
            ast.parse(py.read_text(encoding='utf-8'),filename=str(py))
            count+=1
    csv_path=ROOT/'data/IMDB_sample.csv'
    with csv_path.open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)>=30
    assert {'positive','negative'}=={r['sentiment'] for r in rows}
    assert (ROOT/'RAG-2/ResearchPapers/DEMO_RAG_Overview.pdf').read_bytes().startswith(b'%PDF-')
    assert (ROOT/'.env.example').exists() and (ROOT/'.gitignore').exists()
    print(f'PASS: {count} notebooks and {count} .py scripts parse successfully.')
    print(f'PASS: {len(rows)} synthetic sentiment examples and demo PDF are present.')
    print('NOTE: Live LLM requests, training outcomes, and RAG downloads need your local setup.')

if __name__=='__main__': main()
