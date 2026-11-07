#!/usr/bin/env python3
"""Check derived data integrity, resolved TeX and the actual main-text limit."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    provenance = json.loads((PROJECT / 'generated/provenance.json').read_text())
    for relative, expected in provenance['generated_files_sha256'].items():
        assert digest(PROJECT / relative) == expected, relative
    assert digest(PROJECT / 'scripts/export_results.py') == provenance['exporter_sha256']
    assert provenance['gpu_worker_count'] == 124
    assert provenance['byte_matched'] is False
    assert provenance['independent_confirmation'] is False
    assert provenance['original_six_mechanisms_active'] is False
    bib = (PROJECT / 'references.bib').read_text()
    bibkeys = set(re.findall(r'@\w+\{([^,]+),', bib))
    cited = set()
    for source in PROJECT.rglob('*.tex'):
        if 'build' in source.parts:
            continue
        for group in re.findall(r'\\cite\w*\{([^}]+)\}', source.read_text()):
            cited.update(k.strip() for k in group.split(','))
    assert cited <= bibkeys, cited - bibkeys
    report = {'scope': 'document_only_no_new_gpu_inference',
              'producer_revision': provenance['producer_sha'],
              'numerical_provenance_sha256': digest(PROJECT / 'generated/provenance.json'),
              'all_generated_numerical_hashes_valid': True,
              'citation_count': len(cited), 'documents': {}}
    bad = (r'Undefined control sequence', r'LaTeX Error:',
           r'Citation .* undefined', r'Reference .* undefined',
           r'There were undefined', r'Label\(s\) may have changed',
           r'Empty bibliography', r'Fatal error')
    for name in ('main', 'appendix', 'fox_response'):
        pdf = PROJECT / 'build' / f'{name}.pdf'
        log = (PROJECT / 'build' / f'{name}.log').read_text()
        for pattern in bad:
            assert not re.search(pattern, log), (name, pattern)
        overfull = [float(v) for v in re.findall(r'Overfull \\hbox \(([\d.]+)pt too wide\)', log)]
        assert not overfull or max(overfull) < 2.0, (name, overfull)
        assert not re.search(r'Overfull \\vbox', log), name
        info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
        pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M).group(1))
        text = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-'], text=True)
        assert 'EviCastKV' in text, name
        assert not re.search(r'\b(TBD|TODO|FIXME)\b', text), name
        report['documents'][name] = {'pdf_sha256': digest(pdf), 'pages': pages,
                                      'max_overfull_hbox_pt': max(overfull, default=0)}
        if name == 'main':
            assert '119.77' in text and '41.14' in text
            aux = (PROJECT / 'build/main.aux').read_text()
            label = re.search(r'\\newlabel\{main-text-end\}\{\{[^}]*\}\{(\d+)\}', aux)
            assert label, 'No main-text end label'
            end_page = int(label.group(1))
            def heading_lines(page: str) -> list[str]:
                # The official template has line numbers and spaced small caps.
                return [re.sub(r'[^A-Za-z]', '', re.split(r' {3,}',
                        re.sub(r'^\s*\d+\s+', '', line).strip())[0]).lower()
                        for line in page.splitlines()]
            reference_pages = [i for i, page in enumerate(text.split('\f'), 1)
                               if 'references' in heading_lines(page)]
            assert len(reference_pages) == 1, reference_pages
            body_pages = max(end_page, reference_pages[0] - 1)
            assert body_pages <= 10, body_pages
            disclosure_pages = [i for i, page in enumerate(text.split('\f'), 1)
                                if 'aiusagedisclosure' in heading_lines(page)]
            assert len(disclosure_pages) == 1 and disclosure_pages[0] > reference_pages[0]
            assert 'Prof. Fox' not in text, 'Named private response leaked into anonymous paper'
            report['documents'][name]['main_text_pages'] = body_pages
            report['documents'][name]['ai_disclosure_present'] = True
    report['document_checks_passed'] = True
    report['empirical_limits_resolved_by_writing'] = False
    report['document_source_sha256'] = {
        str(path.relative_to(PROJECT)): digest(path)
        for path in sorted(PROJECT.rglob('*'))
        if path.is_file() and 'build' not in path.relative_to(PROJECT).parts
        and path.suffix in ('.tex', '.bib', '.py', '.sh', '.md')
    }
    combined = PROJECT / 'build/complete_reading_copy.pdf'
    if combined.exists():
        combined_info = subprocess.check_output(['pdfinfo', str(combined)], text=True)
        combined_pages = int(re.search(r'^Pages:\s+(\d+)', combined_info, re.M).group(1))
        assert combined_pages == sum(v['pages'] for v in report['documents'].values())
        report['complete_reading_copy'] = {'pages': combined_pages,
                                           'pdf_sha256': digest(combined),
                                           'contains_named_private_response': True}
    output = PROJECT / 'build/validation.json'
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'document_source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
