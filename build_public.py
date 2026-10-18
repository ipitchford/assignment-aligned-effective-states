"""Rebuild the manuscript PDFs from Markdown; requires Pandoc and pdfLaTeX."""
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parent
for stem in ('paper', 'supplement'):
    subprocess.run(['pandoc', stem+'.md', '--standalone', '--number-sections',
                    '--toc', '-V', 'fontsize=11pt', '-V', 'papersize=a4',
                    '-V', 'geometry:margin=22mm', '-o', stem+'.tex'],
                   cwd=root, check=True)
    path = root / (stem+'.tex')
    text = path.read_text()
    text = text.replace('\\usepackage{longtable,booktabs,array}',
                        '\\usepackage{longtable,booktabs,array}\n'
                        '\\setlength{\\emergencystretch}{3em}')
    # Allow long content identifiers to wrap without changing their digits.
    text = re.sub(r'\\texttt\{([a-f0-9]{64})\}',
                  lambda m: r'\texttt{'+r'\allowbreak{}'.join(m[1][i:i+8] for i in range(0,64,8))+'}', text)
    path.write_text(text)
    for iteration in range(4):
        subprocess.run(['pdflatex', '-interaction=nonstopmode',
                        '-halt-on-error', stem+'.tex'], cwd=root, check=True)
