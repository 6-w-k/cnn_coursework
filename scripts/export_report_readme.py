"""Convert the current edited Word report into a GitHub Markdown report.

Usage: python scripts/export_report_readme.py REPORT.docx OUTPUT_DIRECTORY
Requires python-docx. Preserves document-order text, tables, figures and equations.
"""
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote
from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table


def math_tex(node):
    name = node.tag.split('}')[-1]
    children = list(node)
    def part(tag):
        found = node.find(qn('m:' + tag))
        return math_tex(found) if found is not None else ''
    if name == 't':return (node.text or '').replace('×', r'\times ').replace('Φ', r'\Phi ').replace('η', r'\eta ').replace('π', r'\pi ').replace('·', r'\cdot ')
    if name == 'f':return r'\frac{' + part('num') + '}{' + part('den') + '}'
    if name == 'sSub':return '{' + part('e') + '}_{' + part('sub') + '}'
    if name == 'sSup':return '{' + part('e') + '}^{' + part('sup') + '}'
    if name.endswith('Pr'):return ''
    return ''.join(math_tex(c) for c in children)


def export(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    images = output / 'docs/images';images.mkdir(parents=True, exist_ok=True)
    doc = Document(source)
    body, headings, summary = [], [], []
    started = False
    image_count = 0
    table_count = 0
    equation_count = 0
    skip_toc = False
    for element in doc.element.body:
        if element.tag == qn('w:p'):
            p = Paragraph(element, doc)
            text = p.text.strip()
            if p.style.name == 'Heading 1' and text == '目录':skip_toc = True;continue
            if p.style.name == 'Heading 1' and re.match(r'^\d+\s', text):
                started = True;skip_toc = False
            if not started:
                if not skip_toc and len(text) > 150:summary.append(text)
                continue
            if p.style.name.startswith('Heading'):
                level = int(p.style.name.split()[-1]) + 1
                anchor = 'section-' + str(len(headings) + 1)
                if level == 2:headings.append((text, anchor))
                body.append(f'<a id="{anchor}"></a>\n\n' + '#' * level + ' ' + text if level == 2 else '#' * level + ' ' + text)
            elif text:
                # Clarify percentage points without altering the submitted Word file.
                text = text.replace('牺牲 0.52% 的测试准确率', '牺牲 0.52 个百分点的测试准确率')
                if text == '图1 训练集示例':text = '图1 CIFAR-10 数据集示例'
                body.append(text.replace('\n', '  \n'))
            for drawing in element.iter(qn('a:blip')):
                relation = drawing.get(qn('r:embed'))
                if not relation:continue
                image_count += 1
                part = doc.part.related_parts[relation]
                suffix = Path(str(part.partname)).suffix
                filename = f'figure-{image_count:02d}{suffix}'
                (images / filename).write_bytes(part.blob)
                body.append(f'![实验报告图 {image_count}](docs/images/{filename})')
            for formula in element.iter(qn('m:oMath')):
                body.append('$$\n' + math_tex(formula) + '\n$$')
                equation_count += 1
        elif element.tag == qn('w:tbl') and started:
            table_count += 1
            t = Table(element, doc)
            rows = [[c.text.replace('\n', '<br>').replace('|', r'\|').strip() for c in row.cells] for row in t.rows]
            if rows:
                lines = ['| ' + ' | '.join(rows[0]) + ' |', '| ' + ' | '.join(['---'] * len(rows[0])) + ' |']
                lines.extend('| ' + ' | '.join(row) + ' |' for row in rows[1:])
                body.append('\n'.join(lines))
    intro = ['# 基于 ResNet-18 的图像分类结构改进与实验分析',
             'CIFAR-10 · ResNet-18 · 深度可分离卷积 · SE 注意力 · GELU / Mish',
             '[运行说明](#run) · [数据集与样例](#dataset) · [实验输出](results/formal_75f7c238/RESULTS.md) · [修改后的 Word 报告](docs/report.docx)',
             '以下报告正文从当前修改后的 Word 文件提取；目录使用网页锚点重新生成。',
             '## 摘要', *summary, '## 目录']
    intro.append('\n'.join(f'- [{title}](#{anchor})' for title, anchor in headings))
    (output / 'README.md').write_text('\n\n'.join(intro + body) + '\n', encoding='utf-8')
    manifest = {'source_filename':source.name, 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'images':image_count, 'tables':table_count, 'equations':equation_count,
                'chapters':[h[0] for h in headings],
                'adaptations':['Web anchor contents replace Word page-number contents',
                               '0.52% accuracy decrease clarified as 0.52 percentage points',
                               'Dataset example caption generalized to avoid an incorrect split claim']}
    (output / 'docs/report_conversion.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2), encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=True))


if __name__ == '__main__':export(*sys.argv[1:3])
