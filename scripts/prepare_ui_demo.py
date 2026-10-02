"""Build display metadata from test inputs. Never change labels or evaluation records.

Run with --register to add four explicitly named pending UI-demo forecasts to localhost.
These copies are for UI demonstration, NOT independent model evaluation.
"""
import argparse
import csv
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'serving_app' / 'static'
parser = argparse.ArgumentParser()
parser.add_argument('--register', action='store_true')
args = parser.parse_args()
metadata = {}
chosen = {}
for filename in ['test_normal.csv', 'test_drift.csv']:
    with (ROOT/'data'/'synthetic'/filename).open() as handle:
        for row in csv.DictReader(handle):
            metadata[row['video_id']] = {
                'category': row['category'],
                'subscriber_count_at_publish': int(row['subscriber_count_at_publish']),
                'source': f'data/synthetic/{filename}',
            }
            if filename == 'test_normal.csv':
                chosen.setdefault(row['category'], row)

demo_rows = []
for category, row in chosen.items():
    copy = {k:v for k,v in row.items() if k != 'target_views_day7'}
    copy['video_id'] = f'ui_demo_pending_{category}'
    metadata[copy['video_id']] = {**metadata[row['video_id']], 'source_video_id': row['video_id'], 'ui_demo': True}
    demo_rows.append(copy)
(STATIC/'video-metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')

themes = {
 'gaming': ('#15182e','#8a55ff','GAME GUIDE','LEVEL UP','게임 설정 가이드'),
 'education': ('#0b3042','#49dbb8','LEARN SMART','PYTHON 101','처음 배우는 자동화'),
 'entertainment': ('#3d124b','#ffac51','ON AIR','BEST MOMENTS','이번 주 하이라이트'),
 'lifestyle': ('#254d44','#e6c477','DAILY LIFE','MY ROUTINE','일상을 바꾸는 루틴'),
}
assets = STATIC/'demo-thumbnails'
assets.mkdir(exist_ok=True)
for cat,(bg,accent,kicker,title,sub) in themes.items():
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">
<rect width="640" height="360" rx="20" fill="{bg}"/>
<circle cx="550" cy="135" r="155" fill="{accent}" opacity=".24"/>
<circle cx="540" cy="230" r="90" fill="{accent}" opacity=".35"/>
<rect x="36" y="35" width="190" height="32" rx="8" fill="{accent}"/>
<g font-family="Arial,Apple SD Gothic Neo,sans-serif"><text x="48" y="58" font-size="20" font-weight="700" fill="{bg}">{kicker}</text>
<text x="34" y="160" font-size="47" font-weight="800" fill="white">{title}</text>
<text x="36" y="218" font-size="31" font-weight="700" fill="white">{sub}</text>
<text x="36" y="322" font-size="20" fill="white" opacity=".8">VIEWPILOT · DEMO ARTWORK</text></g>
<rect x="526" y="290" width="78" height="36" rx="7" fill="{bg}"/><text x="537" y="316" font-family="Arial" font-size="23" fill="white">DEMO</text>
</svg>'''
    (assets/f'{cat}.svg').write_text(svg)
print(f'Prepared {len(metadata)} metadata records and 4 demo artwork files.')
if args.register:
    for row in demo_rows:
        req=Request('http://127.0.0.1:8000/api/v1/videos/predictions',
                    data=json.dumps(row).encode(),headers={'Content-Type':'application/json'},method='POST')
        with urlopen(req,timeout=60) as response:
            result=json.load(response)
        print(row['video_id'],result['predicted_views_day7'])
