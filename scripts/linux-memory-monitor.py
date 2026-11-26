#!/usr/bin/env python3
"""Live VIctory process, system, Vulkan memory, and frame monitor."""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import psutil
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

parser = argparse.ArgumentParser()
parser.add_argument('--pid', type=int, required=True)
parser.add_argument('--log', type=Path, required=True)
parser.add_argument('--target-gib', type=float, default=36)
args = parser.parse_args()

GIB = 1024 ** 3
fps_re = re.compile(r'([\d.]+) fps, ([\d.]+) ms/frame.*?, (\d+) draws/frame, dropped (\d+) without a host pipeline \((\d+) being made\) and (\d+) indirect')
card_re = re.compile(r"the card's own: budget (\d+) MB, in use (\d+) MB.*?PC's shared with the card: budget (\d+) MB, in use (\d+) MB")
copies_re = re.compile(r"host copies in the card's memory: (\d+) copies ([\d.]+) MB")

app = QApplication(sys.argv)
window = QMainWindow()
window.setWindowTitle('VIctory • Live system and memory monitor')
root = QWidget()
layout = QVBoxLayout(root)
window.setCentralWidget(root)
heading = QLabel('VICTORY  /  LIVE PERFORMANCE')
heading.setStyleSheet('font-size: 19px; font-weight: 700; color: #f19a38;')
layout.addWidget(heading)
status = QLabel('Reading the running game…')
layout.addWidget(status)
grid = QGridLayout()
layout.addLayout(grid)
widgets = {}
rows = [
    ('GAME', 'fps'), ('FRAME TIME', 'frame'), ('DRAWS / FRAME', 'draws'), ('DROPPED DRAWS', 'dropped'),
    ('PIPELINES STILL BUILDING', 'pending'), ('CPU • GAME PROCESS', 'proc_cpu'), ('CPU • WHOLE SYSTEM', 'sys_cpu'),
    ('GAME RAM • RESIDENT / VIRTUAL', 'ram'), ('GAME THREADS', 'threads'), ('SYSTEM RAM AVAILABLE', 'sys_ram'),
    ('RTX 3060 • VRAM USED / TOTAL', 'vram'), ('RTX 3060 • GPU LOAD / POWER', 'gpu'),
    ('GPU SHARED MEMORY • DRIVER REPORTED', 'shared'), ('SHARED GPU MEMORY BUDGET', 'shared_budget'),
    ('VICTORY BUFFERS • ALLOCATED IN VRAM', 'pool'), ('REQUESTED HOST POOL TARGET', 'target'),
]
for i, (name, key) in enumerate(rows):
    label = QLabel(name)
    label.setStyleSheet('color:#9aa6b2; font-size:10px; font-weight:600;')
    value = QLabel('—')
    value.setStyleSheet('font-size:14px; font-weight:600; color:#edf2f7;')
    grid.addWidget(label, i, 0)
    grid.addWidget(value, i, 1)
    widgets[key] = value

note = QLabel('Shared memory is driver-reported system RAM usage. The 36 GiB target is a test target, not a reserved or guaranteed unified pool.')
note.setWordWrap(True)
note.setStyleSheet('color:#e5bd79; font-size:11px; padding-top:8px;')
layout.addWidget(note)
window.setStyleSheet('QMainWindow,QWidget { background:#15191e; color:#edf2f7; }')
window.resize(520, 560)

log_offset = 0
last_frame = None
last_card = None
last_copies = None

def update():
    global log_offset, last_frame, last_card, last_copies
    try:
        with args.log.open('r', errors='replace') as f:
            f.seek(log_offset)
            for line in f:
                m = fps_re.search(line)
                if m:
                    last_frame = tuple(m.groups())
                m = card_re.search(line)
                if m:
                    last_card = tuple(map(int, m.groups()))
                m = copies_re.search(line)
                if m:
                    last_copies = (int(m.group(1)), float(m.group(2)))
            log_offset = f.tell()
    except OSError:
        pass
    try:
        proc = psutil.Process(args.pid)
        proc_cpu = proc.cpu_percent(None)
        rss, vms = proc.memory_info().rss, proc.memory_info().vms
        threads = proc.num_threads()
        widgets['proc_cpu'].setText(f'{proc_cpu:.0f}% of one core  •  {proc_cpu / psutil.cpu_count() :.1f}% of {psutil.cpu_count()} logical CPUs')
        widgets['ram'].setText(f'{rss / GIB:.1f} / {vms / GIB:.1f} GiB')
        widgets['threads'].setText(str(threads))
        status.setText(f'Game process {args.pid} is running  •  updated continuously')
    except psutil.Error:
        status.setText('Game process is not running')
    widgets['sys_cpu'].setText(f'{psutil.cpu_percent(None):.0f}% across {psutil.cpu_count()} logical CPUs')
    vm = psutil.virtual_memory()
    widgets['sys_ram'].setText(f'{vm.available / GIB:.1f} / {vm.total / GIB:.1f} GiB available / total')
    if last_frame:
        fps, frame, draws, dropped, pending, indirect = last_frame
        widgets['fps'].setText(f'{float(fps):.1f} FPS')
        widgets['frame'].setText(f'{float(frame):.0f} ms')
        widgets['draws'].setText(f'{int(draws):,}')
        widgets['dropped'].setText(f'{int(dropped):,} ordinary  •  {int(indirect):,} indirect')
        widgets['pending'].setText(f'{int(pending):,}')
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits'], text=True, timeout=1).strip().split(',')
        util, used, total, power = map(float, out)
        widgets['vram'].setText(f'{used/1024:.1f} / {total/1024:.1f} GiB')
        widgets['gpu'].setText(f'{util:.0f}%  •  {power:.0f} W')
    except Exception:
        pass
    if last_card:
        budget, used, shared_budget, shared_used = last_card
        widgets['shared'].setText(f'{shared_used/1024:.1f} GiB in use  •  {shared_budget/1024:.1f} GiB driver budget')
        widgets['shared_budget'].setText(f'{shared_budget/1024:.1f} GiB')
        widgets['pool'].setText(f'{used/1024:.1f} GiB RTX-local used  •  {budget/1024:.1f} GiB budget')
    if last_copies:
        widgets['pool'].setToolTip(f"VIctory reports {last_copies[0]} host buffer copies totaling {last_copies[1]:.0f} MiB in RTX-local memory")
    widgets['target'].setText(f'{args.target_gib:.0f} GiB requested for experiment  •  not reserved')

proc = psutil.Process(args.pid)
proc.cpu_percent(None)
psutil.cpu_percent(None)
timer = QTimer()
timer.timeout.connect(update)
timer.start(1000)
update()
window.show()
sys.exit(app.exec())
