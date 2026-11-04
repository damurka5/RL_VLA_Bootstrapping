"""Pasteable model repair; run from the repository root, then remeasure grips."""

from pathlib import Path
import re
import xml.etree.ElementTree as ET

p = Path("robots/widowx200/widowx200_mujoco/wx200.xml")
original = p.read_text()
text = original
options = ET.fromstring(text).findall("option")
if not options:
    marker = '<compiler autolimits="true" angle="radian"/>'
    if text.count(marker) != 1:
        raise SystemExit("Unexpected XML revision; no files changed")
    text = text.replace(marker, marker + '\n  <option cone="elliptic" impratio="10"/>', 1)
elif len(options) != 1 or options[0].get("cone") != "elliptic" or float(options[0].get("impratio", "1")) != 10:
    raise SystemExit("Existing solver options differ; no files changed")

for name in ("left_finger", "right_finger"):
    matches = list(re.finditer(rf'<joint name="{name}"[^>]*/>', text))
    if len(matches) != 1:
        raise SystemExit("Unexpected finger joint; no files changed: " + name)
    match = matches[0]
    tag = re.sub(r'\s+sol(?:ref|imp)limit="[^"]*"', "", match.group(0))
    tag = tag[:-2].rstrip() + ' solreflimit="0.004 1" solimplimit="0.99 0.999 0.001"/>'
    text = text[:match.start()] + tag + text[match.end():]

ET.fromstring(text)
if text != original:
    backup = p.with_name(p.name + ".before_grip_repair")
    if not backup.exists():
        backup.write_text(original)
    p.write_text(text)
    print("Installed contact/limit repair; backup:", backup)
else:
    print("Contact/limit repair already installed")
