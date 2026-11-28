# Verifying a dump before blaming the port

A damaged dump makes the port look broken in ways that are hard to chase: the
title traps on its own assertion, or a loading screen sits there with sound and
never finishes, and nothing the loader prints says "a file is wrong". This page
is the ten-minute check to run **before** spending a day on the loader.

## Why a size check is not enough

An extraction that runs out of disk space leaves the file at its full length
with a hole of zeros in it. The directory entry, the manifest, `census` and any
other length-based check will all agree that the file is exactly right, and the
port will read zeros where the title expects records.

The only thing that catches this is a hash of the contents, ideally the piece
hashes the container already carries.

## Symptoms that point here

* the title's own `int3` during world bootstrap (`+0x1330A44` on the build this
  was written against, with `[owner+0x5C]` disagreeing with the record it read);
* `+0x1CE6E13` immediately after the first `.rpf` opens — this one is the
  title's "a file I need is not there" assert;
* a loading screen that stays up for minutes with music, no error of any kind,
  and no new file opened by the port;
* textures or meshes that are simply absent while everything else renders.

## Checking against the torrent's piece hashes

If the dump came from a torrent, the `.torrent` file already contains a SHA-1
per piece. Map pieces to files, hash what is on disk, compare. Python only, no
libraries:

```python
import hashlib, os, sys

TOR = sys.argv[1]          # the .torrent file
ROOT = sys.argv[2]         # the folder the files were extracted into

def bdecode(b, i=0):
    c = b[i:i+1]
    if c == b'i':
        j = b.index(b'e', i); return int(b[i+1:j]), j+1
    if c == b'l':
        i += 1; out = []
        while b[i:i+1] != b'e':
            v, i = bdecode(b, i); out.append(v)
        return out, i+1
    if c == b'd':
        i += 1; out = {}
        while b[i:i+1] != b'e':
            k, i = bdecode(b, i); v, i = bdecode(b, i); out[k] = v
        return out, i+1
    j = b.index(b':', i); n = int(b[i:j]); return b[j+1:j+1+n], j+1+n

info = bdecode(open(TOR, 'rb').read())[0][b'info']
plen = info[b'piece length']
pieces = info[b'pieces']
hashes = [pieces[i:i+20] for i in range(0, len(pieces), 20)]

files, off = [], 0
for f in info.get(b'files', []):
    path = '/'.join(p.decode() for p in f[b'path'])
    files.append((path, off, f[b'length'])); off += f[b'length']
if not files:
    files = [(info[b'name'].decode(), 0, info[b'length'])]; off = info[b'length']

total = off
def covering(lo, hi):
    return [(p, s, l) for p, s, l in files if not (s + l <= lo or s >= hi)]

bad = []
for idx in range(len(hashes)):
    lo, hi = idx * plen, min(idx * plen + plen, total)
    h = hashlib.sha1(); ok = True
    for path, s, l in covering(lo, hi):
        fp = os.path.join(ROOT, path.replace('/', os.sep))
        a, b = max(lo, s) - s, min(hi, s + l) - s
        try:
            with open(fp, 'rb') as fh:
                fh.seek(a); left = b - a
                while left > 0:
                    c = fh.read(min(1 << 20, left))
                    if not c: ok = False; break
                    h.update(c); left -= len(c)
        except OSError:
            ok = False
        if not ok: break
    if not ok or h.digest() != hashes[idx]:
        bad.append(idx)
    if idx and idx % 2000 == 0:
        print('  checked', idx, 'of', len(hashes), 'bad', len(bad), flush=True)

print('mismatched pieces:', len(bad))
for idx in bad[:20]:
    lo = idx * plen
    print('  piece', idx, 'covers', [p for p, _, _ in covering(lo, min(lo + plen, total))])
```

Run it against the extracted folder. A mismatch names the file and the byte
range; the first one is usually where the title misbehaves. To save time on a
200 GB dump, stop after the packs you care about by filtering the pieces against
the file names you see in the output.

## What it looks like when it fires

On the machine this page was written on, one piece failed. The bytes it covered
were the **last 3.24 GB of `navmeshes.rpf`**, all zeros — 194 of that file's 401
pieces. Every size check had passed, because the file was the right length.

Restoring that one file was the whole fix: the title's assertion stopped firing,
the loading screen finished, and the unmodified loader ran menus, world and
heavy scenes for minutes with no crash log.

## What to do with a bad file

Re-extract that file from a complete source, then run the check again until it
reports zero mismatches. Replacing the single damaged file is enough; there is
no need to re-download a whole dump.

If the source is a multi-volume archive and only some volumes are at hand, a
copy of the volumes with the missing ones renamed away is enough for 7-Zip to
list and extract the files that lie entirely inside those volumes: it reports a
missing next volume, but it writes the files it could reach. Check those files
against the piece hashes before trusting them.
