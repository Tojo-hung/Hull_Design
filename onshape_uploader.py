"""
onshape_uploader.py  —  Upload DXF sections to Onshape via REST API
====================================================================

Reads the folder produced by export_dxf_sections() (one DXF per station,
filenames like station_02_x1120mm.dxf) and:

  1. Creates a new Onshape document.
  2. Imports each DXF file into that document as its own tab.
  3. Prints a FeatureScript snippet you can paste into a Feature Studio
     to auto-create offset sketch planes at the correct X positions.

Setup
-----
  1.  Go to https://dev-portal.onshape.com -> API Keys -> Create new API key.
  2.  Set the two environment variables (or edit the constants below):

        set ONSHAPE_ACCESS_KEY=your-access-key
        set ONSHAPE_SECRET_KEY=your-secret-key

  3.  Install requests if needed:
        pip install requests

Usage
-----
  python onshape_uploader.py  <dxf_folder>  [--doc-name "My Hull"]

  <dxf_folder>   Folder written by "Export DXF Sections" in the app.
  --doc-name     Optional name for the new Onshape document.
"""

import argparse
import base64
import hashlib
import hmac
import json
import mimetypes
import os
import re
import string
import time
import uuid
from pathlib import Path


# ── Onshape credentials ───────────────────────────────────────────────────────
#  Override here or via environment variables.
ACCESS_KEY = os.environ.get('ONSHAPE_ACCESS_KEY', 'on_YjSUpeLth8hLewW3VreQw')
SECRET_KEY = os.environ.get('ONSHAPE_SECRET_KEY', 'W8dnvxMga6iANHBZAVfSWkvpmkNAWG1edVoAt3ozZSn2sGha')
BASE_URL   = 'https://cad.onshape.com'


# ── Low-level signed request ──────────────────────────────────────────────────

def _make_nonce():
    return ''.join([string.ascii_letters[int(c) % 52]
                    for c in str(uuid.uuid4().int)[:25]])


def _sign_request(method, path, query, content_type, body_bytes,
                  access_key, secret_key):
    """Build the HMAC-SHA256 Authorization header required by Onshape."""
    date   = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime())
    nonce  = _make_nonce()
    method = method.lower()

    query_str = query if query else ''
    # Onshape requires a trailing newline in the signed string
    to_sign = (method + '\n' + nonce + '\n' + date + '\n' +
               content_type + '\n' + path + '\n' + query_str + '\n').lower()

    sig = hmac.new(
        secret_key.encode('utf-8'),
        to_sign.encode('utf-8'),
        hashlib.sha256,
    ).digest()
    encoded = base64.b64encode(sig).decode('utf-8')

    auth = (f'On {access_key}:HmacSHA256:{encoded}')
    headers = {
        'Date':          date,
        'On-Nonce':      nonce,
        'Authorization': auth,
        'Content-Type':  content_type,
        'Accept':        'application/json',
    }
    return headers


def _request(method, path, query='', body_bytes=b'',
             content_type='application/json'):
    import requests

    if not ACCESS_KEY or not SECRET_KEY:
        raise RuntimeError(
            'Onshape API keys not set.\n'
            'Set environment variables ONSHAPE_ACCESS_KEY and ONSHAPE_SECRET_KEY,\n'
            'or edit ACCESS_KEY / SECRET_KEY at the top of onshape_uploader.py.'
        )

    url     = BASE_URL + path + (('?' + query) if query else '')
    headers = _sign_request(method, path, query, content_type,
                            body_bytes, ACCESS_KEY, SECRET_KEY)

    resp = requests.request(
        method, url, headers=headers,
        data=body_bytes if method.upper() != 'GET' else None,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json() if resp.content else {}


# ── Onshape API helpers ───────────────────────────────────────────────────────

def create_document(name):
    """Create a new Onshape document; return (did, wid)."""
    body = json.dumps({'name': name, 'ownerType': 0}).encode()
    data = _request('POST', '/api/documents', body_bytes=body)
    did  = data['id']
    wid  = data['defaultWorkspace']['id']
    print(f'Created document: {name}')
    print(f'  URL: https://cad.onshape.com/documents/{did}')
    print(f'  Document ID : {did}')
    print(f'  Workspace ID: {wid}')
    return did, wid


def import_dxf(did, wid, dxf_path):
    """
    Import a DXF file into the document as a new tab.
    Returns the new element ID.
    """
    import requests

    filename = Path(dxf_path).name

    # Blob upload endpoint
    path = f'/api/blobelements/d/{did}/w/{wid}'
    url  = BASE_URL + path

    # Build the multipart body first so we can extract the exact Content-Type
    # header (including boundary) and sign with it.
    with open(dxf_path, 'rb') as fh:
        files = {'file': (filename, fh, 'application/octet-stream')}
        prep = requests.Request('POST', url, files=files).prepare()

    content_type = prep.headers.get('Content-Type', '')

    # Sign with the actual content-type (including boundary)
    auth_headers = _sign_request('POST', path, '', content_type, b'', ACCESS_KEY, SECRET_KEY)
    prep.headers.update(auth_headers)
    prep.headers['Accept'] = 'application/json'

    session = requests.Session()
    resp = session.send(prep, timeout=120)

    if not resp.ok:
        print(f'  Upload error {resp.status_code}: {resp.text}')
    resp.raise_for_status()
    data = resp.json()
    eid  = data.get('id')
    print(f'  Uploaded: {filename} -> element {eid}')
    return eid


# ── FeatureScript generator ───────────────────────────────────────────────────

def generate_featurescript(stations, did, doc_name):
    """
    Return a FeatureScript snippet that creates one offset plane per station
    plus a stub loft.  Paste this into an Onshape Feature Studio.

    stations: list of (index, x_mm, dxf_filename) tuples
    """
    lines = [
        '// -----------------------------------------------------------',
        f'// Auto-generated by onshape_uploader.py for: {doc_name}',
        '// Paste into a Feature Studio, then add a Loft across the',
        '// named sketch planes.',
        '// -----------------------------------------------------------',
        '',
        'FeatureScript 2176;',
        'import(path : "onshape/std/geometry.fs", version : "2176.0");',
        '',
        'annotation { "Feature Type Name" : "Hull Planes" }',
        'export const hullPlanes = defineFeature(function(context is Context,',
        '                                                  id is Id,',
        '                                                  definition is map)',
        '{',
        '    const frontPlane = qBuiltinPart(BuiltInType.FRONT_PLANE);',
    ]

    for i, (idx, x_mm, fname) in enumerate(stations):
        plane_id = f'plane_{idx:02d}'
        lines.append(f'')
        lines.append(f'    // Station {idx}: X = {x_mm} mm  ({fname})')
        lines.append(f'    opPlane(context, id + "{plane_id}", {{')
        lines.append(f'        "plane" : plane(vector(0, 0, {x_mm}) * millimeter,')
        lines.append(f'                         Z_VEC),')
        lines.append(f'    }});')

    lines += [
        '',
        '    // Add a Loft manually across the sketch planes imported from DXF.',
        '}, {});',
        '',
    ]
    return '\n'.join(lines)


# ── Station parser ────────────────────────────────────────────────────────────

def parse_station_folder(folder):
    """
    Return list of (index, x_mm, filepath) sorted by x_mm.
    Expects filenames like  station_02_x1120mm.dxf
    """
    pattern = re.compile(r'station_(\d+)_x(\d+)mm\.dxf$', re.IGNORECASE)
    stations = []
    for p in sorted(Path(folder).glob('*.dxf')):
        m = pattern.match(p.name)
        if m:
            stations.append((int(m.group(1)), int(m.group(2)), p))
    stations.sort(key=lambda t: t[1])
    return stations


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dxf_folder', help='Folder containing station DXF files')
    parser.add_argument('--doc-name', default='Moth Hull',
                        help='Name for the new Onshape document (default: "Moth Hull")')
    parser.add_argument('--featurescript-only', action='store_true',
                        help='Only print the FeatureScript; do not call the Onshape API')
    args = parser.parse_args()

    stations = parse_station_folder(args.dxf_folder)
    if not stations:
        print(f'No station DXF files found in: {args.dxf_folder}')
        print('Expected filenames like:  station_02_x1120mm.dxf')
        return

    print(f'Found {len(stations)} station DXFs:')
    for idx, x_mm, fp in stations:
        print(f'  [{idx:02d}]  x={x_mm:5d} mm  {fp.name}')
    print()

    # Always print the FeatureScript — useful regardless of API usage
    fs = generate_featurescript(
        [(idx, x_mm, fp.name) for idx, x_mm, fp in stations],
        did='<document-id>',
        doc_name=args.doc_name,
    )
    fs_path = Path(args.dxf_folder) / 'hull_planes.fs'
    fs_path.write_text(fs, encoding='utf-8')
    print(f'FeatureScript written to: {fs_path}')
    print('  Paste this into an Onshape Feature Studio to create offset planes.\n')

    if args.featurescript_only:
        return

    # Upload to Onshape
    try:
        import requests  # noqa — check early
    except ImportError:
        print('requests not installed.  Run:  pip install requests')
        return

    if not ACCESS_KEY or not SECRET_KEY:
        print('Onshape API keys not set.')
        print('Set environment variables:')
        print('  ONSHAPE_ACCESS_KEY=your-key')
        print('  ONSHAPE_SECRET_KEY=your-secret')
        print()
        print('Or run with --featurescript-only to skip the upload.')
        return

    did, wid = create_document(args.doc_name)
    print()
    print('Uploading DXF files...')
    for idx, x_mm, fp in stations:
        import_dxf(did, wid, fp)

    # Re-print the FeatureScript with the real document ID
    fs2 = generate_featurescript(
        [(idx, x_mm, fp.name) for idx, x_mm, fp in stations],
        did=did,
        doc_name=args.doc_name,
    )
    fs_path.write_text(fs2, encoding='utf-8')

    print()
    print('Done.')
    print(f'  Onshape document: https://cad.onshape.com/documents/{did}')
    print(f'  FeatureScript   : {fs_path}')
    print()
    print('Next steps in Onshape:')
    print('  1. Open the document above.')
    print('  2. Add a new Feature Studio tab.')
    print('  3. Paste the contents of hull_planes.fs into it.')
    print('  4. In your Part Studio, Insert > Feature Studio Features > Hull Planes.')
    print('  5. For each station tab, open it, select all, copy,')
    print('     then paste into a sketch on the corresponding plane.')
    print('  6. Loft across all sketch planes.')


if __name__ == '__main__':
    main()
