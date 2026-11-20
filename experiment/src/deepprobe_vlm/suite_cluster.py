"""Bounded, complete workload census without copying unrelated workload specs.

A short-lived loopback kubectl proxy negotiates server-side metadata-only
responses and follows every continuation token. Objects in the project's
name/identity/owner scope are then read exactly and UID checked.
No partial census or previous successful result is usable for admission.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
import selectors
import signal
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import re
import subprocess
import time


class ClusterQueryError(ValueError):
    pass


RESOURCES = (('pytorchjobs', 'PyTorchJob', 'kubeflow.org/v1', '/apis/kubeflow.org/v1/pytorchjobs'),
             ('pods', 'Pod', 'v1', '/api/v1/pods'))
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')
ANNOTATIONS = ('deepprobe/binding-sha256', 'deepprobe/profile-sha256')
METADATA_ACCEPT = 'application/json;as=PartialObjectMetadataList;g=meta.k8s.io;v=v1'
MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class _TransientQueryError(Exception):
    pass


@contextmanager
def _metadata_connection(tools, deadline, clock):
    """Expose only a short-lived loopback proxy, using kubectl's existing auth."""
    process = None
    with tempfile.TemporaryFile() as errors:
        try:
            process = subprocess.Popen([tools['kubectl'], '--kubeconfig', tools['kubeconfig'],
                '--request-timeout=25s', 'proxy', '--port=0', '--address=127.0.0.1',
                '--accept-hosts=^127\\.0\\.0\\.1$'], stdout=subprocess.PIPE, stderr=errors,
                start_new_session=True)
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                ready = selector.select(max(0, min(10, deadline - clock())))
                require(ready, 'Metadata proxy did not become ready within the query deadline')
                line = os.read(process.stdout.fileno(), 4096).decode('utf-8', 'replace').strip()
            match = re.fullmatch(r'Starting to serve on 127\.0\.0\.1:([0-9]+)', line)
            require(match and process.poll() is None, 'Metadata proxy did not report a live loopback endpoint')
            endpoint = 'http://127.0.0.1:' + match[1]
            # Never send loopback requests through an inherited HTTP proxy.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

            def get(path, timeout):
                request = urllib.request.Request(endpoint + path, headers={'Accept': METADATA_ACCEPT})
                try:
                    with opener.open(request, timeout=timeout) as response:
                        require(response.status == 200, 'Metadata API response is not successful')
                        payload = response.read(MAX_RESPONSE_BYTES + 1)
                except urllib.error.HTTPError as exc:
                    if exc.code in {429, 500, 502, 503, 504}:
                        raise _TransientQueryError('metadata HTTP ' + str(exc.code)) from exc
                    raise ClusterQueryError('Metadata API unavailable: HTTP ' + str(exc.code)) from exc
                except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                    raise _TransientQueryError('metadata transport unavailable') from exc
                require(len(payload) <= MAX_RESPONSE_BYTES, 'Metadata page exceeds bounded response size')
                return payload.decode('utf-8')
            yield get
        finally:
            if process is not None:
                # This process group was created only for this query. No job or
                # GPU process is signaled, and no proxy can outlive admission.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
                if process.stdout is not None:
                    process.stdout.close()
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                    raise ClusterQueryError('Metadata proxy descendants did not exit; admission remains closed')

def require(ok, message):
    if not ok:
        raise ClusterQueryError(message)


def project_workloads(items, project_name_prefixes):
    """Include unlabeled historical names and the full UID ownership closure."""
    require(project_name_prefixes and all(isinstance(p, str) and p for p in project_name_prefixes),
            'Complete project name scope is required')
    by_uid = {}
    for item in items:
        meta = item.get('metadata', {})
        uid = meta.get('uid')
        if uid:
            require(uid not in by_uid, 'Duplicate workload UID in census')
            by_uid[uid] = item
    selected = {id(item) for item in items if
        any(item.get('metadata', {}).get('name', '').startswith(p) for p in project_name_prefixes)
        or any(str(key).startswith('deepprobe/') for key in item.get('metadata', {}).get('labels', {}))
        or any(key in item.get('metadata', {}).get('annotations', {}) for key in ANNOTATIONS)
        or any(any(ref.get('name', '').startswith(p) for p in project_name_prefixes)
               for ref in item.get('metadata', {}).get('ownerReferences', []))}
    while True:
        owners = {ref.get('uid') for item in items if id(item) in selected
                  for ref in item.get('metadata', {}).get('ownerReferences', [])}
        uids = {item.get('metadata', {}).get('uid') for item in items if id(item) in selected}
        expanded = selected | {id(item) for item in items if item.get('metadata', {}).get('uid') in owners
            or any(ref.get('uid') in (uids | owners) for ref in item.get('metadata', {}).get('ownerReferences', []))}
        if expanded == selected:
            return [item for item in items if id(item) in selected]
        selected = expanded


def _metadata_page(payload, kind, api_version):
    try:
        page = json.loads(payload)
    except (ValueError, TypeError) as exc:
        raise ClusterQueryError('Invalid metadata page JSON') from exc
    require(isinstance(page, dict) and page.get('kind') == 'PartialObjectMetadataList'
            and page.get('apiVersion') == 'meta.k8s.io/v1',
            'Server did not honor PartialObjectMetadataList; full-spec fallback is forbidden')
    header = page.get('metadata')
    require(isinstance(header, dict) and isinstance(header.get('resourceVersion'), str)
            and header['resourceVersion'], 'Metadata list resourceVersion is missing')
    continuation = header.get('continue', '')
    require(isinstance(continuation, str), 'Invalid metadata continuation')
    remaining = header.get('remainingItemCount')
    require(remaining is None or type(remaining) is int and remaining >= 0,
            'Invalid metadata remaining-item count')
    require(continuation or remaining in {None, 0}, 'Incomplete metadata list: items remain')
    require(isinstance(page.get('items'), list), 'Metadata page items missing')
    rows = []
    for item in page['items']:
        require(isinstance(item, dict) and set(item) <= {'apiVersion', 'kind', 'metadata'}
                and item.get('kind') in {None, 'PartialObjectMetadata'}
                and item.get('apiVersion') in {None, 'meta.k8s.io/v1'}, 'Unexpected metadata object kind')
        meta = item.get('metadata')
        require(isinstance(meta, dict) and all(isinstance(meta.get(k), str) and NAME.fullmatch(meta[k])
                for k in ('namespace', 'name', 'uid')) and isinstance(meta.get('resourceVersion'), str)
                and meta['resourceVersion'], 'Metadata object lacks exact identity')
        labels, owners = meta.get('labels', {}), meta.get('ownerReferences', [])
        annotations = meta.get('annotations', {})
        require(isinstance(labels, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in labels.items()),
                'Invalid workload labels')
        require(isinstance(annotations, dict), 'Invalid workload annotations')
        require(isinstance(owners, list) and all(isinstance(r, dict) and isinstance(r.get('uid'), str)
                and r['uid'] for r in owners), 'Invalid workload owner references')
        metadata = {key: meta[key] for key in ('namespace', 'name', 'uid', 'resourceVersion')}
        metadata.update(labels=labels, ownerReferences=owners)
        if meta.get('deletionTimestamp'):
            metadata['deletionTimestamp'] = meta['deletionTimestamp']
        annotations = {key: annotations[key] for key in ANNOTATIONS if key in annotations}
        if annotations:
            metadata['annotations'] = annotations
        rows.append(dict(apiVersion=api_version, kind=kind, metadata=metadata))
    return rows, header['resourceVersion'], continuation

def list_workloads(tools, *, project_name_prefixes, command=None, timeout_seconds=120,
                   attempts=3, sleep=None, clock=None, metadata_connection=None):
    """Complete server-side metadata pagination, then exact project UID reads."""
    command, sleep, clock = command or subprocess.run, sleep or time.sleep, clock or time.monotonic
    metadata_connection = metadata_connection or _metadata_connection
    require(type(attempts) is int and 1 <= attempts <= 3 and 0 < timeout_seconds <= 180,
            'Workload query must have a short bounded retry/deadline policy')
    started, calls, retries, received_bytes = clock(), 0, 0, 0
    deadline = started + timeout_seconds

    def retry(operation, label):
        nonlocal calls, retries, received_bytes
        for attempt in range(attempts):
            remaining = deadline - clock()
            require(remaining > 0, 'Complete workload query deadline exceeded; admission remains closed')
            calls += 1
            try:
                payload = operation(min(30, remaining))
            except (subprocess.TimeoutExpired, _TransientQueryError):
                require(attempt + 1 < attempts,
                        'Complete workload GET unavailable (' + label + ', attempt ' + str(attempt + 1)
                        + '/' + str(attempts) + '); admission remains closed')
                retries += 1
                sleep(min(attempt + 1, max(0, deadline - clock())))
                continue
            require(isinstance(payload, str) and len(payload.encode('utf-8')) <= MAX_RESPONSE_BYTES,
                    'Workload response exceeds bounded size')
            received_bytes += len(payload.encode('utf-8'))
            return payload
        raise ClusterQueryError('Complete workload GET unavailable')

    def request(args):
        def get(timeout):
            argv = [tools['kubectl'], '--kubeconfig', tools['kubeconfig'],
                    '--request-timeout=' + str(max(1, min(25, int(timeout)))) + 's', *args]
            result = command(argv, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return result.stdout.decode('utf-8') if isinstance(result.stdout, bytes) else result.stdout
            error = result.stderr.decode('utf-8', 'replace') if isinstance(result.stderr, bytes) else str(result.stderr)
            if any(token in error.lower() for token in ('timeout', 'timed out', 'deadline exceeded',
                'temporarily unavailable', 'connection reset', 'connection refused', 'unexpected eof',
                'too many requests', 'serviceunavailable', 'service unavailable', 'internalerror',
                'internal server error', 'unable to connect to the server')):
                raise _TransientQueryError('exact GET transport unavailable')
            raise ClusterQueryError('Exact workload GET failed; admission remains closed')
        return retry(get, args[1])

    census, versions, pages = [], {}, {}
    identities, uids = set(), set()
    with metadata_connection(tools, deadline, clock) as metadata_get:
        for resource, kind, api_version, endpoint in RESOURCES:
            continuation, seen_tokens, version = '', set(), None
            for page_number in range(1, 1025):
                query = {'limit': '100'}
                if continuation:
                    query['continue'] = continuation
                path = endpoint + '?' + urllib.parse.urlencode(query)
                payload = retry(lambda timeout: metadata_get(path, timeout), resource + ' metadata')
                rows, current_version, continuation = _metadata_page(payload, kind, api_version)
                require(version is None or version == current_version,
                        'Metadata snapshot resourceVersion changed during pagination')
                version = current_version
                for row in rows:
                    meta = row['metadata']; identity = (kind, meta['namespace'], meta['name'])
                    require(identity not in identities and meta['uid'] not in uids,
                            'Duplicate workload in paginated metadata census')
                    identities.add(identity); uids.add(meta['uid']); census.append(row)
                if not continuation:
                    pages[resource], versions[resource] = page_number, version
                    break
                require(continuation not in seen_tokens, 'Metadata pagination token repeated')
                seen_tokens.add(continuation)
            else:
                raise ClusterQueryError('Metadata pagination limit reached before a complete list')
    census_bytes = received_bytes
    selected = project_workloads(census, project_name_prefixes)
    exact = {}
    resources = {kind: resource for resource, kind, _, _ in RESOURCES}
    for row in selected:
        meta = row['metadata']
        payload = request(['get', resources[row['kind']], meta['name'], '-n', meta['namespace'],
                           '--show-managed-fields=false', '-o', 'json'])
        try:
            full = json.loads(payload)
        except (ValueError, TypeError) as exc:
            raise ClusterQueryError('Invalid exact workload response') from exc
        require(isinstance(full, dict) and full.get('kind') == row['kind']
                and full.get('apiVersion') == row['apiVersion'], 'Exact workload kind changed')
        observed = full.get('metadata', {})
        require(all(observed.get(k) == meta[k] for k in ('name', 'namespace', 'uid')),
                'Workload UID changed between census and exact read')
        require(isinstance(observed.get('resourceVersion'), str) and observed['resourceVersion'],
                'Exact workload resourceVersion missing')
        def scope_identity(metadata):
            return ({k: v for k, v in metadata.get('labels', {}).items() if k.startswith('deepprobe/')},
                    {k: v for k, v in metadata.get('annotations', {}).items() if k in ANNOTATIONS},
                    sorted((r.get('uid'), r.get('name', '')) for r in metadata.get('ownerReferences', [])))
        require(scope_identity(observed) == scope_identity(meta),
                'Project workload ownership or identity changed during exact read')
        exact[meta['uid']] = full
    items = [exact.get(row['metadata']['uid'], row) for row in census]
    # A changed project owner/label relation must never leave an uninspected spec.
    require(all(row['metadata']['uid'] in exact for row in project_workloads(items, project_name_prefixes)),
            'Project workload scope changed during exact reads; admission remains closed')
    finished = clock()
    require(finished <= deadline, 'Complete workload query deadline exceeded; admission remains closed')
    return {'apiVersion': 'v1', 'kind': 'List', 'metadata': {'continue': ''}, 'items': items,
            'census': {'complete': True, 'all_namespaces': True, 'resource_versions': versions, 'pages': pages,
                       'metadata_mode': 'server_partial_object_metadata',
                       'objects': len(census), 'exact_project_objects': len(exact),
                       'get_calls': calls, 'transport_retries': retries,
                       'census_bytes': census_bytes, 'exact_object_bytes': received_bytes - census_bytes,
                       'elapsed_seconds': finished - started}}
