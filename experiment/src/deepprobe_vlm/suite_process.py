"""Track suite descendants across torchrun's per-rank POSIX sessions."""
import os
import ctypes
import platform
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid

MARKER = 'DEEPPROBE_SUITE_PROCESS_TOKEN'
MARKER_CHAIN = MARKER + '_CHAIN'
MAX_MARKER_DEPTH = 16
TOKEN_PATTERN = re.compile(r'[0-9a-f]{32}')
LINUX_PROC = Path('/proc/self/stat').is_file()


class GroupCleanupError(RuntimeError):
    group_cleanup_verified = False


def _marker_tokens(environment):
    """Read bounded ancestry without dropping identities or exposing tokens."""
    current, chain = environment.get(MARKER), environment.get(MARKER_CHAIN)
    if current is None and chain is None:
        return ()
    if not isinstance(current, str) or TOKEN_PATTERN.fullmatch(current) is None:
        raise GroupCleanupError('invalid suite process identity')
    if chain is None:
        # Preserve a parent launched by the original single-marker supervisor.
        return (current,)
    if not isinstance(chain, str) or len(chain) > MAX_MARKER_DEPTH * 33 - 1:
        raise GroupCleanupError('suite process ancestry exceeds its bound')
    tokens = tuple(chain.split(','))
    if (not 1 <= len(tokens) <= MAX_MARKER_DEPTH
            or any(TOKEN_PATTERN.fullmatch(token) is None for token in tokens)
            or len(set(tokens)) != len(tokens) or tokens[-1] != current):
        raise GroupCleanupError('invalid suite process ancestry')
    return tokens


def _child_environment(environment, token):
    ancestors = _marker_tokens(environment)
    if not isinstance(token, str) or TOKEN_PATTERN.fullmatch(token) is None or token in ancestors:
        raise GroupCleanupError('invalid new suite process identity')
    if len(ancestors) >= MAX_MARKER_DEPTH:
        raise GroupCleanupError('suite process ancestry exceeds its bound; child not started')
    return dict(environment, **{MARKER: token, MARKER_CHAIN: ','.join((*ancestors, token))})


def _process_marker_tokens(raw):
    environment = {}
    names = {name.encode(): name for name in (MARKER, MARKER_CHAIN)}
    for entry in raw.split(b'\0'):
        key, separator, value = entry.partition(b'=')
        if key not in names:
            continue
        name = names[key]
        if not separator or name in environment:
            raise GroupCleanupError('ambiguous suite process identity in process inventory')
        try:
            environment[name] = value.decode('ascii')
        except UnicodeDecodeError:
            raise GroupCleanupError('invalid suite process identity in process inventory') from None
    return _marker_tokens(environment)


class PidfdAPI:
    """Native bindings or the explicitly validated 64-bit Linux syscall ABI."""
    def __init__(self):
        if platform.system() != 'Linux' or platform.machine() not in {'x86_64','aarch64'} or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise GroupCleanupError('pidfd requires verified 64-bit Linux x86_64/aarch64 ABI')
        if hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'):
            self.backend = 'python_native'
            self.open = lambda pid: os.pidfd_open(pid,0)
            self.send = lambda fd,sig: signal.pidfd_send_signal(fd,sig,None,0)
        else:
            self.backend = 'linux_syscall_434_424'
            library = ctypes.CDLL(None,use_errno=True)
            call = library.syscall
            call.restype = ctypes.c_long
            def invoke(number,*arguments):
                result = call(ctypes.c_long(number),*arguments)
                if result == -1:
                    error = ctypes.get_errno()
                    raise OSError(error,os.strerror(error))
                return int(result)
            self.open = lambda pid: invoke(434,ctypes.c_int(pid),ctypes.c_uint(0))
            self.send = lambda fd,sig: invoke(424,ctypes.c_int(fd),ctypes.c_int(sig),ctypes.c_void_p(),ctypes.c_uint(0))
        # Verify before a child starts, and close even when the signal-0 probe
        # fails. Missing kernel support is never an os.kill fallback.
        descriptor = None
        try:
            descriptor = self.open(os.getpid())
            self.send(descriptor,0)
        except Exception as exc:
            raise GroupCleanupError('pidfd self-test failed before child start: '+str(exc)) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)


def process_table():
    rows = {}
    if LINUX_PROC:
        deadline = time.monotonic() + 2
        for path in Path('/proc').glob('[0-9]*/stat'):
            if time.monotonic() > deadline:
                raise GroupCleanupError('process inventory exceeded bounded read deadline')
            try:
                fields = path.read_text().rsplit(')', 1)[1].split()
                uid = path.stat().st_uid
            except (FileNotFoundError, ProcessLookupError):
                continue
            pid = int(path.parent.name)
            rows[pid] = {'pid':pid, 'ppid':int(fields[1]), 'pgid':int(fields[2]),
                         'sid':fields[3], 'state':fields[0], 'start':fields[19], 'uid':uid}
    else:
        # CPU regression support on the developer Mac. Production uses Linux
        # /proc start ticks, inherited markers and pidfds below.
        result = subprocess.run(['/bin/ps','-axo','pid=,ppid=,pgid=,sess=,stat=,lstart='],
                                capture_output=True,text=True,check=True,timeout=2)
        for line in result.stdout.splitlines():
            fields = line.split()
            pid = int(fields[0])
            rows[pid] = {'pid':pid,'ppid':int(fields[1]),'pgid':int(fields[2]),'sid':fields[3],
                         'state':fields[4][0],'start':' '.join(fields[5:]),'uid':os.getuid()}
    return rows


class OwnedTree:
    def __init__(self, process, token, pidfds=None):
        self.process, self.token, self.pidfds = process, token, pidfds
        self.identities = {}
        self.marker = (MARKER + '=' + token).encode()
        rows = process_table()
        if process.pid in rows:
            self.identities[process.pid] = rows[process.pid]['start']

    def refresh(self):
        self.process.poll()
        rows = process_table()
        owned = {pid for pid,start in self.identities.items()
                 if pid in rows and rows[pid]['start'] == start}
        if not LINUX_PROC:
            # Mac-only regression fallback keeps the newly created process group.
            owned |= {pid for pid, row in rows.items() if row['pgid'] == self.process.pid}
        if LINUX_PROC:
            deadline = time.monotonic() + 2
            for pid,row in rows.items():
                if time.monotonic() > deadline:
                    raise GroupCleanupError('owned process marker inventory exceeded deadline')
                if row['uid'] != os.getuid() or row['state'] in {'Z','X'}:
                    continue
                try:
                    with open('/proc/' + str(pid) + '/environ','rb') as stream:
                        raw = stream.read(1_048_577)
                    if len(raw) > 1_048_576:
                        raise GroupCleanupError('same-user process environment exceeded read bound')
                except (FileNotFoundError,ProcessLookupError):
                    continue
                if self.token in _process_marker_tokens(raw):
                    owned.add(pid)
        while True:
            children = {pid for pid,row in rows.items() if row['ppid'] in owned}
            if children <= owned:
                break
            owned |= children
        for pid in owned:
            self.identities[pid] = rows[pid]['start']
        if os.getpid() in owned:
            raise GroupCleanupError('refusing to own or signal the suite worker')
        # Zombies have already exited: they cannot write files or hold CUDA.
        return {pid:rows[pid] for pid in owned if rows[pid]['state'] not in {'Z','X'}}

    def send(self, members, sig):
        for pid,expected in members.items():
            if pid <= 1 or pid == os.getpid():
                raise GroupCleanupError('unsafe owned process identity')
            descriptor = None
            try:
                if LINUX_PROC:
                    # A pidfd pins the target process across the identity check
                    # and signal; PID reuse cannot target an unrelated process.
                    descriptor = self.pidfds.open(pid)
                if LINUX_PROC:
                    try:
                        observed_start = Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
                    except FileNotFoundError:
                        continue
                else:
                    observed_start = process_table().get(pid,{}).get('start')
                if observed_start != expected['start']:
                    continue
                if descriptor is not None:
                    self.pidfds.send(descriptor,sig)
                else:
                    os.kill(pid,sig)
            except ProcessLookupError:
                continue
            finally:
                if descriptor is not None:
                    os.close(descriptor)

    def stop(self):
        for sig,seconds in ((signal.SIGTERM,2),(signal.SIGKILL,4)):
            deadline = time.monotonic()+seconds
            while True:
                members = self.refresh()
                if not members and self.quiet_after_leader_exit():
                    return
                self.send(members,sig)
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
        if not self.quiet_after_leader_exit():
            raise GroupCleanupError('owned descendant/session tree remains alive; archive forbidden')

    def quiet_after_leader_exit(self):
        if self.process.poll() is None:
            return False
        # Reap first, then take independent fresh inventories. A child forked
        # during the prior scan may only become visible after its leader exits.
        self.process.wait(timeout=1)
        if self.refresh():
            return False
        time.sleep(0.05)
        return not self.refresh()


def run_group(argv, *, stdout, stderr, env, cwd, timeout, on_started=None):
    if timeout <= 0:
        raise ValueError('no phase budget remains; command not started')
    token = uuid.uuid4().hex
    child_env = _child_environment(env, token)
    pidfds = PidfdAPI() if LINUX_PROC else None
    process = tree = None
    previous, pending = {}, []
    spawning = True
    deadline = time.monotonic()+timeout

    def interrupted(signum,frame):
        if spawning:
            pending.append(signum)
        else:
            raise RuntimeError('suite worker interrupted by signal '+str(signum))

    try:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):
            previous[sig] = signal.signal(sig,interrupted)
        process = subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=child_env,
                                   cwd=cwd,start_new_session=True,stdin=subprocess.DEVNULL,close_fds=True)
        tree = OwnedTree(process,token,pidfds)
        if on_started is not None:
            on_started(process)
        spawning = False
        if pending:
            raise RuntimeError('suite worker interrupted during child spawn: '+str(pending))
        while True:
            members = tree.refresh()
            code = process.poll()
            if code is not None:
                if not tree.quiet_after_leader_exit():
                    raise RuntimeError('command leader exited while owned descendants remained')
                result = subprocess.CompletedProcess(argv,code)
                result.group_cleanup_verified = True
                return result
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(argv,timeout)
            time.sleep(min(0.05,max(0,deadline-time.monotonic())))
    except BaseException as exc:
        for sig in previous:
            signal.signal(sig,signal.SIG_IGN)
        if process is not None:
            try:
                if tree is None:
                    tree = OwnedTree(process,token,pidfds)
                tree.stop()
            except BaseException as cleanup:
                raise GroupCleanupError('unable to verify all owned descendants exited: '+str(cleanup)) from exc
        exc.group_cleanup_verified = True
        exc.group_returncode = process.returncode if process is not None else None
        raise
    finally:
        for sig,handler in previous.items():
            signal.signal(sig,handler)
