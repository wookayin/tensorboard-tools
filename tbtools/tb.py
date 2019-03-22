#!/usr/bin/env python

from __future__ import print_function

import subprocess
import argparse
import os.path
import sys
import socket
import contextlib
import psutil  # pip install psutil

parser = argparse.ArgumentParser(description=r'''
Launch tensorboard on multiple directories in an easy way.
''')
parser.add_argument('--port', default=6006, type=int,
                    help='The port to use for tensorboard')
parser.add_argument('--quiet', '-q', action='store_true',
                    help='Run in silent mode')
parser.add_argument('--auto', action='append', nargs='?',
                    help='Automatically detect python process in progress; '
                         'specify pattern to filter with.')
parser.add_argument('dirs', nargs='*', type=str,
                    help='directories of train instances to monitor')

RED    = lambda msg: ("\033[0;31m") + str(msg) + ('\033[0m')
GREEN  = lambda msg: ("\033[0;32m") + str(msg) + ('\033[0m')
YELLOW = lambda msg: ("\033[0;33m") + str(msg) + ('\033[0m')
WHITE  = lambda msg: ("\033[1;37m") + str(msg) + ('\033[0m')


def get_available_port(begin, end):
    """
    Get an available port within a range [begin, end).
    Raises an exception if no available port is found.
    """
    for port in range(begin, end):
        _s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with contextlib.closing(_s) as s:
            available = s.connect_ex(('127.0.0.1', port))
            if available:
                return port
    raise RuntimeError("No available ports")


def is_cmdline_tensorboard(cmdline):
    """
    Heuristically detect whether the current process is tensorboard or not.
    """
    if len(cmdline) < 1:
        return False

    if os.path.basename(cmdline[0]) in ('python', 'python3'):
        return is_cmdline_tensorboard(cmdline[1:])
    elif os.path.basename(cmdline[0]) == 'tensorboard':
        return True
    return False


def scan_train_dirs(*patterns):
    print(YELLOW("Auto-scanning train_dirs from running processes ..."))

    def _scan():
        for proc in psutil.process_iter():
            try:
                for f in proc.open_files():
                    if is_cmdline_tensorboard(proc.cmdline()):
                        continue
                    if '.tfevents.' in f.path:
                        yield proc, f
            except psutil.AccessDenied:
                pass

    dirs = []
    for proc, f in _scan():
        dirname = os.path.dirname(f.path)
        ctime = os.path.getctime(f.path)
        print(" Detected %s from process %s" % (f.path, WHITE(proc.pid)))
        if not patterns or any(p in f.path for p in patterns):
            dirs.append(dirname)

    return list(sorted(set(dirs)))

def main():
    args, unknown_args = parser.parse_known_args()
    args.dirs = [s for s in args.dirs if os.path.isdir(s)]

    if args.auto is not None:  # -- auto flag given
        if args.dirs:
            print(RED('Error: --auto flag should be used without dirs'))
            return 1
        args.auto = [s for s in args.auto if s]
        args.dirs = scan_train_dirs(*args.auto)

    if not args.dirs:
        print(RED('Error: No valid directories to watch. '))
        parser.print_usage()
        return 1

    for s in args.dirs:
        print(GREEN('Monitoring %s ...' % s))
    print('')

    port = get_available_port(args.port, args.port + 100)
    print(YELLOW("Tensorboard Running at port {} !".format(port)))

    cmd = ['tensorboard',
           '--port', str(port),
           '--logdir', ','.join(["%s:%s" % (os.path.basename(s), s) for s in args.dirs]),
           # TODO make additional TF parameters configurable
           '--samples_per_plugin', 'images=100',
           ]
    if args.quiet:
        cmd += [' 2>/dev/null']

    print(WHITE(subprocess.list2cmdline(cmd)))
    print('', flush=True)

    # Change tmux pane/window title
    title_msg = u"(tb:%d) %s" % (port, " ".join([os.path.basename(s) for s in args.dirs]))
    sys.stdout.buffer.write(b'\033]2;' + title_msg.encode() + b'\007')
    sys.stdout.flush()

    # Disable CUDA for tensorboard by default
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.call(cmd, shell=False, env=my_env)


if __name__ == '__main__':
    sys.exit(main())
