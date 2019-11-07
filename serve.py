#!/usr/bin/env python3

import multiprocessing
import os
import subprocess
import signal
import sys

cpu_count = multiprocessing.cpu_count()

model_server_timeout = os.environ.get('MODEL_SERVER_TIMEOUT', 60)
model_server_workers = os.environ.get('MODEL_SERVER_WORKERS', cpu_count)



def sigterm_handler(nginx_pid, gunicorn_pid):
    try:
        os.kill(nginx_pid, signal.SIGQUIT)
    except OSError:
        pass
    try:
        os.kill(gunicorn_pid, signal.SIGTERM)
    except OSError:
        pass

    sys.exit(0)



def start_server():
    print('Starting server with {} workers.'.format(model_server_workers))



    ## link logstreams to standardout, this way they will show in the container logs
    subprocess.check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
    subprocess.check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])

    nginx = subprocess.Popen(['nginx', '-c', '{}/config/nginx.config'.format(os.getcwd())])
    gunicorn = subprocess.Popen(['gunicorn',
                                 '-k', 'gevent',
                                 '--timeout', str(model_server_timeout),
                                 '-b', 'unix:/tmp/gunicorn.sock',
                                 '-w', str(model_server_workers),
                                 'agentapi.api-aws'])


    signal.signal(signal.SIGTERM, lambda a, b: sigterm_handler(nginx.pid, gunicorn.pid))

    # If either subprocess exits, so do we.
    pids = set([nginx.pid, gunicorn.pid])
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break

    sigterm_handler(nginx.pid, gunicorn.pid)
    print('Inference server exiting')


if __name__ == '__main__':
    start_server()

