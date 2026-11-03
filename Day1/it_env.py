"""Simulated IT environment shared by the labs. Nothing here touches a real machine.

One application server (app-01) with a real fault in it, one developer workstation (ws-114),
a software catalogue and three users. Tools read and mutate this dictionary, so an agent's
actions have consequences it can observe on its next step. Call reset_env() before each run.

The fault on app-01 is a three-hop chain: a deploy turned on DEBUG logging, the logs filled
/var, and the payments service now crashes on start. Restarting without freeing space fails.
"""
ENV, AUDIT_LOG, TICKETS, HYPOTHESES = {}, [], [], []


def reset_env():
    """Restore the simulated environment to its broken starting state."""
    AUDIT_LOG.clear()
    TICKETS.clear()
    HYPOTHESES.clear()
    ENV.clear()
    ENV.update(
        users={
            "ravi": {"role": "admin"},
            "asha": {"role": "developer"},
            "meera": {"role": "contractor"},
        },
        catalog={
            "vscode": {"approved": True, "requires_admin": False},
            "git": {"approved": True, "requires_admin": False},
            "postgresql-client": {"approved": True, "requires_admin": False},
            "docker-desktop": {"approved": True, "requires_admin": True},
            "wireshark": {"approved": False, "requires_admin": True},
        },
        hosts={
            "ws-114": {"installed": [], "services": {}, "disk": {"/": 38}, "big_dirs": {}, "logs": {}, "changes": [],
                       "memory": "16 GB total, 6.2 GB used"},
            "app-01": {
                "installed": ["nginx", "redis"],
                "services": {
                    "payments-api": {"state": "failed", "detail": "exit code 1, 14 restarts since 09:42"},
                    "nginx": {"state": "active", "detail": "running since yesterday 18:05"},
                    "redis": {"state": "active", "detail": "running since yesterday 18:05"},
                },
                "disk": {"/": 46, "/var": 100, "/data": 61},
                "big_dirs": {"/var/log/payments": "38G", "/var/lib/docker": "9G", "/var/log/nginx": "1.1G"},
                "memory": "16 GB total, 9.1 GB used (redis 3.2 GB, java 2.9 GB), swap unused",
                "changes": [
                    "today 09:10 deploy payments-api v2.4.1 (change note: log level set to DEBUG for tracing)",
                    "yesterday 18:00 kernel security patch, host rebooted",
                ],
                "logs": {
                    "payments-api": [
                        "09:41:58 DEBUG ledger.sync batch=88121 rows=5000",
                        "09:42:03 ERROR OSError: [Errno 28] No space left on device: '/var/log/payments/app.log'",
                        "09:42:03 FATAL startup aborted: cannot open log file",
                        "09:42:04 systemd: payments-api.service: Main process exited, status=1/FAILURE",
                    ],
                    "nginx": [
                        "09:42:05 [error] connect() failed (111: Connection refused) while connecting to upstream payments-api",
                        "09:42:05 502 Bad Gateway POST /v1/charge",
                    ],
                    "redis": ["09:30:00 Background saving terminated with success"],
                },
            },
        },
    )


reset_env()


def get_host(name):
    return ENV["hosts"].get(name.strip().lower())
