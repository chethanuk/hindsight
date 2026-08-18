import http.server
import socket
import socketserver
import threading
import time
from unittest.mock import patch

import pytest

from hindsight_embed.daemon_embed_manager import DaemonEmbedManager


class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class V6OnlyServer(socketserver.TCPServer):
    address_family = socket.AF_INET6
    allow_reuse_address = True

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        super().server_bind()


class V4OnlyServer(socketserver.TCPServer):
    address_family = socket.AF_INET
    allow_reuse_address = True


@pytest.fixture
def v6_ui_server():
    if not socket.has_ipv6:
        pytest.skip("IPv6 not supported by Python build")

    try:
        server = V6OnlyServer(("::1", 0), HealthHandler)
    except OSError:
        pytest.skip("IPv6 loopback interface unavailable")

    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    port = server.server_address[1]
    yield port
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture
def v4_ui_server():
    try:
        server = V4OnlyServer(("127.0.0.1", 0), HealthHandler)
    except OSError:
        pytest.skip("IPv4 loopback interface unavailable")

    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    port = server.server_address[1]
    yield port
    server.shutdown()
    server.server_close()
    thread.join()


def test_is_ui_running_v6_only(v6_ui_server):
    manager = DaemonEmbedManager()
    assert manager.is_ui_running("", ui_port=v6_ui_server) is True


def test_is_ui_running_v4_only(v4_ui_server):
    manager = DaemonEmbedManager()
    assert manager.is_ui_running("", ui_port=v4_ui_server) is True


def test_is_ui_running_nothing_listening():
    # Find an unused port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        closed_port = s.getsockname()[1]

    manager = DaemonEmbedManager()
    start_time = time.time()
    res = manager.is_ui_running("", ui_port=closed_port)
    elapsed = time.time() - start_time

    assert res is False
    assert elapsed < 5.0


@pytest.mark.parametrize(
    "netstat_output, expected_pid",
    [
        (
            "  TCP    [::1]:9177    [::]:0    LISTENING    4321",
            4321,
        ),
        (
            "  TCP    127.0.0.1:9177    0.0.0.0:0    LISTENING    1234",
            1234,
        ),
        (
            "  Active Connections",
            None,
        ),
        (
            "  Proto  Local Address          Foreign Address        State           PID",
            None,
        ),
        (
            "  TCP    127.0.0.1:54321    127.0.0.1:9177    ESTABLISHED    9999",
            None,
        ),
    ],
)
def test_find_pid_on_port_windows_parsing(netstat_output, expected_pid):
    manager = DaemonEmbedManager()
    with patch("platform.system", return_value="Windows"):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = f"{netstat_output}\n"
            mock_run.return_value.returncode = 0
            pid = manager._find_pid_on_port(9177)
            assert pid == expected_pid
