import pytest
import socket as s


@pytest.yield_fixture
def socket():
    _socket = s.socket(s.AF_INET, s.SOCK_STREAM)
    yield _socket
    _socket.close()


@pytest.fixture(scope='module')
def server():
    class Dummy:
        host_port = 'localhost', 8081
        uri = 'https://%s:%s/' % host_port

    return Dummy


def test_server_connect(socket, server):
    socket.connect(server.host_port)
    assert socket
