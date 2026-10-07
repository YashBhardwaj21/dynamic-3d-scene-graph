"""Integration test: Recovery and State Reset on Reconnection."""

import socket
import time

from ros2_ws.src.d455_bridge.d455_bridge.d455_receiver import D455Receiver


def test_receiver_survives_client_disconnect_and_reconnect():
    temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    temp_sock.bind(("127.0.0.1", 0))
    port = temp_sock.getsockname()[1]
    temp_sock.close()

    receiver = D455Receiver(host="127.0.0.1", port=port, queue_size=4)

    try:
        time.sleep(0.1)

        # Connection 1
        client1 = socket.create_connection(("127.0.0.1", port), timeout=2.0)
        time.sleep(0.1)
        assert receiver.total_reconnections >= 1

        # Sudden disconnect
        client1.close()
        time.sleep(0.2)

        # Connection 2 (Reconnect)
        client2 = socket.create_connection(("127.0.0.1", port), timeout=2.0)
        time.sleep(0.1)
        assert receiver.total_reconnections >= 2

        client2.close()

    finally:
        receiver.destroy_node()
