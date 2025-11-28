import socket
import threading
from handler import SimpleHandler

HOST = "0.0.0.0"
PORT = 9092

def main():
    print(f"\n🚀 Starting Simple VAD AudioSocket Server")
    print(f"🌐 Host: {HOST}")
    print(f"🔌 Port: {PORT}")
    print(f"{'='*60}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print("🔧 Socket created")
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print("⚙️ Socket options configured (SO_REUSEADDR)")
        s.bind((HOST, PORT))
        print(f"🔗 Socket bound to {HOST}:{PORT}")
        s.listen()
        print("👂 Socket listening for connections")
        print(f"\n✅ Server ready! Waiting for Asterisk connections...\n")

        connection_count = 0

        while True:
            try:
                conn, addr = s.accept()
                connection_count += 1
                print(f"\n🎉 [SERVER] New connection #{connection_count} from {addr}")

                # Create handler instance for this connection
                handler = SimpleHandler(conn, addr)

                # Create new handler thread
                handler_thread = threading.Thread(
                    target=handler.handle_client, 
                    daemon=True,
                    name=f"Handler-{connection_count}"
                )
                handler_thread.start()
                print(f"🧵 [SERVER] Started handler thread: {handler_thread.name}")

            except KeyboardInterrupt:
                print("\n⛔ [SERVER] Keyboard interrupt received")
                print("🛑 [SERVER] Shutting down gracefully...")
                break
            except Exception as e:
                print(f"💥 [SERVER] Server error: {e}")
                import traceback
                traceback.print_exc()  


if __name__ == "__main__":
    main()