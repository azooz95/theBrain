import http.server
import socketserver
import threading
import webbrowser
import urllib.parse
import json
import os

from O365 import Account, FileSystemTokenBackend
from dotenv import load_dotenv
from config.config import file_paths

load_dotenv()

client_id = os.getenv("O365_CLIENT_ID")
client_secret = os.getenv("O365_CLIENT_SECRET")
credentials = (client_id, client_secret)

SCOPES = [
    "https://graph.microsoft.com/Mail.ReadWrite",
    "https://graph.microsoft.com/Mail.Send",
    "https://graph.microsoft.com/Calendars.ReadWrite",
    "https://graph.microsoft.com/MailboxSettings.ReadWrite",
]

TOKEN_DIR = file_paths.microsoft_token_dir
FLOW_DIR = file_paths.microsoft_flow_dir

class OAuthHandler(http.server.BaseHTTPRequestHandler):
    last_url = None

    def do_GET(self):
        OAuthHandler.last_url = self.path

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        self.wfile.write(b"<h1>You may close this window</h1>")

    def log_message(self, *args, **kwargs):
        # silence logs
        return


def microsoft_run_local_server(port=5000, user="default", open_browser=True):

    redirect_uri = f"http://localhost:{port}"

    # Prepare backend
    token_backend = FileSystemTokenBackend(
        token_path=TOKEN_DIR,
        token_filename=f"token_{user}.txt"
    )

    account = Account(credentials, token_backend=token_backend)

    if account.is_authenticated:
        print("Already authenticated.")
        return account

    # STEP 1 – Get URL + FLOW
    url, flow = account.con.get_authorization_url(
        requested_scopes=SCOPES,
        redirect_uri=redirect_uri
    )

    # save flow
    os.makedirs(FLOW_DIR, exist_ok=True)
    with open(f"{FLOW_DIR}/flow_{user}.json", "w") as f:
        json.dump(flow, f)

    # STEP 2 – Start local server
    handler = OAuthHandler
    httpd = socketserver.TCPServer(("localhost", port), handler)

    # run in background thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    print(f"Local server running on http://localhost:{port}")
    print("Open the browser to authenticate...")

    if open_browser:
        webbrowser.open(url, new=1)

    # STEP 3 – Wait for redirect
    print("Waiting for authentication response...")
    while OAuthHandler.last_url is None:
        pass

    # stop server
    httpd.shutdown()
    httpd.server_close()

    # full URL
    response_url = f"{redirect_uri}{OAuthHandler.last_url}"
    print("Response URL:", response_url)

    # STEP 4 – Exchange code for token
    with open(f"{FLOW_DIR}/flow_{user}.json") as f:
        loaded_flow = json.load(f)

    result = account.con.request_token(
        response_url,
        flow=loaded_flow
    )

    if result:
        print("Authentication successful — token saved.")
    else:
        print("Authentication failed.")
        return None

    return account


if __name__ == "__main__":
    account = microsoft_run_local_server(port=5000, user="default")
