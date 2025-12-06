import os
import json
import socket
import threading
from typing import List, Optional

from dateparser.search import search_dates
import psutil
from dotenv import load_dotenv

# Google API
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# OAuth error surfaces
from oauthlib.oauth2.rfc6749.errors import (
    OAuth2Error,
    AccessDeniedError,
    MismatchingStateError,
)

from config.config import file_paths

load_dotenv()

_google_account_lock = threading.Lock()
_google_account_singleton = None

class CredentialError(Exception):
    pass

class GoogleAccount:

    def __init__(self, credentials_path: str | None, token_path: str | None, scopes: List[str], service_name: str, version: str) -> None:

        # Set credentials path explicitly (env or config)
        self.credentials_path = (
            credentials_path
            or getattr(file_paths, "google_credentials_path", None)
        )
        if not self.credentials_path or not os.path.exists(self.credentials_path):
            raise CredentialError(
                "Google credentials.json not found. Set GOOGLE_CREDENTIALS_PATH or file_paths.google_credentials_path."
            )

        self.scopes = scopes
        self.service_name = service_name
        self.version = version
        self.service = None

        # token path from config or alongside credentials.json
        self.token_path = token_path or getattr(file_paths, "google_token_path", None)

        self._ready = threading.Event()
        self._error: Optional[Exception] = None
        self._auth_lock = threading.Lock()
        self._auth_started = False

    def is_refresh_needed(self) -> bool:
        if not os.path.exists(self.token_path):
            return True
        try:
            creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)
            return creds.expired and bool(getattr(creds, "refresh_token", None))
        except Exception:
            return True

    def activate_by_token(self):
        creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)
        return creds

    def _is_port_free(self, host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.connect_ex((host, port)) != 0

    def _pick_web_redirect(self) -> tuple[str, int]:
        # Choose a free port from allowlist (must be registered in Google Console)
        host = os.getenv("GOOGLE_REDIRECT_HOST", "127.0.0.1")
        ports_env = os.getenv("GOOGLE_REDIRECT_PORTS", os.getenv("GOOGLE_REDIRECT_PORT", "3000"))
        ports = [int(p.strip()) for p in str(ports_env).split(",") if p.strip().isdigit()]
        if not ports:
            ports = [3000]
        for p in ports:
            if self._is_port_free(host, p):
                print(
                    "Using Web OAuth client. Ensure this Redirect URI is authorized in Google Cloud Console:\n"
                    f"- http://{host}:{p}/\n"
                    "(APIs & Services â Credentials â OAuth 2.0 Client IDs â Authorized redirect URIs)"
                )
                return host, p
        # If none are free, report and still return the first to show a clear error later
        print(
            "All listed redirect ports are busy. Free one or update GOOGLE_REDIRECT_PORTS.\n"
            f"Tried: {ports}"
        )
        return host, ports[0]

    def authenticate(self):
        with self._auth_lock:
            if self._auth_started:
                return  # someone else is doing it
            self._auth_started = True

        try:
            if self.is_refresh_needed():
                # Detect client type
                with open(self.credentials_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                client_type = "installed" if "installed" in cfg else ("web" if "web" in cfg else None)

                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, self.scopes)

                try:
                    if client_type == "installed":
                        # Desktop client supports random port
                        creds = flow.run_local_server(
                            host="127.0.0.1",
                            port=0,
                            access_type="offline",
                            prompt="consent",
                            open_browser=True,
                            redirect_uri_trailing_slash=True,
                        )
                    else:
                        # Web client must use a registered, fixed redirect
                        host, port = self._pick_web_redirect()
                        try:
                            creds = flow.run_local_server(
                                host=host,
                                port=port,
                                access_type="offline",
                                prompt="consent",
                                open_browser=True,
                                redirect_uri_trailing_slash=True,
                            )
                        except OSError as e:
                            print(f"Local sign-in server failed on {host}:{port}: {e}")
                            raise

                except (AccessDeniedError, MismatchingStateError) as e:
                    self._error = e
                    print("Sign-in was cancelled. Please try again and allow access.")
                    return None
                except OAuth2Error as e:
                    self._error = e
                    print("Couldnât complete Google sign-in. Please try again.")
                    return None
                except OSError as e:
                    # Console flow often fails with Web clients; surface actionable steps
                    self._error = e
                    print(
                        "Couldnât start the local sign-in server. Fix by either:\n"
                        "- Using a Desktop OAuth client (preferred), or\n"
                        "- Freeing the selected port and adding its redirect URI in Google Console."
                    )
                    return None
                except Exception as e:
                    self._error = e
                    print(f"Unexpected error during Google sign-in: {e}")
                    return None

                with open(self.token_path, "w") as f:
                    f.write(creds.to_json())
            else:
                creds = self.activate_by_token()

            try:
                self.service = build(self.service_name, self.version, credentials=creds)
            except Exception as e:
                self._error = e
                raise CredentialError("Couldnât create Google Calendar service.") from e
        finally:
            self._ready.set()
        return self.service

    def wait_until_ready(self, timeout: Optional[int] = 180) -> None:
        if self.service:
            return
        if not self._ready.wait(timeout=timeout):
            raise CredentialError("Google authentication did not complete in time.")
        if self._error:
            raise CredentialError(f"Google authentication failed: {self._error}") from self._error

    def events(self):
        # Wait for async auth to complete before returning the API handle
        if not self.service:
            self.wait_until_ready(timeout=int(os.getenv("GOOGLE_AUTH_TIMEOUT", "180")))
        if not self.service:
            raise CredentialError("Google Calendar service is not initialized.")
        return self.service.events()
    
    def _clean_stuck_ports(self, ports):
        """Force-close any process that might be blocking OAuth ports."""
        for port in ports:
            for conn in psutil.net_connections():
                if conn.laddr.port == port:
                    try:
                        p = psutil.Process(conn.pid)
                        p.terminate()
                    except Exception:
                        pass

_google_accounts: dict[str, GoogleAccount] = {}
_google_account_lock = threading.Lock()

# def google_authenticate(token_path: str) -> GoogleAccount:
#     with _google_account_lock:
#         # Return existing session for same user/token
#         if token_path in _google_accounts:
#             return _google_accounts[token_path]

#         scopes = ["https://www.googleapis.com/auth/calendar"]
#         google_account = GoogleAccount(
#             token_path=token_path,
#             credentials_path=file_paths.google_credentials_path,
#             scopes=scopes,
#             service_name="calendar",
#             version="v3"
#         )

#         # Run authentication synchronously for safety
#         google_account.authenticate()

#         # Store instance for re-use
#         _google_accounts[token_path] = google_account
#         return google_account


def google_authenticate(token_path: str) -> 'GoogleAccount':
    with _google_account_lock:
        # Return existing instance if already created
        if token_path in _google_accounts:
            return _google_accounts[token_path]

        scopes = ["https://www.googleapis.com/auth/calendar"]
        google_account = GoogleAccount(
            token_path=token_path,
            credentials_path=file_paths.google_credentials_path,
            scopes=scopes,
            service_name="calendar",
            version="v3"
        )

        # Run authentication in a separate thread (non-blocking)
        auth_thread = threading.Thread(target=google_account.authenticate, daemon=True)
        auth_thread.start()

        # Store instance immediately for reuse
        _google_accounts[token_path] = google_account
        return google_account

    


if __name__ == "__main__":
    google_authenticate('token.json')