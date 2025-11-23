import os
import json
import socket
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Type

import dateparser
from dateparser.search import search_dates
import psutil
import pytz
from geopy.geocoders import Nominatim
from dotenv import load_dotenv

# Google API
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

# OAuth error surfaces
from oauthlib.oauth2.rfc6749.errors import (
    OAuth2Error,
    AccessDeniedError,
    MismatchingStateError,
)
from google.auth.exceptions import RefreshError

from config.config import file_paths

import asyncio

load_dotenv()

import pydantic 
from pydantic import BaseModel
from langchain.tools import BaseTool
from langchain_community.tools.office365.base import O365BaseTool

class CredentialError(Exception):
    pass

# Single instance to avoid multiple concurrent auth attempts
# _google_account_singleton = None
# _google_account_lock = threading.Lock()

# class GoogleAccount:

#     def __init__(self, credentials_path: str | None, token_path: str | None, scopes: List[str], service_name: str, version: str) -> None:

#         # Set credentials path explicitly (env or config)
#         self.credentials_path = (
#             credentials_path
#             or getattr(file_paths, "google_credentials_path", None)
#         )
#         if not self.credentials_path or not os.path.exists(self.credentials_path):
#             raise CredentialError(
#                 "Google credentials.json not found. Set GOOGLE_CREDENTIALS_PATH or file_paths.google_credentials_path."
#             )

#         self.scopes = scopes
#         self.service_name = service_name
#         self.version = version
#         self.service = None

#         # token path from config or alongside credentials.json
#         self.token_path = token_path or getattr(file_paths, "google_token_path", None)

#         self._ready = threading.Event()
#         self._error: Optional[Exception] = None
#         self._auth_lock = threading.Lock()
#         self._auth_started = False

#     def is_refresh_needed(self) -> bool:
#         if not os.path.exists(self.token_path):
#             return True
#         try:
#             creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)
#             return creds.expired and bool(getattr(creds, "refresh_token", None))
#         except Exception:
#             return True

#     def activate_by_token(self):
#         creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)
#         return creds

#     def _is_port_free(self, host: str, port: int) -> bool:
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#             return s.connect_ex((host, port)) != 0

#     def _pick_web_redirect(self) -> tuple[str, int]:
#         # Choose a free port from allowlist (must be registered in Google Console)
#         host = os.getenv("GOOGLE_REDIRECT_HOST", "127.0.0.1")
#         ports_env = os.getenv("GOOGLE_REDIRECT_PORTS", os.getenv("GOOGLE_REDIRECT_PORT", "3000"))
#         ports = [int(p.strip()) for p in str(ports_env).split(",") if p.strip().isdigit()]
#         if not ports:
#             ports = [3000]
#         for p in ports:
#             if self._is_port_free(host, p):
#                 print(
#                     "Using Web OAuth client. Ensure this Redirect URI is authorized in Google Cloud Console:\n"
#                     f"- http://{host}:{p}/\n"
#                     "(APIs & Services → Credentials → OAuth 2.0 Client IDs → Authorized redirect URIs)"
#                 )
#                 return host, p
#         # If none are free, report and still return the first to show a clear error later
#         print(
#             "All listed redirect ports are busy. Free one or update GOOGLE_REDIRECT_PORTS.\n"
#             f"Tried: {ports}"
#         )
#         return host, ports[0]

#     def authenticate(self):
#         with self._auth_lock:
#             if self._auth_started:
#                 return  # someone else is doing it
#             self._auth_started = True

#         try:
#             if self.is_refresh_needed():
#                 # Detect client type
#                 with open(self.credentials_path, "r", encoding="utf-8") as f:
#                     cfg = json.load(f)
#                 client_type = "installed" if "installed" in cfg else ("web" if "web" in cfg else None)

#                 flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, self.scopes)

#                 try:
#                     if client_type == "installed":
#                         # Desktop client supports random port
#                         creds = flow.run_local_server(
#                             host="127.0.0.1",
#                             port=0,
#                             access_type="offline",
#                             prompt="consent",
#                             open_browser=True,
#                             redirect_uri_trailing_slash=True,
#                         )
#                     else:
#                         # Web client must use a registered, fixed redirect
#                         host, port = self._pick_web_redirect()
#                         try:
#                             creds = flow.run_local_server(
#                                 host=host,
#                                 port=port,
#                                 access_type="offline",
#                                 prompt="consent",
#                                 open_browser=True,
#                                 redirect_uri_trailing_slash=True,
#                             )
#                         except OSError as e:
#                             print(f"Local sign-in server failed on {host}:{port}: {e}")
#                             raise

#                 except (AccessDeniedError, MismatchingStateError) as e:
#                     self._error = e
#                     print("Sign-in was cancelled. Please try again and allow access.")
#                     return None
#                 except OAuth2Error as e:
#                     self._error = e
#                     print("Couldn’t complete Google sign-in. Please try again.")
#                     return None
#                 except OSError as e:
#                     # Console flow often fails with Web clients; surface actionable steps
#                     self._error = e
#                     print(
#                         "Couldn’t start the local sign-in server. Fix by either:\n"
#                         "- Using a Desktop OAuth client (preferred), or\n"
#                         "- Freeing the selected port and adding its redirect URI in Google Console."
#                     )
#                     return None
#                 except Exception as e:
#                     self._error = e
#                     print(f"Unexpected error during Google sign-in: {e}")
#                     return None

#                 with open(self.token_path, "w") as f:
#                     f.write(creds.to_json())
#             else:
#                 creds = self.activate_by_token()

#             try:
#                 self.service = build(self.service_name, self.version, credentials=creds)
#             except Exception as e:
#                 self._error = e
#                 raise CredentialError("Couldn’t create Google Calendar service.") from e
#         finally:
#             self._ready.set()
#         return self.service

#     def wait_until_ready(self, timeout: Optional[int] = 180) -> None:
#         if self.service:
#             return
#         if not self._ready.wait(timeout=timeout):
#             raise CredentialError("Google authentication did not complete in time.")
#         if self._error:
#             raise CredentialError(f"Google authentication failed: {self._error}") from self._error

#     def events(self):
#         # Wait for async auth to complete before returning the API handle
#         if not self.service:
#             self.wait_until_ready(timeout=int(os.getenv("GOOGLE_AUTH_TIMEOUT", "180")))
#         if not self.service:
#             raise CredentialError("Google Calendar service is not initialized.")
#         return self.service.events()
    
#     def _clean_stuck_ports(self, ports):
#         """Force-close any process that might be blocking OAuth ports."""
#         for port in ports:
#             for conn in psutil.net_connections():
#                 if conn.laddr.port == port:
#                     try:
#                         p = psutil.Process(conn.pid)
#                         p.terminate()
#                     except Exception:
#                         pass

# def authunticate() -> GoogleAccount:
#     global _google_account_singleton
#     with _google_account_lock:
#         if _google_account_singleton:
#             return _google_account_singleton

#         scopes = ["https://www.googleapis.com/auth/calendar"]
#         google_account = GoogleAccount(
#             credentials_path=file_paths.google_crenditials_path,
#             scopes=scopes,
#             service_name="calendar",
#             version="v3"
#         )
#         t = threading.Thread(target=google_account.authenticate, daemon=True)
#         t.start()
#         _google_account_singleton = google_account
#         return google_account

from pydantic import Field
from authentications.google_authenticator import GoogleAccount, google_authenticate

class GoogleBaseTool(BaseTool):
    # model_config = {"arbitrary_types_allowed": True}
    path_token: Optional[str] = Field(default=None, description="the path to the google token file")
    account: Optional[GoogleAccount] = Field(default=None, exclude=True)

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.account:
            self.account = google_authenticate(token_path=self.path_token)


class GoogleMeetArgsSchema(BaseModel):
    """Schema for creating a Google Meet link."""
    title: Optional[str] = Field(default="Instant Meeting", description="Title for the meeting")
    start_time: Optional[str] = Field(default=None, description="Optional start time (default: now)")
    end_time: Optional[str] = Field(default=None, description="Optional end time (default: +1 hour)")

class GoogleCalanderArgsSchema(BaseModel):
    """
        a schema for google calendar arguments
    """
    title: str = Field(description="the title of the event to be created")
    start_time: str = Field(description="the start time of the event to be created")
    end_time: str = Field(description="the end time of the event to be created")
    participants: Optional[List[str]] = Field(
        default=None, description="a list of email addresses of the participants to be invited to the event"
    )
    location: Optional[str] = Field(
        default=None, description="the location of the event to be created"
    )
    description: Optional[str] = Field(
        default=None, description="the description of the event to be created"
    )
    add_meet: Optional[bool] = Field(
        default=True, description="whether to add a Google Meet link to the event"
    )


class GoogleCalendarToolkit(GoogleBaseTool):

    """
        a toolkit for Google Calendar operations
    """

    name: str = "google_calendar_toolkit"
    args_schema: type[pydantic.BaseModel] = GoogleCalanderArgsSchema
    description: str = (
        "A toolkit for Google Calendar operations, including creating events with "
        "specified details such as title, time, participants, location, description and Google Meet link."
    )

    model_config = pydantic.ConfigDict(
        extra="forbid",
    )

    def _run(self,
        title: str,
        start_time: str,
        end_time: str,
        participants: Optional[List[str]] = None,
        location: Optional[str] = None,
        description: Optional[str] = None,
        add_meet: Optional[bool] = True,
    ) -> Dict[str, Any]:
        """
        Returns (created_event, meet_link_or_placeholder)
        """

        # parse and ensure timezone-aware datetimes
        start_time = dateparser.parse(start_time)
        end_time = dateparser.parse(end_time)
        if start_time is None or end_time is None:
            raise ValueError("Could not parse start_time or end_time")

        # Ensure timezone-aware; fallback to DEFAULT_TIMEZONE (UTC)
        default_tz = os.getenv("DEFAULT_TIMEZONE", "UTC")
        tz = pytz.timezone(default_tz)
        if start_time.tzinfo is None:
            start_time = tz.localize(start_time)
        if end_time.tzinfo is None:
            end_time = tz.localize(end_time)

        # Google expects attendees as list of {"email": ...}
        attendees = [{"email": e} for e in (participants or [])]

        event = {
            "summary": title,
            "location": location,
            "description": description,
            "start": {"dateTime": start_time.isoformat(), "timeZone": start_time.tzinfo.key if hasattr(start_time.tzinfo, "key") else str(start_time.tzinfo)},
            "end": {"dateTime": end_time.isoformat(), "timeZone": end_time.tzinfo.key if hasattr(end_time.tzinfo, "key") else str(end_time.tzinfo)},
            "attendees": attendees,
            "reminders": {"useDefault": True},
        }

        if add_meet:
            event["conferenceData"] = {"createRequest": {"requestId": f"req-{int(datetime.now().timestamp())}"}}

        try:
            created_event = (
                self.account.events()
                .insert(calendarId="primary", body=event, conferenceDataVersion=1)
                .execute()
            )
            link = (
                created_event.get("hangoutLink")
                or created_event.get("conferenceData", {})
                .get("entryPoints", [{}])[0]
                .get("uri", "No link")
            )
            return {"created_event": created_event, "meet_link": link}
        except AssertionError as e:
            raise ValueError(f"Error: {e}")

    

class GoogleCreateMeet(GoogleBaseTool):
    """Tool for creating a new Google Meet meeting."""

    name: str = "google_create_meet"
    description: str = "Creates a new Google Meet link using Google Calendar API."
    args_schema: Type[BaseModel] = GoogleMeetArgsSchema

    def _run(
        self,
        title: str = "Instant Meeting",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        timezone: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Creates a Google Meet event and returns the meeting link."""

        # Parse times or use defaults
        if not timezone:
            timezone = "UTC"

        tz = pytz.timezone(timezone)
        start = dateparser.parse(start_time) if start_time else datetime.now(tz)
        end = dateparser.parse(end_time) if end_time else (start + timedelta(hours=1))

        event = {
            "summary": title,
            "start": {"dateTime": start.isoformat(), "timeZone": tz.zone},
            "end": {"dateTime": end.isoformat(), "timeZone": tz.zone},
            "conferenceData": {
                "createRequest": {
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                    "requestId": f"meet-{int(datetime.now().timestamp())}",
                }
            },
        }

        try:
            created_event = (
                self.account.events()
                .insert(calendarId="primary", body=event, conferenceDataVersion=1)
                .execute()
            )
            link = (
                created_event.get("hangoutLink")
                or created_event.get("conferenceData", {})
                .get("entryPoints", [{}])[0]
                .get("uri", "No link")
            )
            return f"Google Meet created successfully!\n🔗 Link: {link}"
        except Exception as e:
            raise RuntimeError(f"Failed to create Google Meet: {e}") from e



class O365TeamsMeetingSchema(BaseModel):
    """Schema for creating an O365 Teams meeting."""
    subject: Optional[str] = Field(description="Title for the meeting")
    start_time: Optional[str] = Field(description="Start time of the meeting")
    end_time: Optional[str] = Field(description="End time of the meeting")
    participants: Optional[List[str]] = Field(
        default=None, description="List of email addresses of the participants"
    )

class O365CreateTeamsMeeting(O365BaseTool):
    """Tool for creating a new O365 Teams meeting."""

    name: str = "o365_create_teams_meeting"
    description: str = "Creates a new O365 Teams meeting using Microsoft Graph API."
    args_schema: Type[BaseModel] = O365TeamsMeetingSchema

    def _run(
        self,
        subject: Optional[str] = "Instant Meeting",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        participants: Optional[List[str]] = None,
        timezone: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Creates an O365 Teams meeting and returns the meeting link."""
        # Implementation would go here
        schedule = self.account.schedule()
        calendar = schedule.get_default_calendar()
        event = calendar.new_event()

        if not timezone:
            timezone = "UTC"
        tz = pytz.timezone(timezone)
        event.subject = subject
        event.start = dateparser.parse(start_time) if start_time else datetime.now(tz)
        event.end = dateparser.parse(end_time) if end_time else (event.start + timedelta(hours=1))

        for email in participants:
            event.attendees.add(email)

        event.is_online_meeting = True
        event.online_meeting_provider = "teamsForBusiness"  # <— required

        event.save()
        return event.online_meeting.join_url
if __name__ == "__main__":
    google_calendar_toolkit = GoogleCalendarToolkit()
    response = google_calendar_toolkit._run(
        title="Team Sync Meeting",
        start_time="2024-07-01 10:00 AM",
        end_time="2024-07-01 11:00 AM",
        participants=["alice@example.com", "bob@example.com"],
    )   
    print(response)