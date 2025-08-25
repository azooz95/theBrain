import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import dateparser
from dateparser.search import search_dates
import pytz
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
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

load_dotenv()


class CredentialError(Exception):
    """User-friendly credential/authentication problem."""


class MeetingSchedulingAgent:
    """
    Schedules meetings on Google Calendar from a natural-language prompt.
    Replies default to a friendly, detailed markdown summary. 
    """

    SCOPES = ["https://www.googleapis.com/auth/calendar"]

    def __init__(self) -> None:
        self._history: str = ""
        self.service = None
        self._init_error: Optional[str] = None
        self.response_style = os.getenv("RESPONSE_STYLE", "summary").lower().strip()
        try:
            self.service = self.initialize_google_calendar()
        except CredentialError as e:
            self._init_error = str(e)
        except FileNotFoundError as e:
            self._init_error = str(e)
        except Exception:
            self._init_error = "Unexpected error while setting up Google Calendar."

    # Google auth with hardening

    def initialize_google_calendar(self):
        creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
        if not os.path.exists(creds_path):
            raise CredentialError(
                "OAuth client file is missing. Please set GOOGLE_CREDENTIALS_PATH."
            )

        token_path = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
        creds: Optional[Credentials] = None

        # Load existing token.json safely
        try:
            if os.path.exists(token_path):
                creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)
        except Exception:
            creds = None  # corrupt token -> force new login

        # Try to refresh if possible
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                creds = None  # revoked/invalid refresh -> re-auth
            except Exception:
                creds = None

        # If still not valid, run the OAuth flow
        if not creds or not creds.valid:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, self.SCOPES)
                creds = flow.run_local_server(
                    port=0, access_type="offline", prompt="consent", open_browser=True
                )
            except (AccessDeniedError, MismatchingStateError) as e:
                raise CredentialError(
                    "Sign-in was cancelled. Please try again and allow access."
                ) from e
            except OAuth2Error as e:
                raise CredentialError(
                    "Couldn’t complete Google sign-in. Please try again."
                ) from e
            except OSError as e:
                raise CredentialError(
                    "Couldn’t start the local sign-in server. Try again on a different port/network."
                ) from e
            except Exception as e:
                raise CredentialError("Unexpected error during Google sign-in.") from e

            # Safely persist token (non-fatal if it fails)
            try:
                with open(token_path, "w") as f:
                    f.write(creds.to_json())
            except Exception:
                pass

        # Build Calendar service
        try:
            return build("calendar", "v3", credentials=creds)
        except HttpError as e:
            raise CredentialError(
                "Google Calendar API error. Your account may lack permissions."
            ) from e
        except Exception as e:
            raise CredentialError("Couldn’t connect to Google Calendar.") from e


    # Natural-language parsing

    def _extract_participants(self, text: str) -> List[str]:
        emails = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        return list(dict.fromkeys([e.strip("<>,. ") for e in emails]))

    def _extract_title(self, text: str) -> str:
        m = re.search(
            r"(?:about|re|regarding|titled)\s*[:\- ]+(.+?)(?:\.|,| with | at | on | tomorrow| next | in \d+|$)",
            text,
            flags=re.I,
        )
        if m:
            return m.group(1).strip().title()
        return "Meeting"

    def _extract_time_range(self, text: str, tz: str) -> Tuple[datetime, datetime]:
        now = datetime.now(pytz.timezone(tz))
        found = search_dates(text, settings={"RETURN_AS_TIMEZONE_AWARE": True})
        if found:
            future = [dt for _, dt in found if dt > now]
            start = future[0] if future else found[-1][1]
        else:
            parsed = dateparser.parse(
                text, settings={"RETURN_AS_TIMEZONE_AWARE": True, "PREFER_DATES_FROM": "future"}
            )
            start = parsed if parsed else now + timedelta(hours=1)
        start = start.astimezone(pytz.timezone(tz))
        m = re.search(r"for\s+(\d+)\s*(minutes?|mins?|hours?|hrs?)", text, flags=re.I)
        if m:
            qty = int(m.group(1))
            unit = m.group(2).lower()
            dur = timedelta(minutes=qty) if "min" in unit else timedelta(hours=qty)
        else:
            dur = timedelta(minutes=30)
        end = start + dur
        return start, end

    def _extract_location(self, text: str) -> Optional[str]:
        if re.search(r"\b(meet|google meet|zoom|teams|video|online)\b", text, re.I):
            return None
        m = re.search(r"(?:at|in)\s+([A-Za-z0-9 \-_,']{3,})", text, flags=re.I)
        if not m:
            return None
        location_query = m.group(1).strip().rstrip(".")
        try:
            geolocator = Nominatim(user_agent="smart_schedule_agent")
            loc = geolocator.geocode(location_query, timeout=5)
            if loc:
                return f"{loc.address}"
        except Exception:
            pass
        return location_query

    def _guess_timezone_from_location(self, location: Optional[str]) -> str:
        default_tz = os.getenv("DEFAULT_TIMEZONE", "UTC")
        if not location:
            return default_tz
        try:
            geolocator = Nominatim(user_agent="smart_schedule_agent")
            loc = geolocator.geocode(location, timeout=5)
            if not loc:
                return default_tz
            tf = TimezoneFinder()
            tzname = tf.timezone_at(lng=loc.longitude, lat=loc.latitude)
            return tzname or default_tz
        except Exception:
            return default_tz


    # Calendar ops

    def add_event_to_calendar(self, details: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Returns (created_event, meet_link_or_placeholder)
        """
        start: datetime = details["start"]
        end: datetime = details["end"]
        attendees = [{"email": e} for e in details.get("participants", [])]
        location = details.get("location")
        add_meet = details.get("add_meet", True)  # default create Meet link

        event = {
            "summary": details["title"],
            "location": location,
            "description": details.get("description", ""),
            "start": {"dateTime": start.isoformat(), "timeZone": start.tzinfo.key if hasattr(start.tzinfo, "key") else str(start.tzinfo)},
            "end": {"dateTime": end.isoformat(), "timeZone": end.tzinfo.key if hasattr(end.tzinfo, "key") else str(end.tzinfo)},
            "attendees": attendees,
            "reminders": {"useDefault": True},
        }

        if add_meet:
            event["conferenceData"] = {"createRequest": {"requestId": f"req-{int(datetime.now().timestamp())}"}}

        try:
            created_event = (
                self.service.events()
                .insert(calendarId="primary", body=event, conferenceDataVersion=1)
                .execute()
            )
            link = (
                created_event.get("hangoutLink")
                or created_event.get("conferenceData", {})
                .get("entryPoints", [{}])[0]
                .get("uri", "No link")
            )
            return created_event, (link or "No link")
        except HttpError:
            return None, "No link"
        except Exception:
            return None, "No link"


    # Reply builders

    def _format_summary(self, meeting_details: Dict[str, Any], meet_link: Optional[str]) -> str:
        start: datetime = meeting_details["start"]
        end: datetime = meeting_details["end"]
        duration_min = int((end - start).total_seconds() // 60)
        timezone_label = start.tzname() or (start.tzinfo.key if hasattr(start.tzinfo, "key") else str(start.tzinfo))
        formatted_time = start.strftime("%A, %B %d • %I:%M %p").lstrip("0")
        summary = (
            f"✅ Your meeting has been scheduled for **{formatted_time} ({timezone_label})** "
            f"and will last **{duration_min} minutes**."
        )
        if meeting_details.get("participants"):
            summary += f"\n📧 **Participants:** {', '.join(meeting_details['participants'])}"
        if meeting_details.get("add_meet") and meet_link and meet_link != "No link":
            summary += f"\n🔗 **Join via Google Meet:** [Click to join]({meet_link})"
        summary += "\n🔔 A reminder will be sent 30 minutes before the event."
        return summary

    def _format_short(self, title: str, start: datetime, meet_link: Optional[str]) -> str:
        when = start.strftime("%a %b %d, %I:%M %p").lstrip("0")
        suffix = f" Link: {meet_link}" if meet_link and meet_link != "No link" else ""
        return f"Done! “{title}” on {when}.{suffix}"


    # Main entry

    def run(self, input: str) -> str:
        # Short-circuit if auth isn’t ready
        if self._init_error:
            return f"Can’t access Calendar: {self._init_error}"

        if not input or len(input.strip()) < 5:
            return "Tell me who/when, and I’ll set it up."

        text = input.strip()
        self._history += f" {text}"

        # Extract
        participants = self._extract_participants(text)
        location_hint = self._extract_location(text)
        tz = self._guess_timezone_from_location(location_hint) if location_hint else os.getenv("DEFAULT_TIMEZONE", "UTC")
        start, end = self._extract_time_range(text, tz)
        title = self._extract_title(text)

        # Should we add a Meet link? If user explicitly says phone/office only, disable.
        add_meet = not bool(re.search(r"\b(phone call|call only|in person|office only)\b", text, re.I))

        # Build details & create
        details = {
            "title": title,
            "start": start,
            "end": end,
            "participants": participants,
            "location": location_hint,
            "description": "",
            "add_meet": add_meet,
        }
        created, meet_link = self.add_event_to_calendar(details)

        if not created:
            return "Couldn’t schedule it now—please try again in a bit."

        if self.response_style == "short":
            return self._format_short(title, start, meet_link)
        else:
            return self._format_summary(details, meet_link)


    # Optional helpers

    def clear_cached_token(self) -> bool:
        token_path = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
        try:
            if os.path.exists(token_path):
                os.remove(token_path)
            return True
        except Exception:
            return False
