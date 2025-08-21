import re
import pytz
import dateparser
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dateparser.search import search_dates
from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from dotenv import load_dotenv
import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

load_dotenv()

class MeetingSchedulingAgent:
    SCOPES = ['https://www.googleapis.com/auth/calendar']

    def __init__(self):
        self.service = self.initialize_google_calendar()
        self._history = ""

    def initialize_google_calendar(self) -> Any:
        creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
        if not creds_path or not os.path.exists(creds_path):
            raise FileNotFoundError("❌ GOOGLE_CREDENTIALS_PATH not set or file does not exist.")

        token_path = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
        creds = None

       
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)

        
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None  

        
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, self.SCOPES)
           
            creds = flow.run_local_server(port=3000, access_type='offline', prompt='consent')
            with open(token_path, "w") as f:
                f.write(creds.to_json())

        
        return build("calendar", "v3", credentials=creds)

    def run(self, input: str) -> str:
        if not input or len(input.strip()) < 5:
            return "❗ Please describe the meeting you'd like to schedule "

        self._history += f" {input.strip()}"
        meeting_details = self.parse_meeting_request(self._history, allow_partial=True)

        missing = []
        if not meeting_details.get("start"):
            missing.append("time")
        if not meeting_details.get("duration"):
            missing.append("duration")
        if not meeting_details.get("participants"):
            missing.append("participants")

        if missing:
            prompts = {
                "time": "⏰ What time would you like the meeting to start?",
                "duration": "⏱️ How long should the meeting last?",
                "participants": "📧 Who should be invited to the meeting?"
            }
            response = "⚠️ I need a bit more info to schedule your meeting:\n"
            for field in missing:
                response += f"• {prompts[field]}\n"
            response += "\nPlease provide the missing details so I can proceed."
            return response

        # Reset history after successful scheduling
        self._history = ""

        event, meet_link = self.add_event_to_calendar(meeting_details)
        if not event:
            return "❌ Failed to create the calendar event."

        self.configure_reminders(event.get("id"))
        if meeting_details.get("participants"):
            self.invite_participants(event.get("id"), meeting_details["participants"])

        start_dt = datetime.fromisoformat(meeting_details["start"]).astimezone(
            pytz.timezone(meeting_details["timezone"])
        )
        end_dt = datetime.fromisoformat(meeting_details["end"]).astimezone(
            pytz.timezone(meeting_details["timezone"])
        )
        duration_min = int((end_dt - start_dt).total_seconds() / 60)

        formatted_time = start_dt.strftime("%A, %B %d at %I:%M %p")
        timezone_label = start_dt.strftime("%Z")

        summary = f"✅ Your meeting has been scheduled for **{formatted_time} ({timezone_label})** and will last **{duration_min} minutes**."
        if meeting_details.get("participants"):
            summary += f"\n📧 **Participants:** {', '.join(meeting_details['participants'])}"
        if meeting_details.get("add_meet") and meet_link:
            summary += f"\n🔗 **Join via Google Meet:** [Click to join]({meet_link})"
        summary += "\n🔔 A reminder will be sent 30 minutes before the event."

        return summary

    def parse_meeting_request(self, user_input: str, allow_partial: bool = False) -> Optional[Dict[str, Any]]:
        tf = TimezoneFinder()
        geolocator = Nominatim(user_agent="timezone_locator")

        results = search_dates(user_input, settings={'PREFER_DATES_FROM': 'future'})
        meeting_start = results[0][1] if results else None

        if not results and re.search(r'\b(now|immediately)\b', user_input, re.IGNORECASE):
            meeting_start = datetime.now()

        # Enhanced duration parsing: accepts "30 min" and "for 30 minutes"
        duration_match = re.search(
            r'(?:for\s*)?(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?)',
            user_input,
            re.IGNORECASE
        )
        if duration_match:
            value = float(duration_match.group(1))
            unit = duration_match.group(2).lower()
            duration_minutes = int(value * 60) if 'hour' in unit else int(value)
        else:
            duration_minutes = None

        timezone_match = re.search(r'\b([A-Za-z]+/[A-Za-z_]+)\b', user_input)
        user_timezone = timezone_match.group(1) if timezone_match else "UTC"

        if not timezone_match:
            city_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', user_input)
            if city_match:
                try:
                    location = geolocator.geocode(city_match.group(1))
                    if location:
                        user_timezone = tf.timezone_at(lat=location.latitude, lng=location.longitude) or "UTC"
                except:
                    user_timezone = "UTC"

        try:
            tz = pytz.timezone(user_timezone)
        except:
            tz = pytz.UTC
            user_timezone = "UTC"

        if meeting_start:
            meeting_start = meeting_start if meeting_start.tzinfo else tz.localize(meeting_start)
            meeting_start_utc = meeting_start.astimezone(pytz.UTC)
            meeting_end_utc = meeting_start_utc + timedelta(minutes=duration_minutes or 60)
        else:
            meeting_start_utc, meeting_end_utc = None, None

        participant_emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_input)

        return {
            "summary": "Scheduled Meeting",
            "start": meeting_start_utc.isoformat() if meeting_start_utc else None,
            "end": meeting_end_utc.isoformat() if meeting_end_utc else None,
            "timezone": user_timezone,
            "add_meet": True,
            "participants": participant_emails,
            "duration": duration_minutes if duration_minutes else None
        }

    def add_event_to_calendar(self, details: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        event = {
            "summary": details["summary"],
            "start": {"dateTime": details["start"], "timeZone": details["timezone"]},
            "end": {"dateTime": details["end"], "timeZone": details["timezone"]}
        }
        if details.get("add_meet"):
            event["conferenceData"] = {
                "createRequest": {
                    "requestId": f"meet-{datetime.now().timestamp()}",
                    "conferenceSolutionKey": {"type": "hangoutsMeet"}
                }
            }
        try:
            created_event = self.service.events().insert(
                calendarId="primary", body=event, conferenceDataVersion=1
            ).execute()
            link = created_event.get("conferenceData", {}).get("entryPoints", [{}])[0].get("uri", "No Link")
            return created_event, link
        except Exception:
            return None, "No Link"

    def configure_reminders(self, event_id: str):
        try:
            event = self.service.events().get(calendarId="primary", eventId=event_id).execute()
            event["reminders"] = {
                "useDefault": False,
                "overrides": [
                    {"method": "email", "minutes": 30},
                    {"method": "popup", "minutes": 10}
                ]
            }
            self.service.events().update(calendarId="primary", eventId=event_id, body=event).execute()
        except:
            pass

    def invite_participants(self, event_id: str, participants: List[str]):
        try:
            event = self.service.events().get(calendarId="primary", eventId=event_id).execute()
            event["attendees"] = [{"email": email} for email in participants]
            self.service.events().update(calendarId="primary", eventId=event_id, body=event).execute()
        except:
            pass
