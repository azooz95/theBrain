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
from config.config import file_paths

class MeetingSchedulingAgent:
    SCOPES = ['https://www.googleapis.com/auth/calendar']

    def __init__(self):
        self.service = self.initialize_google_calendar()
        self._history = ""  # Accumulate multi-turn inputs

    def initialize_google_calendar(self) -> Any:
        flow = InstalledAppFlow.from_client_secrets_file(file_paths.google_crenditials_path, self.SCOPES)
        print("Redirect URI being used:", flow.redirect_uri)
        creds = flow.run_local_server(port=3000, timeout_seconds=10)

        return build('calendar', 'v3', credentials=creds)

    def run(self, input: str) -> str:
        if not input or len(input.strip()) < 5:
            return "❗ Please describe the meeting you'd like to schedule (e.g. time, duration, participants)."

        self._history += f" {input.strip()}"
        meeting_details = self.parse_meeting_request(self._history, allow_partial=True)

        missing = [field for field in ["start", "duration", "participants"] if not meeting_details.get(field)]
        if missing:
            prompts = {
                "start": "⏰ What time would you like the meeting to start?",
                "duration": "⏱️ How long should the meeting last?",
                "participants": "📧 Who should be invited to the meeting?"
            }
            response = "⚠️ I need a bit more info to schedule your meeting:\n"
            response += "\n".join(f"• {prompts[field]}" for field in missing)
            response += "\n\nPlease provide the missing details so I can proceed."
            return response

        self._history = ""  # Reset history after scheduling

        event, meet_link = self.add_event_to_calendar(meeting_details)
        if not event:
            return "❌ Failed to create the calendar event."

        self.configure_reminders(event.get("id"))
        if meeting_details.get("participants"):
            self.invite_participants(event.get("id"), meeting_details["participants"])

        start_dt = datetime.fromisoformat(meeting_details["start"]).astimezone(pytz.timezone(meeting_details["timezone"]))
        duration_min = int((datetime.fromisoformat(meeting_details["end"]) - start_dt).total_seconds() / 60)

        summary = f"✅ Your meeting has been scheduled for **{start_dt.strftime('%A, %B %d at %I:%M %p')} ({start_dt.strftime('%Z')})** and will last **{duration_min} minutes**."
        if meeting_details.get("participants"):
            summary += f"\n📧 **Participants:** {', '.join(meeting_details['participants'])}"
        if meeting_details.get("add_meet") and meet_link:
            summary += f"\n🔗 **Join via Google Meet:** [Click to join]({meet_link})"
        summary += "\n🔔 A reminder will be sent 30 minutes before the event."

        return summary

    def view_events(self, keyword: Optional[str] = None, start_date: Optional[str] = None) -> str:
        now = datetime.utcnow().isoformat() + "Z"
        events_result = self.service.events().list(
            calendarId='primary',
            timeMin=start_date or now,
            maxResults=10,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])

        if not events:
            return "📅 No upcoming events found."

        event_list = "\n".join(f"• {event['summary']} ({event['start'].get('dateTime', event['start'].get('date'))})"
                               for event in events if not keyword or keyword.lower() in event['summary'].lower())

        return f"📅 **Upcoming Events:**\n{event_list}"

    def update_event(self, event_id: str, new_details: Dict[str, Any]) -> str:
        try:
            event = self.service.events().get(calendarId="primary", eventId=event_id).execute()
            event.update(new_details)
            updated_event = self.service.events().update(calendarId="primary", eventId=event_id, body=event).execute()
            return f"✅ Event updated successfully: {updated_event.get('summary')}"
        except:
            return "❌ Failed to update the event."

    def delete_event(self, event_id: str) -> str:
        try:
            self.service.events().delete(calendarId="primary", eventId=event_id).execute()
            return "🗑️ Event deleted successfully."
        except:
            return "❌ Failed to delete the event."

    def parse_meeting_request(self, user_input: str, allow_partial: bool = False) -> Optional[Dict[str, Any]]:
        tf = TimezoneFinder()
        geolocator = Nominatim(user_agent="timezone_locator")

        results = search_dates(user_input, settings={'PREFER_DATES_FROM': 'future'})
        meeting_start = results[0][1] if results else None

        duration_match = re.search(r'(?:for\s*)?(\d+(?:\.\d+)?)\s*(minutes?|hours?)', user_input, re.IGNORECASE)
        duration_minutes = int(float(duration_match.group(1)) * (60 if 'hour' in duration_match.group(2).lower() else 1)) if duration_match else None

        timezone_match = re.search(r'\b([A-Za-z]+/[A-Za-z_]+)\b', user_input)
        user_timezone = timezone_match.group(1) if timezone_match else "UTC"

        participant_emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_input)

        return {
            "summary": "Scheduled Meeting",
            "start": meeting_start.isoformat() if meeting_start else None,
            "end": (meeting_start + timedelta(minutes=duration_minutes)).isoformat() if meeting_start and duration_minutes else None,
            "timezone": user_timezone,
            "add_meet": True,
            "participants": participant_emails,
            "duration": duration_minutes
        }

# Usage Example:
# agent = MeetingSchedulingAgent()
# print(agent.run("Schedule a meeting tomorrow at 3 PM for 45 minutes with alice@example.com and bob@example.com"))
# print(agent.view_events())