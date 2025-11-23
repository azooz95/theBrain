from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse, HTMLResponse
from O365 import Account, FileSystemTokenBackend
import json
import os
from dotenv import load_dotenv

import requests

from config.config import file_paths

load_dotenv()

TOEKEN_DIR = file_paths.microsoft_token_dir
FLOW_PATH = file_paths.microsoft_flow_dir

client_id = os.getenv("O365_CLIENT_ID")
client_secret = os.getenv("O365_CLIENT_SECRET")
credentials = (client_id, client_secret)

SCOPES = [
    "https://graph.microsoft.com/Mail.ReadWrite",
    "https://graph.microsoft.com/Mail.Send",
    "https://graph.microsoft.com/Calendars.ReadWrite",
    "https://graph.microsoft.com/MailboxSettings.ReadWrite",
]

app = FastAPI()

# -----------------------------
# Flow storage (simple local file)
# -----------------------------

def store_flow(flow: dict, flow_name: str = 'default'):
    with open(f'{FLOW_PATH}/flow_{flow_name}.json', "w") as f:
        json.dump(flow, f)

def load_flow(flow_name: str = 'default') -> dict:
    if not os.path.exists(f'{FLOW_PATH}/flow_{flow_name}.json'):
        return {}
    with open(f'{FLOW_PATH}/flow_{flow_name}.json', "r") as f:
        return json.load(f)


# -----------------------------
# Step 1 — Redirect user to Microsoft login
# -----------------------------
@app.get("/")
async def auth_step_one(request: Request, user_info: str = 'default'):

    callback_url = str(request.url_for("auth_step_two_callback"))

    token_backend = FileSystemTokenBackend(
        token_path=TOEKEN_DIR,
        token_filename=f'token_{user_info}.txt',
    )

    account = Account(credentials, token_backend=token_backend)

    if account.is_authenticated:
        return Response(content="Already authenticated", status_code=200)

    # <-- THIS RETURNS (url, flow)
    url, flow = account.con.get_authorization_url(
        requested_scopes=SCOPES,
        redirect_uri=callback_url
    )

    # **************************
    # FIX: Save the flow (contains the correct state)
    # **************************
    store_flow(flow, flow_name=user_info)

    return RedirectResponse(url=url)


# -----------------------------
# Step 2 — Callback from Azure
# -----------------------------
@app.get("/getAToken")
async def auth_step_two_callback(request: Request, user_info: str = 'default'):

    flow = load_flow(flow_name=user_info)  # Load saved flow

    requested_url = str(request.url)
    print("Requested URL:", requested_url)
    print("Loaded FLOW:", flow)

    token_backend = FileSystemTokenBackend(
        token_path=TOEKEN_DIR,
        token_filename=f'token_{user_info}.txt',
    )

    account = Account(credentials, token_backend=token_backend)

    # Exchange auth code for access/refresh token
    result = account.con.request_token(
        requested_url,
        flow=flow
    )

    if result:
        return HTMLResponse("<h1>Authentication Successful — Token Saved</h1>")
    else:
        return HTMLResponse("<h1>Authentication Failed</h1>", status_code=400)

@app.get('/test_api')
async def test_api(request: Request):

    result = await auth_step_one(request=request)

    # print(result.url)
    # if result.status_code == 307:
        # result = await auth_step_two_callback(request=request, user_info='default')

    return result.status_code
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
