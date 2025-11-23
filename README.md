# theBrain
let your office talk


## Main Technology 

- Python 3.12
- Milvus
- LangGraph
- Gemini API
- Google Workspace (Email, Calander)

## project structure

To install dependencies

```
pip install -e .
```

Run the following code in the terminal for linux
```
export PYTHONPATH=$(pwd)
```

Run the following for Windows

```
set PYTHONPATH=%cd%
```

you can find any model in this [google drive link](https://drive.google.com/drive/folders/1QhqoEk-XC5qz1uP_bZHgE-0sA0Hihu0s?usp=sharing) to download

To run the project api
```
uvicorn apis.main:app --reload --port 8000
```
#  theBrain - Environment Configuration

This project uses environment variables to manage API keys, credentials, and service configurations. Below is a list of all required variables, their purposes, and default behaviors.

---

##  Environment Variables

###  API Keys & Service Tokens

| Variable | Description |
|---------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key for accessing generative AI services. |
| `GOOGLE_API_KEY` | General Google API key used across various GCP services. |
| `GOOGLE_CREDENTIALS_PATH` | Path to your GCP credentials JSON file. Example: `C:/Users/Omneya/theBrain/credentials.json`. |
| `OPENAI_API_KEY` | API key for accessing OpenAI's GPT models. |
| `SENDGRID_API_KEY` | Used for sending emails via the SendGrid service. |
| `APP_PASSWORD` | App-specific password for the Gmail SMTP service. Never use your actual Gmail password. |
| `SMTP_SERVER` | SMTP server address for sending emails. Default is `smtp.gmail.com`. |
| `SMTP_PORT` | SMTP server port (default secure port is `465`). |
| `SENDER_EMAIL` | Email used as sender (e.g., `aabdoh@joe13th.com`). |
| `EMAIL_ADDRESS` | Secondary contact or sender email (e.g., `joe13brain@gmail.com`). |

---

###  AI & Vector Database

| Variable | Description |
|----------|-------------|
| `MILVUS_URI` | URL of the Milvus (Zilliz) vector database instance. |
| `MILVUS_TOKEN` | API token for authenticating with Milvus. |

---

### Trello Integration

| Variable | Description |
|----------|-------------|
| `TRELLO_API_KEY` | API key to connect with Trello. |
| `TRELLO_TOKEN` | Auth token used to access Trello boards and cards. |
| `BASE_URL` | Base URL for Trello API (usually `https://api.trello.com/1`). |

---

###  Slack Bot Integration

| Variable | Description |
|----------|-------------|
| `SLACK_BOT_TOKEN` | Bot token for accessing Slack workspace via API. |
| `SLACK_USER_ID` | Slack user ID to target for bot interactions. |

---

##  Database Configuration

| Variable | Description |
|----------|-------------|
| `DATABASE_BASE_URL` | SQLAlchemy-compatible database URI. Supports MySQL, SQLite, etc. |

** Default Behavior:**  
If `DATABASE_BASE_URL` is not set or the connection fails, 
you can execute the `testSQL.py` script in the database folder and using the value sqlite:///theBrain/db/huge_test.db .



