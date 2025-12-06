import jwt
import datetime
from jwt import PyJWTError as JWTError

from fastapi import Depends, HTTPException, status, FastAPI
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()

SECRET_KEY = "JWT_brain_BE_SECRETTS"
FAKE_USER = {
    "username": "admin",
    "email": "aziz.alhaj@gmail.com",
    "user_id": "1"
}

ALGORITHM = "HS256"
DEFAULT_EXPIRATION_HOURS = 24 * 365 # 1 year
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def generate_token(user_data, expiration_hours=DEFAULT_EXPIRATION_HOURS):
    payload = {
        'user_id': user_data["user_id"],
        'username': user_data["username"],
        'email': user_data["email"],
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=expiration_hours)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    return token

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_sub": False})
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_sub": False})
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/users")
def read_users_me(current_user: str = Depends(get_current_user)):
    return {"user_id": current_user, "email": FAKE_USER["email"]}

if __name__ == "__main__":
    pass
    # user_id = "1"  # Example
    token = generate_token(FAKE_USER)
    print(token)
    print(f"Generated JWT Token for user: {token}")
    print(verify_token(token))

    # decode_token = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    # print(f"Decoded Token: {decode_token}")

    # import uvicorn
    # uvicorn.run(app, host="localhost", port=8000)
