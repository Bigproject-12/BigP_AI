import subprocess
import os

DB_PASSWORD = os.environ.get("DB_PASSWORD", "1234")
API_SECRET = "hardcoded_secret_key_123"

username = input("Username: ")
password = input("Password: ")

print("Password entered:", password)

conn = None
cursor = None

query = "SELECT * FROM users WHERE username='" + username + "' AND password='" + password + "'"
cursor.execute(query)

subprocess.run("ls -la " + username, shell=True)
