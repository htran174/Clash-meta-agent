import os
from dotenv import load_dotenv
import requests
from langchain_openai import ChatOpenAI

#--------------------------------------------------------
#Use this file to test if the .env is set-up correctly
#--------------------------------------------------------

# ---------------------------------------------------------
# Load .env
# ---------------------------------------------------------
load_dotenv()

CR_API_KEY = os.getenv("CR_API_KEY")
PLAYER_TAG = os.getenv("PLAYER_TAG")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("Loaded keys:")
print("  CR_API_KEY exists?", CR_API_KEY is not None)
print("  PLAYER_TAG =", PLAYER_TAG)
print("  OPENAI_API_KEY exists?", OPENAI_API_KEY is not None)

# ---------------------------------------------------------
# Test Clash Royale API
# ---------------------------------------------------------
def test_clash_royale():
    print("\nTesting Clash Royale API...")

    if not CR_API_KEY:
        print("❌ CR_API_KEY missing in .env")
        return

    tag_no_hash = PLAYER_TAG.replace("#", "")
    url = f"https://api.clashroyale.com/v1/players/%23{tag_no_hash}/battlelog"

    headers = {
        "Authorization": f"Bearer {CR_API_KEY}"
    }

    try:
        r = requests.get(url, headers=headers)
        print("Status Code:", r.status_code)

        if r.status_code == 200:
            data = r.json()
            print("Battlelog length:", len(data))
            print("Sample fields:", list(data[0].keys()))
            print("✅ Clash Royale API key works!")
        else:
            print("❌ Clash Royale API failed:")
            print(r.text)

    except Exception as e:
        print("❌ Error talking to Clash Royale API:")
        print(e)


# ---------------------------------------------------------
# Test OpenAI API (LangChain)
# ---------------------------------------------------------
def test_openai():
    print("\nTesting OpenAI API...")

    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY missing in .env")
        return

    try:
        llm = ChatOpenAI(model="gpt-4.1-mini")
        resp = llm.invoke("Say 'keys working'")
        print("OpenAI Response:", resp.content)
        print("✅ OpenAI key works!")

    except Exception as e:
        print("❌ Error talking to OpenAI:")
        print(e)


# ---------------------------------------------------------
# Run tests
# ---------------------------------------------------------
if __name__ == "__main__":
    test_clash_royale()
    test_openai()
