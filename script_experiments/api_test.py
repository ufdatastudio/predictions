from dotenv import load_dotenv
import os
load_dotenv()


if __name__ == "__main__":
    api_key = os.getenv("NAVI_GATOR_API_KEY")
    print(api_key)
