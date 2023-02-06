import sys
import requests
import time
import os

TOKEN = "ENTER_TELEGRAM_TOKEN"
CHAT_ID = "ENTER_CHAT_ID"


class Logger:
    def __init__(self):
        self.start_time = None
        self.latest_message_id = None

    def start(self):
        self.start_time = time.time()

    def send(self, message):
        res = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}")
        res = res.json()
        if "result" in res and "message_id" in res["result"]:
            self.latest_message_id = res["result"]["message_id"]

    def update_latest(self, message):
        if self.latest_message_id is None:
            self.send(message)
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/editMessageText?chat_id={CHAT_ID}&message_id={self.latest_message_id}&text={message}")

    def to_timely_msg(self, message):
        elapsed_time = time.time() - self.start_time
        time_string = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        user = os.environ["USER"]
        return f"{user} [{time_string}]: {message}"

    def send_timely(self, message):
        timely_message = self.to_timely_msg(message)
        self.send(timely_message)

    def send_userly(self, message):
        user = os.environ["USER"]
        self.send(f"{user}: {message}")

    def update_latest_timely(self, message):
        timely_message = self.to_timely_msg(message)
        self.update_latest(timely_message)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:python logger.py <message>")
        sys.exit()
    logger = Logger()
    logger.send_userly(" ".join(sys.argv[1:]))
