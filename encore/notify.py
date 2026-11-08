"""Stay reachable when home alone: phrases can also send a push notification.

Uses ntfy.sh (free, no account). The family installs the ntfy app and subscribes
to the same topic. Topics are public by name, so use a long random topic.

This is a convenience, not a medical alert system: it only fires when the user
can still produce a cue in range of the mic or camera. It does not detect falls.
"""

import urllib.request

ACTIONS = {
    "checkin": ("{name} checked in", "{text}", "default", "white_check_mark"),
    "callme": ("{name} would like a call", "{text}", "high", "telephone_receiver"),
    "help": ("{name} needs help now", "{text}", "urgent", "rotating_light"),
}


class Notifier:
    def __init__(self, topic=None, name: str = "Encore user",
                 server: str = "https://ntfy.sh", display=print):
        self.topic = topic
        self.name = name
        self.server = server.rstrip("/")
        self.display = display

    def handle(self, action: str, text: str) -> bool:
        if not action or not action.startswith("notify:"):
            return False
        kind = action.split(":", 1)[1]
        title, body, priority, tag = ACTIONS.get(kind, ("{name}", "{text}", "default", "speech_balloon"))
        return self.send(title.format(name=self.name), body.format(text=text), priority, tag)

    def send(self, title: str, message: str, priority: str = "default", tags: str = "") -> bool:
        if not self.topic:
            self.display(f"[notify] no topic set, would send: {title} | {message}")
            return False
        req = urllib.request.Request(
            f"{self.server}/{self.topic}", data=message.encode("utf-8"), method="POST",
            headers={"Title": title, "Priority": priority, "Tags": tags})
        try:
            with urllib.request.urlopen(req, timeout=5):
                pass
            self.display(f"[notify] sent: {title}")
            return True
        except Exception as e:  # network failures must never crash the speech engine
            self.display(f"[notify] failed to send ({e}). Message was: {title}")
            return False
