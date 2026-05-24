import json
import os

class PromptMemory:
    def __init__(self):
        self.path = "history/prompts.json"
        os.makedirs("history", exist_ok=True)

        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)

    def save_prompt(self, prompt):
        data = self.load_history()
        data.append(prompt)

        with open(self.path, "w") as f:
            json.dump(data[-100:], f, indent=4)

    def load_history(self):
        with open(self.path, "r") as f:
            return json.load(f)

    def analyze_history(self):
        history = self.load_history()
        joined = " ".join(history[-15:])
        return joined