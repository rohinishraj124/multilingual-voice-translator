import csv
import json
import os
from datetime import datetime

class HistoryManager:

    def __init__(self):

        self.database_file = (
            "history/database/history.json"
        )

        os.makedirs(
            "history/database",
            exist_ok=True
        )

        self.history = []

        self.load_history()

    def load_history(self):

        if not os.path.exists(
            self.database_file
        ):
            return

        try:

            with open(
                self.database_file,
                "r",
                encoding="utf-8"
            ) as file:

                self.history = json.load(
                    file
                )

        except Exception:

            self.history = []

    def save_history(self):

        with open(
            self.database_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.history,
                file,
                ensure_ascii=False,
                indent=4
            )

    def add_record(
        self,
        source_language,
        target_language,
        source_text,
        translated_text
    ):

        record = {
            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "source_language":
                source_language,

            "target_language":
                target_language,

            "source_text":
                source_text,

            "translated_text":
                translated_text
        }

        self.history.append(
            record
        )

        self.save_history()

    def get_history(self):

        return self.history

    def clear_history(self):

        self.history.clear()

        self.save_history()

    def export_csv(self):

        os.makedirs(
            "history/records",
            exist_ok=True
        )

        timestamp = datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S"
        )

        filename = (
            f"history/records/"
            f"translations_{timestamp}.csv"
        )

        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "timestamp",
                    "source_language",
                    "target_language",
                    "source_text",
                    "translated_text"
                ]
            )

            writer.writeheader()

            writer.writerows(
                self.history
            )

        return filename

