from translation.languages import (
    LANGUAGES
)


class LanguageManager:

    @staticmethod
    def get_code(
        language_name
    ):

        return LANGUAGES.get(
            language_name
        )

    @staticmethod
    def get_names():

        return list(
            LANGUAGES.keys()
        )