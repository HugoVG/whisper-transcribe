"""Tests for pipeline.realtime.entities — live entity extraction (nl + en)."""

import unittest
from datetime import date

from pipeline.realtime.entities import (
    collapse_repeated_digits,
    extract_entities,
    normalize_number_words,
)

TODAY = date(2026, 11, 26)  # a Thursday


class TestNormalizeNumberWords(unittest.TestCase):
    def test_dutch_run(self):
        self.assertEqual(normalize_number_words("vijf zes zeven vier"), "5674")

    def test_english_run(self):
        self.assertEqual(normalize_number_words("one two three"), "123")

    def test_single_word_kept_without_keyword(self):
        # "een" is a common Dutch article — must not become "1"
        self.assertEqual(normalize_number_words("een moment"), "een moment")

    def test_single_word_after_keyword(self):
        self.assertEqual(normalize_number_words("ticket nummer vijf"), "ticket nummer 5")


class TestCollapseRepeated(unittest.TestCase):
    def test_double(self):
        self.assertEqual(collapse_repeated_digits("56745674"), "5674")

    def test_no_double(self):
        self.assertEqual(collapse_repeated_digits("5674"), "5674")
        self.assertEqual(collapse_repeated_digits("0101"), "01")


class TestTicketNumbers(unittest.TestCase):
    def test_ti_prefix_spoken_digits(self):
        ents = extract_entities("Ja hoor, TI 0 1", today=TODAY)
        tickets = [e for e in ents if e.type == "ticket_number"]
        self.assertEqual(len(tickets), 1)
        self.assertEqual(tickets[0].value, "TI01")

    def test_ticket_keyword(self):
        ents = extract_entities("het ticket nummer is 12345", today=TODAY)
        tickets = [e for e in ents if e.type == "ticket_number"]
        self.assertTrue(any(e.value == "12345" for e in tickets))

    def test_number_words_ticket(self):
        ents = extract_entities("TI vijf zes", today=TODAY)
        tickets = [e for e in ents if e.type == "ticket_number"]
        self.assertEqual(tickets[0].value, "TI56")

    def test_long_digit_run(self):
        ents = extract_entities("5674", today=TODAY)
        numbers = [e for e in ents if e.type == "number"]
        self.assertEqual(numbers[0].value, "5674")


class TestDeviceNumbers(unittest.TestCase):
    def test_bare_serial(self):
        ents = extract_entities("ja, met device nummer NL012345", today=TODAY)
        devices = [e for e in ents if e.type == "device_number"]
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].value, "NL012345")

    def test_bare_pattern_alone(self):
        ents = extract_entities("het gaat om NL012345", today=TODAY)
        devices = [e for e in ents if e.type == "device_number"]
        self.assertEqual(devices[0].value, "NL012345")


class TestDates(unittest.TestCase):
    def test_dutch_weekday_date(self):
        ents = extract_entities("op donderdag 5 december", today=TODAY)
        dates = [e for e in ents if e.type == "date"]
        self.assertTrue(any(e.value == "2026-12-05" for e in dates))

    def test_dutch_date_no_weekday(self):
        ents = extract_entities("afspraak op 5 december", today=TODAY)
        dates = [e for e in ents if e.type == "date"]
        self.assertEqual(dates[0].value, "2026-12-05")

    def test_english_date(self):
        ents = extract_entities("on Thursday, December 5", today=TODAY)
        dates = [e for e in ents if e.type == "date"]
        self.assertEqual(dates[0].value, "2026-12-05")

    def test_english_date_day_first(self):
        ents = extract_entities("on the 5th of december", today=TODAY)
        dates = [e for e in ents if e.type == "date"]
        self.assertTrue(any(e.value == "2026-12-05" for e in dates))

    def test_date_next_year_when_passed(self):
        ents = extract_entities("3 januari", today=TODAY)
        dates = [e for e in ents if e.type == "date"]
        self.assertEqual(dates[0].value, "2027-01-03")

    def test_relative_tomorrow(self):
        ents = extract_entities("morgen", today=TODAY)
        dates = [e for e in ents if e.type == "date"]
        self.assertEqual(dates[0].value, "2026-11-27")

    def test_relative_today_en(self):
        ents = extract_entities("today", today=TODAY)
        dates = [e for e in ents if e.type == "date"]
        self.assertEqual(dates[0].value, "2026-11-26")


class TestContact(unittest.TestCase):
    def test_email(self):
        ents = extract_entities("stuur naar jan@example.nl", today=TODAY)
        emails = [e for e in ents if e.type == "email"]
        self.assertEqual(emails[0].value, "jan@example.nl")

    def test_phone(self):
        ents = extract_entities("bel me op 0612345678", today=TODAY)
        phones = [e for e in ents if e.type == "phone_number"]
        self.assertEqual(phones[0].value, "0612345678")

    def test_no_entities_in_plain_text(self):
        ents = extract_entities("Hallo, goeden dag", today=TODAY)
        self.assertEqual(ents, [])


if __name__ == "__main__":
    unittest.main()
