"""Tests for extract_zones module."""

import os

import pytest

from extract_zones import extract_zones

DIR = os.path.dirname(__file__)
FULL_IMAGE = os.path.join(DIR, "2026-02-10_08-53-49.png")
CROPPED_IMAGE = os.path.join(DIR, "2026-02-10_08-53-49-0000.png")


class TestCroppedImage:
    """Tests using the cropped text-only image."""

    def test_first_four_lines(self):
        """Verify the first four extracted lines match expected zone labels."""
        lines = extract_zones(CROPPED_IMAGE)
        assert len(lines) >= 4, f"Expected at least 4 lines, got {len(lines)}"

        # Line 1: Resistance Zone 26268.00-26360.50 (Short-term Bias Confirming Zone)
        assert "Resistance Zone" in lines[0]
        assert "26268" in lines[0]
        assert "26360" in lines[0]
        assert "Short-term Bias Confirming Zone" in lines[0]

        # Line 2: 24H Range Extreme High
        assert lines[1] == "24H Range Extreme High"

        # Line 3: Resistance Zone 26174.00-26237.00 (Intraday Bias Confirming Zone)
        assert "Resistance Zone" in lines[2]
        assert "Confirming Zone" in lines[2]

        # Line 4: 24H Range Exhaustion High
        assert lines[3] == "24H Range Exhaustion High"

    def test_extracts_multiple_zones(self):
        """Verify that multiple zone labels are extracted."""
        lines = extract_zones(CROPPED_IMAGE)
        assert len(lines) >= 15, f"Expected at least 15 lines, got {len(lines)}"

    def test_contains_support_and_resistance(self):
        """Verify both support and resistance zones are found."""
        lines = extract_zones(CROPPED_IMAGE)
        text = "\n".join(lines)
        assert "Resistance Zone" in text
        assert "Support Zone" in text

    def test_contains_key_zone_labels(self):
        """Verify specific zone labels from the image are extracted."""
        lines = extract_zones(CROPPED_IMAGE)
        text = "\n".join(lines)

        # These zones should be reliably detected
        assert "25950" in text, "Should detect Resistance Zone around 25950"
        assert "25448" in text, "Should detect Resistance Zone around 25448"
        assert "25373" in text, "Should detect Initial Resistance around 25373"
        assert "25241" in text, "Should detect Aggressive Support around 25241"
        assert "25155" in text or "2515" in text, "Should detect Pre-Market Support"
        assert "24806" in text, "Should detect Support Zone around 24806"
        assert "24407" in text, "Should detect Support Zone around 24407"
        assert "24144" in text, "Should detect Support Zone around 24144"


class TestFullChartImage:
    """Tests using the full chart screenshot."""

    def test_first_four_lines(self):
        """Verify the first four extracted lines match expected zone labels."""
        lines = extract_zones(FULL_IMAGE)
        assert len(lines) >= 4, f"Expected at least 4 lines, got {len(lines)}"

        assert "Resistance Zone" in lines[0]
        assert "26268" in lines[0]
        assert "Short-term Bias Confirming Zone" in lines[0]

        assert lines[1] == "24H Range Extreme High"

        assert "Resistance Zone" in lines[2]
        assert "Confirming Zone" in lines[2]

        assert lines[3] == "24H Range Exhaustion High"

    def test_extracts_multiple_zones(self):
        """Verify that multiple zone labels are extracted."""
        lines = extract_zones(FULL_IMAGE)
        assert len(lines) >= 15, f"Expected at least 15 lines, got {len(lines)}"
