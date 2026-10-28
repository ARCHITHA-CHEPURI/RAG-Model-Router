import unittest
from unittest.mock import Mock, patch

import requests

from dashboard import request_json


class DashboardRequestTests(unittest.TestCase):
    @patch("dashboard.requests.request")
    def test_gets_json_from_configured_api(self, request):
        response = Mock()
        response.json.return_value = {"status": "ok"}
        request.return_value = response

        result = request_json(
            "GET",
            "http://localhost:8000/",
            "/health",
        )

        self.assertEqual(result, {"status": "ok"})
        request.assert_called_once_with(
            "GET",
            "http://localhost:8000/health",
            json=None,
            params=None,
            timeout=180,
        )
        response.raise_for_status.assert_called_once()

    @patch("dashboard.requests.request")
    def test_includes_error_detail_when_api_is_unavailable(self, request):
        request.side_effect = requests.ConnectionError("connection refused")

        with self.assertRaisesRegex(RuntimeError, "Could not reach router API"):
            request_json("GET", "http://localhost:8000", "/health")

    @patch("dashboard.requests.request")
    def test_rejects_non_object_api_responses(self, request):
        response = Mock()
        response.json.return_value = ["unexpected"]
        request.return_value = response

        with self.assertRaisesRegex(RuntimeError, "unexpected response"):
            request_json("GET", "http://localhost:8000", "/api/runs")


if __name__ == "__main__":
    unittest.main()
