"""Reject exhausted native retrieval errors without changing search or EM."""
import json


class RetrievalFailure(RuntimeError):
    def __init__(self, reason, row):
        self.reason, self.row = reason, row
        # Keep constructor args serializable when a Ray worker reports failure.
        super().__init__(reason, row)

    def __str__(self):
        return f'Native Search-QA retrieval failed ({self.reason}, row {self.row}); no score or update for this batch'

    def receipt(self):
        # Do not persist remote error bodies, URLs, or request headers.
        return {'status': 'invalid_retrieval_failure', 'reason': self.reason, 'row_in_batch': self.row}


def failure_reason(observation):
    """Inspect the native tool envelope, not text quoted inside a passage."""
    if not isinstance(observation, str):
        return None
    text = observation.strip()
    begin, end = '<information>', '</information>'
    if not (text.startswith(begin) and text.endswith(end)):
        return None
    try:
        payload = json.loads(text[len(begin):-len(end)])
    except (TypeError, ValueError):
        return None
    message = payload.get('result') if isinstance(payload, dict) else None
    if not isinstance(message, str):
        return None
    if message.startswith('Search error:'):
        return 'transport_or_http_error'
    if message.startswith('Error processing search results:'):
        return 'malformed_search_results'
    if (message == 'Search request failed or timed out after retries.'
            or message.startswith('Unknown API state (')):
        return 'unavailable_retriever'
    return None


def require_retrieval_ok(anchors):
    for row, anchor in enumerate(anchors):
        reason = failure_reason(anchor)
        if reason:
            raise RetrievalFailure(reason, row)
