from fimicyber.loaders.euvsdisinfo import _map_row


def test_euvsdisinfo_debunk_id_not_in_description():
    row = {
        "article_id": "article-123",
        "debunk_id": "leaky-group-999",
        "keywords": "election fraud; military aid",
        "article_publisher": "example publisher",
        "article_domain": "example.test",
        "article_url": "https://example.test/a",
        "article_language": "English",
        "class": "disinformation",
        "debunk_date": "2024-01-02",
    }

    ev = _map_row(row, 0)

    assert ev is not None
    assert ev.campaign_id == "euvsdisinfo:debunk:leaky-group-999"
    assert "leaky-group-999" not in ev.description
    assert "Debunk group" not in ev.description
