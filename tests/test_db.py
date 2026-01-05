from oddsETL.db import OddsDB
from supabase import create_client, Client
import pytest
import os

@pytest.fixture
def db_fixture():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    db = OddsDB(db_url=url, db_key=key)
    return db

def test_get_event_ids_by_sport(db_fixture):
    db = db_fixture

    sport_key = "americanfootball_nfl"
    event_ids = db.get_event_ids_by_sport(sport_key)

    assert isinstance(event_ids, list)
    for event_id in event_ids:
        assert isinstance(event_id, str)
    assert len(event_ids) == len(set(event_ids))  # Ensure uniqueness