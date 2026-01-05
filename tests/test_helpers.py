import json
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import pandas as pd
from datetime import datetime

from oddsETL.helpers import (
    _american_to_probability,
    map_sport_to_event_ids,
    flatten_odds_h2h,
    flatten_scores,
    flatten_spread_odds,
    flatten_totals_odds,
    get_events,
    process_events,
    process_odds_h2h,
    process_odds_spread,
    process_odds_totals,
    process_scores,
    _save_processed_data,
    _save_raw_data,
    _devig_probs,
    _get_latest_json_from_dir,
    _parse_ts_from_file,
    _swap_name_with_id,
)

def test_map_sport_to_event_ids():
    events = [
        {"id": "e1", "sport_key": "sport_a"},
        {"id": "e2", "sport_key": "sport_b"},
        {"id": "e3", "sport_key": "sport_a"},
    ]

    mapping = map_sport_to_event_ids(events)

    assert mapping == {
        "sport_a": ["e1", "e3"],
        "sport_b": ["e2"],
    }

def test_swap_name_with_id_data(tmp_path: Path):
    participants_path = tmp_path / "participants.csv"
    participants = pd.DataFrame(
        [
            {"participant_id": "id1", "full_name": "Team A"},
            {"participant_id": "id2", "full_name": "Team B"},
            {"participant_id": "id3", "full_name": "Team C"},
        ]
    )
    participants.to_csv(participants_path, index=False)

    df = pd.DataFrame(
        [
            {"home_team": "Team A", "away_team": "Team B"},
            {"home_team": "Team C", "away_team": "Team A"},
        ]
    )

    out = _swap_name_with_id(df, participants_path=participants_path)

    assert out.loc[0, "home_team_id"] == "id1"
    assert out.loc[0, "away_team_id"] == "id2"
    assert out.loc[1, "home_team_id"] == "id3"
    assert out.loc[1, "away_team_id"] == "id1"
    assert "home_team" not in out.columns
    assert "away_team" not in out.columns

def test_devig_probs():

    #happy path test
    row = pd.Series({"home_prob": 0.6, "away_prob": 0.5})
    out = _devig_probs(row, ["home_prob", "away_prob"])
    total = out["home_prob"] + out["away_prob"]
    assert total == pytest.approx(1.0)
    assert out["home_prob"] == pytest.approx(0.6 / 1.1)
    assert out["away_prob"] == pytest.approx(0.5 / 1.1)

    # Test with empty draw probs
    row = pd.Series({"home_prob": 0.7, "away_prob": 0.4, "draw_prob": None})
    out = _devig_probs(row, ["home_prob", "away_prob", "draw_prob"])
    total = out["home_prob"] + out["away_prob"]
    assert total == pytest.approx(1.0)
    assert out["home_prob"] == pytest.approx(0.7 / 1.1)
    assert out["away_prob"] == pytest.approx(0.4 / 1.1)
    assert pd.isna(out["draw_prob"])

def test_american_to_probability():

    odds = 150
    prob = _american_to_probability(odds)
    assert prob == pytest.approx(0.4)

    odds = -200
    prob = _american_to_probability(odds)
    assert prob == pytest.approx(0.6666667)

    odds = None
    prob = _american_to_probability(odds)
    assert prob is None

def test_get_events():

    resp_ok = Mock()
    resp_ok.status_code = 200
    resp_ok.json.side_effect = [
        [{"id": "a1"}, {"id": "a2"}],
        [{"id": "b1"}],
    ]

    resp_fail = Mock()
    resp_fail.status_code = 500
    resp_fail.text = "Server Error"

    with patch("oddsETL.helpers.requests.get") as mock_get, patch(
        "oddsETL.helpers._save_raw_data"
    ) as mock_save:
        mock_get.side_effect = [resp_ok, resp_ok, resp_fail]
        mock_save.return_value = Path("raw_data/events/events_raw_000.json")

        sports = ["sport_a", "sport_b", "sport_c"]
        events = get_events(sports)

    assert len(events) == 3
    assert {e["id"] for e in events} == {"a1", "a2", "b1"}
    assert mock_get.call_count == 3
    mock_save.assert_called_once()

def test_save_raw_data(tmp_path: Path):
    data = [{"id": 1}, {"id": 2}]

    out = _save_raw_data(
        data=data,
        output_file="events_raw",
        entity_type="events",
        base_dir=tmp_path,
    )

    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded == data

def test_get_latest_json_from_dir(tmp_path: Path, entity_type: str = "events"):
    
    entity_dir = tmp_path / entity_type
    entity_dir.mkdir(parents=True)

    file1 = entity_dir / f"{entity_type}_raw_20240101_000000.json"
    file1.write_text('{"data": 1}', encoding="utf-8")

    file2 = entity_dir / f"{entity_type}_raw_20240102_000000.json"
    file2.write_text('{"data": 2}', encoding="utf-8")

    latest_data = _get_latest_json_from_dir(tmp_path, entity_type)
    assert latest_data == { "data": 2 }

def test_save_processed_data(tmp_path: Path):
    
    df = pd.DataFrame(
        [
            {"id": 1, "value": "A"},
            {"id": 2, "value": "B"},
        ]
    )

    out = _save_processed_data(
        df=df,
        output_file="events_processed",
        entity_type="events",
        base_dir=tmp_path,
        file_format="csv",
    )

    assert out.exists()
    loaded = pd.read_csv(out)
    pd.testing.assert_frame_equal(loaded, df)

def test_parse_ts(tmp_path: Path):
    
    file_path = tmp_path / "events_raw_20231231_235959.json"
    ts = _parse_ts_from_file(file_path)
    assert isinstance(ts, datetime)
    assert ts.year == 2023
    assert ts.month == 12
    assert ts.day == 31
    assert ts.hour == 23
    assert ts.minute == 59
    assert ts.second == 59


def test_flatten_odds_h2h():
    
    data = [
        {
            "id": "event1",
            "home_team": "Team A",
            "away_team": "Team B",
            "commence_time": "2024-01-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "bookie1",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Team A", "price": -150},
                                {"name": "Team B", "price": 130},
                            ],
                        }
                    ],
                },
                {
                    "key": "bookie2",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Team A", "price": -140},
                                {"name": "Team B", "price": 120},
                            ],
                        }
                    ],
                },
            ],
        }
    ]

    df = flatten_odds_h2h(data)

    # basic structure tests
    assert not df.empty
    assert "bookmaker" in df.columns
    assert "home_odds" in df.columns
    assert  "away_odds" in df.columns
    assert "market_key" in df.columns

    # test bookmaker extraction
    df.shape[0] == 2  # 2 bookmakers
    assert df['bookmaker'].iloc[0] == 'bookie1'
    assert df['bookmaker'].iloc[1] == 'bookie2'

    # test odds values extraction
    assert df['home_odds'].iloc[0] == -150
    assert df['away_odds'].iloc[0] == 130
    assert df['home_odds'].iloc[1] == -140
    assert df['away_odds'].iloc[1] == 120

    # test market key
    assert all(df['market_key'] == 'h2h')


def test_flatten_spread_odds():
    
    data = [
        {
            "id": "event1",
            "home_team": "Team A",
            "away_team": "Team B",
            "commence_time": "2024-01-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "bookie1",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Team A", "point": -3.5, "price": -110},
                                {"name": "Team B", "point": 3.5, "price": -110},
                            ],
                        }
                    ],
                },
                {
                    "key": "bookie2",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Team A", "point": -4.0, "price": -105},
                                {"name": "Team B", "point": 4.0, "price": -115},
                            ],
                        }
                    ],
                },
            ],
        }
    ]

    df = flatten_spread_odds(data)
    # basic structure tests
    assert not df.empty
    assert "bookmaker" in df.columns
    assert "home_spread" in df.columns
    assert "away_spread" in df.columns
    assert "market_key" in df.columns

    # test bookmaker extraction
    df.shape[0] == 2  # 2 bookmakers
    assert df['bookmaker'].iloc[0] == 'bookie1'
    assert df['bookmaker'].iloc[1] == 'bookie2'

    # test spread values extraction
    assert df['home_spread'].iloc[0] == -3.5
    assert df['away_spread'].iloc[0] == 3.5
    assert df['home_spread'].iloc[1] == -4.0
    assert df['away_spread'].iloc[1] == 4.0

    # test market key
    assert all(df['market_key'] == 'spread')


def test_flatten_totals_odds():
    
    data = [
        {
            "id": "event1",
            "home_team": "Team A",
            "away_team": "Team B",
            "commence_time": "2024-01-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "bookie1",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 45.5, "price": -110},
                                {"name": "Under", "point": 45.5, "price": -110},
                            ],
                        }
                    ],
                },
                {
                    "key": "bookie2",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 46.0, "price": -105},
                                {"name": "Under", "point": 46.0, "price": -115},
                            ],
                        }
                    ],
                },
            ],
        }
    ]

    df = flatten_totals_odds(data)
    
    # basic structure tests
    assert not df.empty
    assert "bookmaker" in df.columns
    assert "total_points" in df.columns
    assert "over_odds" in df.columns
    assert "under_odds" in df.columns
    assert "market_key" in df.columns

    # test bookmaker extraction
    df.shape[0] == 2  # 2 bookmakers
    assert df['bookmaker'].iloc[0] == 'bookie1'
    assert df['bookmaker'].iloc[1] == 'bookie2'

    # test totals values extraction
    assert df['total_points'].iloc[0] == 45.5
    assert df['over_odds'].iloc[0] == -110
    assert df['under_odds'].iloc[0] == -110
    assert df['total_points'].iloc[1] == 46.0
    assert df['over_odds'].iloc[1] == -105
    assert df['under_odds'].iloc[1] == -115

    # test market key
    assert all(df['market_key'] == 'total')


def test_flatten_scores():
    
    data = [
        {
            "id": "event1",
            "home_team": "Team A",
            "away_team": "Team B",
            "commence_time": "2024-01-01T20:00:00Z",
            "completed": False,
            "scores": [
                {"name": "Team A", "score": 24},
                {"name": "Team B","score": 17},
            ],
            "last_update": "2024-01-01T21:30:00Z",
        },
        {
            "id": "event2",
            "home_team": "Team C",
            "away_team": "Team D",
            "commence_time": "2024-01-02T18:00:00Z",
            "completed": False,
            "scores": None,
            "last_update": None,
        },
    ]

    df = flatten_scores(data)

    # basic structure tests
    assert not df.empty
    assert "home_score" in df.columns
    assert "away_score" in df.columns
    assert "completed" in df.columns
    assert "last_update" in df.columns
    assert df.shape[0] == 2
    
    # test score extraction
    assert df['home_score'].iloc[0] == 24
    assert df['away_score'].iloc[0] == 17
    assert pd.isna(df['home_score'].iloc[1])
    assert pd.isna(df['away_score'].iloc[1])

    # test completed and last_updated fields
    assert bool(df['completed'].iloc[0]) is False
    assert df['last_update'].iloc[0] == "2024-01-01T21:30:00Z"
    assert bool(df['completed'].iloc[1]) is False
    assert pd.isna(df['last_update'].iloc[1])

def test_process_events(monkeypatch):

    events = [
        {
            "id": "event1",
            "sport_key": "sport1",
            "sport_title": "Sport 1",
            "commence_time": "2024-01-01T20:00:00Z",
            "home_team": "Team A",
            "away_team": "Team B",
        },
        {
            "id": "event2",
            "sport_key": "sport2",
            "sport_title": "Sport 2",
            "commence_time": "2024-01-02T18:00:00Z",
            "home_team": "Team C",
            "away_team": "Team D",
        },
    ]
    def fake_swap(df, participants_path=None):
        df["home_team_id"] = "p1"
        df["away_team_id"] = "p2"
        return df.drop(columns=["home_team", "away_team"])

    monkeypatch.setattr("oddsETL.helpers._swap_name_with_id", fake_swap)

    df = process_events(events)

    # basic structure tests
    assert not df.empty
    assert "event_id" in df.columns
    assert "sport_key" in df.columns
    assert "commence_time" in df.columns
    assert "home_team_id" in df.columns
    assert "away_team_id" in df.columns
    assert df.shape[0] == 2

def test_process_scores(monkeypatch):
    
    scores = [
        {
            "id": "event1",
            "sport_key": "sport1",
            "sport_title": "Sport 1",
            "home_team": "Team A",
            "away_team": "Team B",
            "commence_time": "2024-01-01T20:00:00Z",
            "completed": True,
            "scores": [
                {"name": "Team A", "score": 30},
                {"name": "Team B", "score": 27},
            ],
            "last_update": "2024-01-01T22:00:00Z",
        },
        {
            "id": "event2",
            "sport_key": "sport2",
            "sport_title": "Sport 2",
            "home_team": "Team C",
            "away_team": "Team D",
            "commence_time": "2024-01-02T18:00:00Z",
            "completed": False,
            "scores": None,
            "last_update": None,
        },
    ]

    def fake_swap(df, participants_path=None):
        df["home_team_id"] = "p1"
        df["away_team_id"] = "p2"
        return df.drop(columns=["home_team", "away_team"])

    monkeypatch.setattr("oddsETL.helpers._swap_name_with_id", fake_swap)

    df = process_scores(scores)

    # basic structure tests
    assert not df.empty
    assert "event_id" in df.columns
    assert "sport_key" in df.columns
    assert "commence_time" in df.columns
    assert "home_team_id" in df.columns
    assert "away_team_id" in df.columns
    assert "home_score" in df.columns
    assert "away_score" in df.columns
    assert "completed" in df.columns
    assert "last_update" in df.columns
    assert df.shape[0] == 2

def test_process_odds_h2h(monkeypatch):
    
    odds = [
        {
            "id": "event1",
            "sport_key": "sport1",
            "home_team": "Team A",
            "away_team": "Team B",
            "commence_time": "2024-01-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "bookie1",
                    "title": "Bookie 1",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Team A", "price": -150},
                                {"name": "Team B", "price": 130},
                            ],
                        }
                    ],
                },
                {
                    "key": "bookie2",
                    "title": "Bookie 2",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Team A", "price": -140},
                                {"name": "Team B", "price": 120},
                            ],
                        }
                    ],
                },
            ],
        },
        {
            "id": "event2",
            "sport_key": "sport2",
            "home_team": "Team C",
            "away_team": "Team D",
            "commence_time": "2024-01-02T18:00:00Z",
            "bookmakers": [
                {
                    "key": "bookie1",
                    "title": "Bookie 1",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Team C", "price": -160},
                                {"name": "Team D", "price": 140},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    def fake_swap(df, participants_path=None):
        df["home_team_id"] = "p1"
        df["away_team_id"] = "p2"
        return df.drop(columns=["home_team", "away_team"])
    
    monkeypatch.setattr("oddsETL.helpers._swap_name_with_id", fake_swap)

    df = process_odds_h2h(odds)

    # basic structure tests
    assert not df.empty
    assert "event_id" in df.columns
    assert "sport_key" in df.columns
    assert "market_key" in df.columns
    assert "commence_time" in df.columns
    assert "home_team_id" in df.columns
    assert "away_team_id" in df.columns
    assert "home_prob" in df.columns
    assert "away_prob" in df.columns
    assert "draw_prob" in df.columns
    assert df.shape[0] == 2  # 1 aggregated row per event


def test_process_odds_spread(monkeypatch):
    
    spread = [
        {
            "id": "event1",
            "sport_key": "sport1",
            "home_team": "Team A",
            "away_team": "Team B",
            "commence_time": "2024-01-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "bookie1",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Team A", "point": -3.5, "price": -110},
                                {"name": "Team B", "point": 3.5, "price": -110},
                            ],
                        }
                    ],
                },
                {
                    "key": "bookie2",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Team A", "point": -4.0, "price": -105},
                                {"name": "Team B", "point": 4.0, "price": -115},
                            ],
                        }
                    ],
                },
            ],
        },
        {
            "id": "event2",
            "sport_key": "sport2",
            "home_team": "Team C",
            "away_team": "Team D",
            "commence_time": "2024-01-02T18:00:00Z",
            "bookmakers": [
                {
                    "key": "bookie1",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Team C", "point": -5.0, "price": -120},
                                {"name": "Team D", "point": 5.0, "price": 100},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    def fake_swap(df, participants_path=None):
        df["home_team_id"] = "p1"
        df["away_team_id"] = "p2"
        return df.drop(columns=["home_team", "away_team"])
    
    monkeypatch.setattr("oddsETL.helpers._swap_name_with_id", fake_swap)

    df = process_odds_spread(spread)

    # basic structure tests
    assert not df.empty
    assert "event_id" in df.columns
    assert "sport_key" in df.columns
    assert "market_key" in df.columns
    assert "commence_time" in df.columns
    assert "home_team_id" in df.columns
    assert "away_team_id" in df.columns
    assert "home_prob" in df.columns
    assert "home_spread" in df.columns
    assert "away_prob" in df.columns
    assert "away_spread" in df.columns
    assert df.shape[0] == 2  # 1 aggregated row per event

def test_process_odds_totals(monkeypatch):
    
    total = [
        {
            "id": "event1",
            "sport_key": "sport1",
            "home_team": "Team A",
            "away_team": "Team B",
            "commence_time": "2024-01-01T20:00:00Z",
            "bookmakers": [
                {
                    "key": "bookie1",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 45.5, "price": -110},
                                {"name": "Under", "point": 45.5, "price": -110},
                            ],
                        }
                    ],
                },
                {
                    "key": "bookie2",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 46.0, "price": -105},
                                {"name": "Under", "point": 46.0, "price": -115},
                            ],
                        }
                    ],
                },
            ],
        },
        {
            "id": "event2",
            "sport_key": "sport2",
            "home_team": "Team C",
            "away_team": "Team D",
            "commence_time": "2024-01-02T18:00:00Z",
            "bookmakers": [
                {
                    "key": "bookie1",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "point": 50.0, "price": -120},
                                {"name": "Under", "point": 50.0, "price": 100},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    def fake_swap(df, participants_path=None):
        df["home_team_id"] = "p1"
        df["away_team_id"] = "p2"
        return df.drop(columns=["home_team", "away_team"])
    monkeypatch.setattr("oddsETL.helpers._swap_name_with_id", fake_swap)
    
    df = process_odds_totals(total)

    # basic structure tests
    assert not df.empty
    assert "event_id" in df.columns
    assert "sport_key" in df.columns
    assert "market_key" in df.columns
    assert "commence_time" in df.columns
    assert "home_team_id" in df.columns
    assert "away_team_id" in df.columns
    assert "total_points" in df.columns
    assert "over_prob" in df.columns
    assert "under_prob" in df.columns
    assert df.shape[0] == 2  # 1 aggregated row per event
