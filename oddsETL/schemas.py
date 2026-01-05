EVENTS_SCHEMA = {
    "event_id": "text",
    "sport_key": "text",
    "home_team_id": "text",
    "away_team_id": "text",
    "commence_time": "timestamp",
}

SCORES_SCHEMA = {
    "event_id": "text",
    "sport_key": "text",
    "commence_time": "timestamp",
    "home_team_id": "text",
    "away_team_id": "text",
    "completed": "boolean",
    "home_score": "integer",
    "away_score": "integer",
    "last_update": "timestamp",
}

H2H_ODDS_SCHEMA = {
    "event_id": "text",
    "sport_key": "text",
    "commence_time": "timestamp",
    "market_key": "text",
    "home_team_id": "text",
    "away_team_id": "text",
    "home_prob": "float",
    "away_prob": "float",
    "draw_prob": "float",
}

SPREAD_ODDS_SCHEMA = {
    "event_id": "text",
    "sport_key": "text",
    "commence_time": "timestamp",
    "market_key": "text",
    "home_team_id": "text",
    "away_team_id": "text",
    "home_spread": "float",
    "away_spread": "float",
    "home_prob": "float",
    "away_prob": "float",
}

TOTALS_ODDS_SCHEMA = {
    "event_id": "text",
    "sport_key": "text",
    "commence_time": "timestamp",
    "market_key": "text",
    "home_team_id": "text",
    "away_team_id": "text",
    "total_points": "float",
    "over_prob": "float",
    "under_prob": "float",
}

