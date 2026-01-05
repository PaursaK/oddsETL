from oddsETL.custom_enums import SportKey
from oddsETL.db import OddsDB
import os
import pandas as pd
from dotenv import load_dotenv
import requests
from oddsETL.helpers import (
    map_sport_to_event_ids,
    get_odds_h2h,
    get_odds_spread,
    get_odds_totals,
    get_scores,
    process_odds_spread,
    process_odds_totals,
    process_odds_h2h,
    process_scores,
)
from typing import Optional, Any

load_dotenv()


def upsert_events_from_scores(supabase_db: OddsDB, scores_df: pd.DataFrame) -> None:
    events_df = scores_df[
        ["event_id", "sport_key", "commence_time", "home_team_id", "away_team_id"]
    ]
    events_records = events_df.to_dict(orient="records")
    if events_records:
        supabase_db.upsert(
            table="events",
            data=events_records,
            conflict_target=["event_id"],
            ignore_duplicates=False,
            returning="minimal",
        )

def run_current(supabase_db: OddsDB, sports: list[SportKey]) -> None:
    for sport in sports:
        scores_data = get_scores([sport], params={"daysFrom": 3})
        scores_df = process_scores(scores_data)
        upsert_events_from_scores(supabase_db, scores_df)
        scores_records = scores_df.to_dict(orient="records")
        if scores_records:
            supabase_db.upsert(
                table="scores",
                data=scores_records,
                conflict_target=["event_id"],
                ignore_duplicates=False,
                returning="minimal",
            )

        # Fetch and process odds for existing events
        for odds_fetcher, odds_processor, table_name, conflict_cols in [
            (
                get_odds_h2h,
                process_odds_h2h,
                "head2head",
                ["event_id", "market_key", "home_team_id", "away_team_id"],
            ),
            (
                get_odds_spread,
                process_odds_spread,
                "spreads",
                ["event_id", "market_key", "home_team_id", "away_team_id"],
            ),
            (
                get_odds_totals,
                process_odds_totals,
                "totals",
                ["event_id", "market_key", "home_team_id", "away_team_id"],
            ),
        ]:
            event_ids_map = map_sport_to_event_ids(scores_data)
            event_ids = event_ids_map.get(sport.value, [])
            if not event_ids:
                continue
            odds_data = odds_fetcher([sport], event_ids_map={sport.value: event_ids})
            odds_df = odds_processor(odds_data)
            odds_records = odds_df.to_dict(orient="records")
            if odds_records:
                supabase_db.upsert(
                    table=table_name,
                    data=odds_records,
                    conflict_target=conflict_cols,
                    ignore_duplicates=False,
                    returning="minimal",
                )

    admin_token = os.getenv("ADMIN_TOKEN")
    if admin_token:
        resp = requests.post(
            "https://pocketbets.onrender.com/api/resolve-transactions",
            headers={"Authorization": f"Bearer {admin_token}"},
            timeout=30,
        )
        print("Resolve transactions response:", resp.status_code, resp.text)


def run() -> None:
    sports = [
        SportKey.NFL,
        SportKey.EPL,
        SportKey.NBA,
        SportKey.NCAAB,
        SportKey.BUNDESLIGA,
        SportKey.NCAAF,
        SportKey.LA_LIGA,
        SportKey.SERIE_A,
        SportKey.LIGUE_1,
    ]

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    supabase_db = OddsDB(db_url=url, db_key=key)

    run_current(supabase_db, sports)

if __name__ == "__main__":
    run()
