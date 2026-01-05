from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable
import os
import pandas as pd
import requests
import math

from .custom_enums import DateFormat, Market, OddsFormat, Region, SportKey

Path("raw_data").mkdir(exist_ok=True)
Path("processed_data").mkdir(exist_ok=True)

def _get_api_config() -> tuple[str, str]:
    api_key = os.getenv("ODDS_API_KEY")
    base_url = os.getenv("ODDS_BASE_URL")
    if not base_url:
        raise ValueError("ODDS_BASE_URL is not set. Check your .env or shell env.")
    if not api_key:
        raise ValueError("ODDS_API_KEY is not set. Check your .env or shell env.")
    return api_key, base_url

def map_sport_to_event_ids(events: list[dict[str, Any]]) -> dict[str, list[str]]:
    """
    Create a mapping from sport_key to list of event_ids.
    Args:
        events: list of event dictionaries from the API
    Returns:
        Dictionary mapping sport_key to list of event_ids
    """
    sport_event_map: dict[str, list[str]] = {}
    for event in events:
        sport_key = event.get("sport_key")
        event_id = event.get("id")
        if sport_key and event_id:
            if sport_key not in sport_event_map:
                sport_event_map[sport_key] = []
            sport_event_map[sport_key].append(event_id)
    return sport_event_map

def get_entity(
    sports_keys: Iterable[SportKey | str],
    *,
    endpoint: str = "",
    entity_type: str,
    params: dict[str, Any] | None = None,
    event_ids_map: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Generic fetcher for Odds API entities (events, scores, odds, etc.)

    Args:
        sports_keys: iterable of SportKey enums or raw sport_key strings
        endpoint: specific endpoint to hit (overrides entity_type if provided)
        entity_type: API entity name ('events', 'scores', etc.)
        params: extra query params (merged with api_key)

    Returns:
        Aggregated list of entity dictionaries
    """

    BATCH_SIZE = 100

    api_key, base_url = _get_api_config()
    query_params = {"api_key": api_key}
    if params:
        query_params.update(params)

    results: list[dict[str, Any]] = []
    failed = 0

    for key in sports_keys:
        sport_key = key.value if isinstance(key, Enum) else key
        url = f"{base_url}/{sport_key}/{endpoint or entity_type}"

        # event IDs filtering
        if event_ids_map and sport_key in event_ids_map:
            event_ids = event_ids_map.get(sport_key) or []
            if not event_ids:
                print(
                    f"No event ids for {entity_type} in sport: {sport_key}; "
                    "skipping eventIds filter."
                )
            else:
                for i in range(0, len(event_ids), BATCH_SIZE):
                    request_params = {
                        **query_params,
                        "eventIds": ",".join(event_ids[i:i + BATCH_SIZE]),
                    }
                    resp = requests.get(url, params=request_params)
                    if resp.status_code != 200:
                        print(
                            f"Failed to fetch {entity_type} for {sport_key} - "
                            f"status code: {resp.status_code}, response body: {resp.text}"
                        )
                        failed += 1
                        continue
                    data = resp.json()
                    if data:
                        results.extend(data)
                continue

        resp = requests.get(url, params=query_params)
        if resp.status_code != 200:
            print(
                f"Failed to fetch {entity_type} for {sport_key} - "
                f"status code: {resp.status_code}, response body: {resp.text}"
            )
            failed += 1
            continue
        data = resp.json()
        if data:
            results.extend(data)

    path = _save_raw_data(results, f"{entity_type}_raw", entity_type)
    print(
        f"Saved {len(results)} {entity_type} from "
        f"{list(sports_keys)} to {path}"
    )

    return results



def get_events(sports_keys: Iterable[SportKey | str]) -> list[dict[str, Any]]:
    return get_entity(
        sports_keys,
        entity_type="events",
    )

def process_events(events: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    
    # if there are no events grab latest file for processing
    if not events:
        events = _get_latest_json_from_dir(
            directory=Path("raw_data"),
            entity_type="events",
        )
    
    # load events into dataframe
    df = pd.DataFrame(events)

    # Keep only the core event columns if present
    keep_cols = ["id", "sport_key", "commence_time", "home_team", "away_team"]
    if df.empty and not df.columns.tolist():
        df = pd.DataFrame(columns=keep_cols)
    else:
        df = df[[c for c in keep_cols if c in df.columns]]
    df = df.rename(columns={"id": "event_id"})

    #read participants.csv from raw_data/ and turn into pandas dataframe
    df = _swap_name_with_id(df)

    # save a csv file processed_data at the respective directory (events)
    _save_processed_data(
        df=df,
        entity_type="events",
        output_file="events_processed",
    )

    return df

def get_scores(
        sports_keys: Iterable[SportKey | str], 
        event_ids_map: dict[str, list[str]] | None = None,
        params: dict[str, Any] | None = None
        ) -> list[dict[str, Any]]:
    return get_entity(
        sports_keys,
        entity_type="scores",
        params=params,
        event_ids_map = event_ids_map,
    )

def flatten_scores(data: list[dict[str, Any]]):
    """
    Flatten the nested scores structure into a tabular format.
    Args:
        data: list of score dictionaries from the API
    Returns:
        DataFrame with flattened scores"""

    row = []
    
    for item in data:

        # extract core event fields
        event_id = item.get("id")
        sport_key = item.get("sport_key")
        commence_time = item.get('commence_time')
        home_team = item.get("home_team", None)
        away_team = item.get("away_team", None)
        completed = item.get("completed", False)
        last_update = item.get("last_update", None)
        home_score = None
        away_score = None

        # flatten scores by matching team names
        for score in (item.get("scores") or []): 
            name = score.get("name", "")
            score = score.get("score", None)
            if home_team == name:
                home_score = score
            elif away_team == name:
                away_score = score

        row.append({
            "id": event_id,
            "sport_key": sport_key,
            "commence_time": commence_time,
            "home_team": home_team,
            "away_team": away_team,
            "completed": completed,
            "home_score": home_score,
            "away_score": away_score,
            "last_update": last_update
        })
            
    return pd.DataFrame(row)


def process_scores(scores: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    """
    Process raw scores data into a flattened DataFrame
    with participant IDs instead of team names.
    Args:
        scores: Optional list of score dictionaries. If None,
                the latest raw scores file will be used.
    Returns:
        None
    """

    # if there are no events grab latest file for processing
    if not scores:
        scores = _get_latest_json_from_dir(
            directory=Path("raw_data"),
            entity_type="scores",
        )

    # transform scores to flattened dataframe
    df = flatten_scores(scores)

    # Keep only the core event columns if present
    keep_cols = [
        "id", 
        "sport_key", 
        "commence_time", 
        "home_team", 
        "away_team", 
        "home_score", 
        "away_score", 
        "completed", 
        "last_update"
        ]
    if df.empty and not df.columns.tolist():
        df = pd.DataFrame(columns=keep_cols)
    else:
        df = df[[c for c in keep_cols if c in df.columns]]
    df = df.rename(columns={"id": "event_id"})

    #read participants.csv from raw_data/ and turn into pandas dataframe
    df = _swap_name_with_id(df)

    # save a csv file processed_data at the respective directory (scores)
    _save_processed_data(
        df=df,
        entity_type="scores",
        output_file="scores_processed",
    )

    return df


def get_odds_h2h(
        sports_keys: Iterable[SportKey | str],
        event_ids_map: dict[str, list[str]],
        region: Region = Region.US, 
        odds_format: OddsFormat = OddsFormat.AMERICAN,
        ) -> list[dict[str, Any]]:
    return get_entity(
        sports_keys,
        endpoint="odds",
        entity_type="h2h_odds",
        params={
            "markets": Market.H2H.value,
            "regions": region.value,
            "oddsFormat": odds_format.value,
            "dateFormat": DateFormat.ISO.value,
        },
        event_ids_map = event_ids_map,
    )

def flatten_odds_h2h(data: list[dict[str, Any]]):
    """
    Flatten the nested odds structure into a tabular format.
    Args:
        data: list of odds dictionaries from the API
    Returns:
        DataFrame with flattened odds
    """
    rows = []
    
    for item in data:

        # extract core event fields
        event_id = item.get("id")
        sport_key = item.get("sport_key")
        commence_time = item.get('commence_time')
        home_team = item.get("home_team", None)
        away_team = item.get("away_team", None)

        # flatten bookmakers, a new row per bookmaker
        for bookie in (item.get("bookmakers") or []): 
            bookmaker = bookie.get("key", None)     
            home_odds = None
            away_odds = None
            draw_odds = None
            for market in (bookie.get("markets") or []):
                if market.get("key") == "h2h":
                    for outcome in (market.get("outcomes") or []):
                        name = outcome.get("name", "")
                        price = outcome.get("price", None)
                        if home_team == name:
                            home_odds = price
                        elif away_team == name:
                            away_odds = price
                        else:
                            draw_odds = price

            rows.append({
                "id": event_id,
                "sport_key": sport_key,
                "market_key": "h2h",
                "commence_time": commence_time,
                "home_team": home_team,
                "away_team": away_team,
                "bookmaker": bookmaker,
                "home_odds": home_odds,
                "away_odds": away_odds,
                "draw_odds": draw_odds
            })
            
    return pd.DataFrame(rows)

def process_odds_h2h(odds: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    # if there are no events grab latest file for processing
    if not odds:
        odds = _get_latest_json_from_dir(
            directory=Path("raw_data"),
            entity_type="h2h_odds",
        )
    
    # flatten odds structure and assign american odds to home and away teams
    df = flatten_odds_h2h(odds)
    if df.empty or not {"home_odds", "away_odds", "draw_odds"}.issubset(df.columns):
        return pd.DataFrame(
            columns=[
                "event_id",
                "sport_key",
                "commence_time",
                "market_key",
                "home_team_id",
                "away_team_id",
                "home_prob",
                "away_prob",
                "draw_prob",
            ]
        )
    
    # turn american odds to probabilities and devig
    df["home_prob"] = df["home_odds"].apply(_american_to_probability)
    df["away_prob"] = df["away_odds"].apply(_american_to_probability)
    df["draw_prob"] = df["draw_odds"].apply(_american_to_probability)
    df = df.drop(columns=["home_odds", "away_odds", "draw_odds"])
    df = df.apply(_devig_probs, axis=1, columns=["home_prob", "away_prob", "draw_prob"])
    df['draw_prob'] = df['draw_prob'].fillna(0.0) # fill missing draw probabilities with 0.0


    # build consensus odds by averaging probabilities from different bookmakers
    df = df.groupby("id", as_index=False).agg({
        "sport_key": "first",
        "market_key": "first",
        "commence_time": "first",
        "home_team": "first",
        "away_team": "first",
        "home_prob": "mean",
        "away_prob": "mean",
        "draw_prob": "mean"
    })
    df.rename(columns={"id": "event_id"}, inplace=True)

    #read participants.csv from raw_data/ and turn into pandas dataframe
    df = _swap_name_with_id(df)

    # save to parquet file processed_data at the respective directory (odds_h2h)
    _save_processed_data(
        df=df,
        entity_type="odds_h2h",
        output_file="odds_h2h_processed",
    )

    return df


def get_odds_spread(
        sports_keys: Iterable[SportKey | str],
        event_ids_map: dict[str, list[str]],
        region: Region = Region.US, 
        odds_format: OddsFormat = OddsFormat.AMERICAN
        ) -> list[dict[str, Any]]:
    return get_entity(
        sports_keys,
        endpoint="odds",
        entity_type="spread_odds",
        params={
            "markets": Market.SPREADS.value,
            "regions": region.value,
            "oddsFormat": odds_format.value,
            "dateFormat": DateFormat.ISO.value,
        },
        event_ids_map = event_ids_map,
    )

def flatten_spread_odds(data: list[dict[str, Any]]):
    """
    Flatten the nested spread odds structure into a tabular format.
    Args:
        data: list of odds dictionaries from the API
    Returns:
        DataFrame with flattened spread odds
    """
    rows = []
    
    for item in data:

        # extract core event fields
        event_id = item.get("id")
        sport_key = item.get("sport_key")
        commence_time = item.get('commence_time')
        home_team = item.get("home_team", None)
        away_team = item.get("away_team", None)

        # flatten bookmakers, a new row per bookmaker
        for bookie in (item.get("bookmakers") or []): 
            bookmaker = bookie.get("key", None)     
            home_spread = None
            away_spread = None
            home_odds = None
            away_odds = None
            for market in (bookie.get("markets") or []):
                if market.get("key") == "spreads":
                    for outcome in (market.get("outcomes") or []):
                        name = outcome.get("name", "")
                        price = outcome.get("price", None)
                        point = outcome.get("point", None)
                        if home_team == name:
                            home_spread = point
                            home_odds = price
                        elif away_team == name:
                            away_spread = point
                            away_odds = price

            rows.append({
                "id": event_id,
                "sport_key": sport_key,
                "market_key": "spreads",
                "commence_time": commence_time,
                "home_team": home_team,
                "away_team": away_team,
                "bookmaker": bookmaker,
                "home_spread": home_spread,
                "home_odds": home_odds,
                "away_spread": away_spread,
                "away_odds": away_odds
            })
            
    return pd.DataFrame(rows)

def process_odds_spread(odds: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    """
    Process raw spread odds data into a flattened DataFrame
    with participant IDs instead of team names.
    Args:
        odds: Optional list of spread odds dictionaries. If None,
                the latest raw spread odds file will be used.
    Returns:
        None"""
    if not odds:
        odds = _get_latest_json_from_dir(
            directory=Path("raw_data"),
            entity_type="spread_odds",
        )


    # flatten odds structure and assign american odds to home and away teams with handicaps
    df = flatten_spread_odds(odds)
    if df.empty or not {"home_odds", "away_odds", "home_spread"}.issubset(df.columns):
        return pd.DataFrame(
            columns=[
                "event_id",
                "sport_key",
                "commence_time",
                "market_key",
                "home_team_id",
                "away_team_id",
                "home_spread",
                "away_spread",
                "home_prob",
                "away_prob",
            ]
        )
    
    # turn american odds to probabilities and devig
    df["home_prob"] = df["home_odds"].apply(_american_to_probability)
    df["away_prob"] = df["away_odds"].apply(_american_to_probability)
    df = df.drop(columns=["home_odds", "away_odds"])
    df = df.apply(_devig_probs, axis=1, columns=["home_prob", "away_prob"])

    # build consensus odds by averaging probabilities from different bookmakers
    df = df.groupby("id", as_index=False).agg({
        "sport_key": "first",
        "market_key": "first",
        "commence_time": "first",
        "home_team": "first",
        "away_team": "first",
        "home_prob": "mean",
        "away_prob": "mean",
        "home_spread": "mean",
    })
    df.rename(columns={"id": "event_id"}, inplace=True)
    
    #adjust home spread to avoid whole numbers
    def push_to_half_avoid_whole(x: float) -> float:
        if pd.isna(x):
            return x
        y = math.trunc(x * 2) / 2   # toward zero
        if y.is_integer():
            y = y - 0.5 if x < 0 else y + 0.5
        return y
    
    # set away spread to negative of home spread
    df['home_spread'] = df['home_spread'].apply(push_to_half_avoid_whole)
    df['away_spread'] = -df['home_spread']
    
    #read participants.csv from raw_data/ and turn into pandas dataframe
    df = _swap_name_with_id(df)

    # save to csv file processed_data at the respective directory (odds_spread)
    _save_processed_data(
        df=df,
        entity_type="odds_spread",
        output_file="odds_spread_processed",
    )

    return df


def get_odds_totals(
        sports_keys: Iterable[SportKey | str],
        event_ids_map: dict[str, list[str]],
        region: Region = Region.US, 
        odds_format: OddsFormat = OddsFormat.AMERICAN
        ) -> list[dict[str, Any]]:
    return get_entity(
        sports_keys,
        endpoint="odds",
        entity_type="totals_odds",
        params={
            "markets": Market.TOTALS.value,
            "regions": region.value,
            "oddsFormat": odds_format.value,
            "dateFormat": DateFormat.ISO.value,
        },
        event_ids_map = event_ids_map,
    )

def flatten_totals_odds(data: list[dict[str, Any]]):
    """
    Flatten the nested totals odds structure into a tabular format.
    Args:
        data: list of odds dictionaries from the API
    Returns:
        DataFrame with flattened totals odds
    """
    rows = []
    
    for item in data:

        # extract core event fields
        event_id = item.get("id")
        sport_key = item.get("sport_key")
        commence_time = item.get('commence_time')
        home_team = item.get("home_team", None)
        away_team = item.get("away_team", None)

        # flatten bookmakers, a new row per bookmaker
        for bookie in (item.get("bookmakers") or []): 
            bookmaker = bookie.get("key", None)     
            total_points = None
            over_odds = None
            under_odds = None
            for market in (bookie.get("markets") or []):
                if market.get("key") == "totals":
                    for outcome in (market.get("outcomes") or []):
                        name = outcome.get("name", "")
                        price = outcome.get("price", None)
                        point = outcome.get("point", None)
                        if name == "Over":
                            total_points = point
                            over_odds = price
                        elif name == "Under":
                            under_odds = price

            rows.append({
                "id": event_id,
                "sport_key": sport_key,
                "market_key": "totals",
                "commence_time": commence_time,
                "home_team": home_team,
                "away_team": away_team,
                "bookmaker": bookmaker,
                "total_points": total_points,
                "over_odds": over_odds,
                "under_odds": under_odds
            })
            
    return pd.DataFrame(rows)

def process_odds_totals(odds: list[dict[str, Any]] | None = None) -> pd.DataFrame:

    if not odds:
        odds = _get_latest_json_from_dir(
            directory=Path("raw_data"),
            entity_type="totals_odds",
        )

    # flatten odds structure and assign american odds to home and away teams with handicaps
    df = flatten_totals_odds(odds)
    if df.empty or not {"over_odds", "under_odds", "total_points"}.issubset(df.columns):
        return pd.DataFrame(
            columns=[
                "event_id",
                "sport_key",
                "commence_time",
                "market_key",
                "home_team_id",
                "away_team_id",
                "total_points",
                "over_prob",
                "under_prob",
            ]
        )

    # turn american odds to probabilities
    df["over_prob"] = df["over_odds"].apply(_american_to_probability)
    df["under_prob"] = df["under_odds"].apply(_american_to_probability)
    df = df.drop(columns=["over_odds", "under_odds"])
    df = df.apply(_devig_probs, axis=1, columns=["over_prob", "under_prob"])

    # build consensus odds by averaging probabilities from different bookmakers
    df = df.groupby("id", as_index=False).agg({
        "sport_key": "first",
        "market_key": "first",
        "commence_time": "first",
        "home_team": "first",
        "away_team": "first",
        "over_prob": "mean",
        "under_prob": "mean",
        "total_points": "mean",
    })
    df.rename(columns={"id": "event_id"}, inplace=True)

    #adjust total points to avoid whole numbers
    def push_to_half_avoid_whole(x: float) -> float:
        if pd.isna(x):
            return x
        y = math.trunc(x * 2) / 2   # toward zero
        if y.is_integer():
            y = y + 0.5
        return y
    df['total_points'] = df['total_points'].apply(push_to_half_avoid_whole)

    #read participants.csv from raw_data/ and turn into pandas dataframe
    df = _swap_name_with_id(df)

    # save to csv file processed_data at the respective directory (odds_totals)
    _save_processed_data(
        df=df,
        entity_type="odds_totals",
        output_file="odds_totals_processed",
    )

    return df

# PRIVATE METHODS

def _parse_ts_from_file(path: Path) -> datetime:
    # split on "_" and take last two pieces: YYYYMMDD and HHMMSS
    stem = path.stem  # events_raw_20251222_153012
    ymd, hms = stem.split("_")[-2:]
    return datetime.strptime(f"{ymd}_{hms}", "%Y%m%d_%H%M%S")

def _save_raw_data(
    data: list[dict[str, Any]],
    output_file: str,
    entity_type: str,
    base_dir: str = "raw_data",
) -> Path:
    """
    Save raw API JSON payloads to raw_data/{entity_type}/ with timestamped filename.
    Args:
        data: list of dictionaries returned by API
        output_file: base filename (e.g. "events", "scores", "odds_h2h")
        entity_type: subdirectory name under raw_data/
        base_dir: root raw directory (default: raw_data)
    Returns:
        Path to the written file
    """

    # create directory raw_data/{entity_type}
    out_dir = Path(base_dir) / entity_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # timestamp (UTC recommended for pipelines)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # full filename
    out_path = out_dir / f"{output_file}_{ts}.json"

    # write JSON
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return out_path


def _save_processed_data(
    df: pd.DataFrame,
    entity_type: str,
    output_file: str,
    base_dir: str = "processed_data",
    file_format: str = "json",
) -> Path:
    """
    Save processed DataFrame to processed_data/{entity_type}/
    with a timestamped filename.

    Args:
        df: pandas DataFrame to persist
        entity_type: subdirectory name (e.g. 'events', 'scores', 'odds')
        output_file: base filename (e.g. 'events')
        base_dir: root processed directory
        file_format: 'parquet' or 'csv' or 'json'

    Returns:
        Path to written file
    """

    out_dir = Path(base_dir) / entity_type
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if file_format == "parquet":
        out_path = out_dir / f"{output_file}_{ts}.parquet"
        df.to_parquet(out_path, index=False)

    elif file_format == "csv":
        out_path = out_dir / f"{output_file}_{ts}.csv"
        df.to_csv(out_path, index=False)

    elif file_format == "json":
        out_path = out_dir / f"{output_file}_{ts}.json"
        df.to_json(out_path, orient="records")

    else:
        raise ValueError(f"Unsupported file_format: {file_format}")

    return out_path


def _swap_name_with_id(
    df: pd.DataFrame,
    participants_path: Path = Path("raw_data") / "participants.csv"
) -> pd.DataFrame:
    """
    Replace team names in home_team and away_team columns with participant_ids.
    Args:
        df: DataFrame containing home_team and away_team columns
        participants_path: Path to participants CSV file
    Returns:
        DataFrame with home_team and away_team replaced by IDs
    """

    participants = pd.read_csv(participants_path)
    participants = participants.dropna(subset=["participant_id", "full_name"])
    name_to_id = dict(zip(participants["full_name"], participants["participant_id"]))

    df["home_team_id"] = df["home_team"].map(name_to_id)
    df["away_team_id"] = df["away_team"].map(name_to_id)
    df = df.drop(columns=["home_team", "away_team"])

    return df

def _get_latest_json_from_dir(
    directory: Path,
    entity_type: str,
) -> list[dict[str, Any]]:
    
    raw_odds_dir = directory / entity_type
    odds_files = list(raw_odds_dir.glob("*.json"))
    if not odds_files:
        raise FileNotFoundError(
            f"No raw odds files found in {raw_odds_dir}. Run get_{entity_type} first."
        )
    latest_file = max(odds_files, key=_parse_ts_from_file)
    odds = json.loads(latest_file.read_text(encoding="utf-8"))
    return odds

def _devig_probs(row, columns: list[str]):
    probs = row[columns]
    total_prob = probs.sum(skipna=True)
    if total_prob <= 0 or pd.isna(total_prob):
        return row
    for col in probs.index:
        if pd.notna(row[col]):
            row[col] = row[col] / total_prob
    return row

def _american_to_probability(odds: float) -> float | None:
    """
    Convert American odds to implied probability.
    Args:
        odds: American odds value
    Returns:
        Implied probability as a float between 0 and 1
    """
    if odds is None:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)
