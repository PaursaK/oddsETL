from datetime import datetime, timezone, timedelta
from typing import Any
from supabase import create_client, Client

class OddsDB:
    def __init__(self, db_url: str, db_key: str):
        self.db_url = db_url
        self.db_key = db_key
        self.client: Client = create_client(self.db_url, self.db_key)

    def upsert(
            self, 
            table: str, 
            data: list[dict[str, Any]],
            conflict_target: list[str],
            ignore_duplicates: bool = False,
            returning: str = "minimal"
            ):
        
        """ Upsert data into the specified table.
        Args:
            table: The name of the table to upsert data into.
            data: The data to be upserted.
            conflict_target: The columns to check for conflicts.
            ignore_duplicates: If True, ignore duplicates. Defaults to False.
            returning: The return type. Defaults to "minimal". 
        """

        query = self.client.table(table).upsert(
            data,
            on_conflict=",".join(conflict_target),
            ignore_duplicates=ignore_duplicates,
            returning=returning
        )

        resp = query.execute()
        return resp
    
    def get_event_ids_by_sport(
        self,
        sport_key: str,
        *,
        days_back: int | None = 3,
    ) -> list[str]:
        """Fetch event IDs for a given sport key from recent, incomplete scores."""
        query = self.client.table("scores").select("event_id").eq("sport_key", sport_key)
        if days_back is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
            query = query.gte("commence_time", cutoff.isoformat())
        query = query.or_("completed.is.null,completed.eq.false")
        response = query.execute()
        records = response.data
        event_ids = [record["event_id"] for record in records]
        return event_ids
