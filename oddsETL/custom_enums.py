from enum import Enum


class SportKey(Enum):
    """supported leagues (for now)"""

    NFL = "americanfootball_nfl"
    NCAAF = "americanfootball_ncaaf"
    NBA = "basketball_nba"
    NCAAB = "basketball_ncaab"
    EPL = "soccer_epl"
    LA_LIGA = "soccer_spain_la_liga"
    LIGUE_1 = "soccer_france_ligue_one"
    BUNDESLIGA = "soccer_germany_bundesliga"
    SERIE_A = "soccer_italy_serie_a"
    
    @classmethod
    def members(cls) -> list["SportKey"]:
        return list(cls)


class Market(Enum):
    """supported markets (for now)"""

    H2H = "h2h"
    SPREADS = "spreads"
    TOTALS = "totals"


class OddsFormat(Enum):
    """odds formats offered by odds-api"""

    AMERICAN = "american"
    DECIMAL = "decimal"


class Region(Enum):
    """uk | us | eu | au. are the regions we are interested in (for now)"""

    US = "us"
    UK = "uk"
    EU = "eu"
    AU = "au"


class DateFormat(Enum):
    """supported date formats from odds-api"""

    ISO = "iso"
    UNIX = "unix"
