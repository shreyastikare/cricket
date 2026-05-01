from __future__ import annotations


TEAM_LOGO_PATHS: dict[str, str] = {
    "Chennai Super Kings": "logos/CSKoutline.png",
    "CSK": "logos/CSKoutline.png",
    "Delhi Capitals": "logos/DCoutline.png",
    "Delhi Daredevils": "logos/DDoutline.png",
    "DD": "logos/DDoutline.png",
    "Deccan Chargers": "logos/DECoutline.png",
    "DEC": "logos/DECoutline.png",
    "Gujarat Titans": "logos/GToutline.png",
    "GT": "logos/GToutline.png",
    "Gujarat Lions": "logos/GLoutline.png",
    "GL": "logos/GLoutline.png",
    "Kochi Tuskers Kerala": "logos/KTKoutline.png",
    "KTK": "logos/KTKoutline.png",
    "Kolkata Knight Riders": "logos/KKRoutline.png",
    "KKR": "logos/KKRoutline.png",
    "Lucknow Super Giants": "logos/LSGoutline.png",
    "LSG": "logos/LSGoutline.png",
    "Mumbai Indians": "logos/MIoutline.png",
    "MI": "logos/MIoutline.png",
    "Punjab Kings": "logos/PBKSoutline.png",
    "PBKS": "logos/PBKSoutline.png",
    "Pune Warriors India": "logos/PWIoutline.png",
    "Pune Warriors": "logos/PWIoutline.png",
    "PWI": "logos/PWIoutline.png",
    "Rajasthan Royals": "logos/RR_Logo.png",
    "RR": "logos/RR_Logo.png",
    "Rising Pune Supergiant": "logos/RPSoutline.png",
    "Rising Pune Supergiants": "logos/RPSoutline.png",
    "RPS": "logos/RPSoutline.png",
    "Royal Challengers Bengaluru": "logos/RCBoutline.png",
    "Royal Challengers Bangalore": "logos/RCBoutline.png",
    "Royal Challengers": "logos/RCBoutline.png",
    "RCB": "logos/RCBoutline.png",
    "Sunrisers Hyderabad": "logos/SRHoutline.png",
    "SunRisers Hyderabad": "logos/SRHoutline.png",
    "SRH": "logos/SRHoutline.png",
}


_TEAM_LOGO_PATHS_NORMALIZED = {
    team_name.strip().lower(): path
    for team_name, path in TEAM_LOGO_PATHS.items()
}


def team_logo_path(team_name: object) -> str | None:
    key = "" if team_name is None else str(team_name).strip().lower()
    return _TEAM_LOGO_PATHS_NORMALIZED.get(key)
