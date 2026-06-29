# Snippet 1 — ELO Ratings (pipeline.py)
def compute_elo(df, k=20, default_elo=1500.0):
    elo = {}
    home_elo_l, away_elo_l, elo_diff_l = [], [], []

    for _, row in df.iterrows():
        h = row["home_team_id"]
        a = row["away_team_id"]
        ra = elo.get(h, default_elo)
        rb = elo.get(a, default_elo)

        home_elo_l.append(ra)
        away_elo_l.append(rb)
        elo_diff_l.append(ra - rb)

        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        sa = 1.0 if row["result"] == 1 else (0.5 if row["result"] == 2 else 0.0)

        elo[h] = ra + k * (sa - ea)
        elo[a] = rb + k * ((1.0 - sa) - (1.0 - ea))

    return home_elo_l, away_elo_l, elo_diff_l, elo


# Snippet 2 — Venue-Split Form (pipeline.py)
def compute_venue_form(df, n=5):
    venue_pts = {}
    hph_l, hpa_l, aph_l, apa_l = [], [], [], []

    for _, row in df.iterrows():
        htid = row["home_team_id"]
        atid = row["away_team_id"]
        res  = row["result"]

        hph_l.append(np.mean(venue_pts.get((htid, 'H'), [])[-n:]) if venue_pts.get((htid, 'H')) else 1.0)
        hpa_l.append(np.mean(venue_pts.get((htid, 'A'), [])[-n:]) if venue_pts.get((htid, 'A')) else 1.0)
        aph_l.append(np.mean(venue_pts.get((atid, 'H'), [])[-n:]) if venue_pts.get((atid, 'H')) else 1.0)
        apa_l.append(np.mean(venue_pts.get((atid, 'A'), [])[-n:]) if venue_pts.get((atid, 'A')) else 1.0)

        h_pts = 3 if res == 1 else (1 if res == 2 else 0)
        a_pts = 3 if res == 0 else (1 if res == 2 else 0)
        venue_pts.setdefault((htid, 'H'), []).append(h_pts)
        venue_pts.setdefault((atid, 'A'), []).append(a_pts)

    return hph_l, hpa_l, aph_l, apa_l


# Snippet 3 — Rest Days (pipeline.py)
def compute_rest_days(df, default_days=7):
    last_match_date = {}
    home_rest_l, away_rest_l = [], []

    for _, row in df.iterrows():
        htid = row["home_team_id"]
        atid = row["away_team_id"]
        match_date = pd.Timestamp(row["date"])

        if htid in last_match_date:
            home_rest_l.append((match_date - last_match_date[htid]).days)
        else:
            home_rest_l.append(default_days)

        if atid in last_match_date:
            away_rest_l.append((match_date - last_match_date[atid]).days)
        else:
            away_rest_l.append(default_days)

        last_match_date[htid] = match_date
        last_match_date[atid] = match_date

    return home_rest_l, away_rest_l
