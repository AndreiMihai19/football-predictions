const axios = require("axios");

const BASE_URL = "https://api.football-data.org/v4";
const VALID_COMPETITIONS = ["PL", "SA", "PD", "BL1", "FL1"];

const TTL_UPCOMING  = 10 * 60 * 1000;
const TTL_STANDINGS = 10 * 60 * 1000;
const TTL_TEAM      = 3  * 60 * 1000;
const TTL_H2H       = 60 * 60 * 1000;
const TTL_FINISHED  = 10 * 60 * 1000;

const cache = new Map();
const inflight = new Map();

function fromCache(key, ttl) {
  const entry = cache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.timestamp > ttl) {
    cache.delete(key);
    return null;
  }
  console.log(`[cache] HIT  ${key}`);
  return entry.data;
}

function toCache(key, data) {
  cache.set(key, { data, timestamp: Date.now() });
}

async function withCache(key, ttl, fn) {
  const cached = fromCache(key, ttl);
  if (cached !== null) return cached;

  if (inflight.has(key)) {
    console.log(`[cache] WAIT (inflight) ${key}`);
    return inflight.get(key);
  }

  const ttlMin = Math.round(ttl / 60000);
  console.log(`[cache] MISS ${key}  (TTL: ${ttlMin}min)`);
  const promise = fn().then((data) => {
    toCache(key, data);
    inflight.delete(key);
    return data;
  }).catch((err) => {
    inflight.delete(key);
    throw err;
  });

  inflight.set(key, promise);
  return promise;
}

function client() {
  return axios.create({
    baseURL: BASE_URL,
    headers: { "X-Auth-Token": process.env.FOOTBALL_API_KEY },
    timeout: 10_000,
  });
}

async function getUpcomingMatches(competition) {
  if (!VALID_COMPETITIONS.includes(competition)) {
    throw new Error(`Competitie invalida: ${competition}. Valide: ${VALID_COMPETITIONS.join(", ")}`);
  }

  return withCache(`upcoming_${competition}`, TTL_UPCOMING, () => _fetchUpcoming(competition));
}

async function _fetchUpcoming(competition) {
  let rawMatches = [];

  try {
    const { data } = await client().get(`/competitions/${competition}/matches`, {
      params: { status: "SCHEDULED" },
    });
    rawMatches = data.matches || [];
  } catch (err) {
    const httpStatus = err.response?.status;
    console.error(`[footballApi] SCHEDULED fetch failed (HTTP ${httpStatus}) pentru ${competition}:`, err.response?.data ?? err.message);
    if (httpStatus !== 429) throw err;
  }

  if (rawMatches.length === 0) {
    console.warn(`[footballApi] SCHEDULED gol pentru ${competition}, incerc TIMED...`);
    try {
      const { data } = await client().get(`/competitions/${competition}/matches`, {
        params: { status: "TIMED" },
      });
      rawMatches = data.matches || [];
    } catch (err) {
      console.error(`[footballApi] TIMED fetch failed pentru ${competition}:`, err.message);
    }
  }

  const todayStart = new Date();
  todayStart.setHours(0, 0, 0, 0);

  const upcoming = rawMatches
    .filter((m) => new Date(m.utcDate) >= todayStart)
    .sort((a, b) => new Date(a.utcDate) - new Date(b.utcDate))
    .slice(0, 20);

  console.log(`[matches] ${upcoming.length} meciuri viitoare dupa filtrare din ${rawMatches.length} total (${competition})`);

  const mapped = upcoming.map((m) => ({
    id:           m.id,
    utcDate:      m.utcDate,
    competition:  competition,
    homeTeam: {
      id:   m.homeTeam.id,
      name: m.homeTeam.name,
      crest: m.homeTeam.crest ?? null,
    },
    awayTeam: {
      id:   m.awayTeam.id,
      name: m.awayTeam.name,
      crest: m.awayTeam.crest ?? null,
    },
    matchday: m.matchday ?? null,
  }));

  if (mapped.length > 0) return mapped;

  console.warn(`[matches] Niciun meci viitor pentru ${competition}`);
  return [];
}

async function getTeamStats(teamId) {
  return withCache(`team_${teamId}`, TTL_TEAM, () => _fetchTeamStats(teamId));
}

async function _fetchTeamStats(teamId) {
  const { data } = await client().get(`/teams/${teamId}/matches`, {
    params: { status: "FINISHED", limit: 10 },
  });

  const matches = data.matches || [];

  let wins = 0, losses = 0, draws = 0;
  let goalsFor = 0, goalsAgainst = 0;
  const formString = [];

  const chrono = [...matches].reverse();

  const homeVenuePts = [];
  const awayVenuePts = [];
  const goalsForPerMatch = [];
  const expPtsAll = [];

  let lastDate = null;

  for (const m of matches) {
    const isHome   = m.homeTeam.id === teamId;
    const scoreFor = isHome ? m.score.fullTime.home : m.score.fullTime.away;
    const scoreAga = isHome ? m.score.fullTime.away : m.score.fullTime.home;

    if (scoreFor == null || scoreAga == null) continue;

    goalsFor     += scoreFor;
    goalsAgainst += scoreAga;

    if (scoreFor > scoreAga)       { wins++;   formString.push("W"); }
    else if (scoreFor < scoreAga)  { losses++; formString.push("L"); }
    else                           { draws++;  formString.push("D"); }

    if (lastDate === null || new Date(m.utcDate) > lastDate) {
      lastDate = new Date(m.utcDate);
    }
  }

  for (const m of chrono) {
    const isHome   = m.homeTeam.id === teamId;
    const scoreFor = isHome ? m.score.fullTime.home : m.score.fullTime.away;
    const scoreAga = isHome ? m.score.fullTime.away : m.score.fullTime.home;
    if (scoreFor == null || scoreAga == null) continue;

    const pts = scoreFor > scoreAga ? 3 : scoreFor === scoreAga ? 1 : 0;
    expPtsAll.push(pts);
    goalsForPerMatch.push(scoreFor);

    if (isHome) homeVenuePts.push(pts);
    else        awayVenuePts.push(pts);
  }

  const n = matches.length || 1;

  const formPts = (wins * 3 + draws * 1) / n;

  const avg = (arr) => arr.length === 0 ? 1.0 : arr.reduce((a, b) => a + b, 0) / arr.length;
  const form_pts_home = parseFloat(avg(homeVenuePts).toFixed(3));
  const form_pts_away = parseFloat(avg(awayVenuePts).toFixed(3));

  const decay = 0.7;
  const N_EXP = 7;
  const recent7 = expPtsAll.slice(-N_EXP);
  let weightedSum = 0, weightTotal = 0;
  recent7.forEach((pts, i) => {
    const w = Math.pow(decay, recent7.length - 1 - i);
    weightedSum += pts * w;
    weightTotal += w;
  });
  const form_pts_exp = recent7.length === 0 ? 1.0 : parseFloat((weightedSum / weightTotal).toFixed(3));

  const last5  = goalsForPerMatch.slice(-5);
  const prior5 = goalsForPerMatch.slice(-10, -5);
  const meanLast  = last5.length  ? last5.reduce((a, b) => a + b, 0) / last5.length  : 0;
  const meanPrior = prior5.length ? prior5.reduce((a, b) => a + b, 0) / prior5.length : 0;
  const goals_trend = parseFloat((meanLast - meanPrior).toFixed(3));

  const days_since_last = lastDate
    ? Math.max(0, Math.round((Date.now() - lastDate.getTime()) / (24 * 60 * 60 * 1000)))
    : 7;

  const firstMatch = matches[0];
  const teamSide   = firstMatch
    ? (firstMatch.homeTeam.id === teamId ? firstMatch.homeTeam : firstMatch.awayTeam)
    : null;

  return {
    teamId,
    teamName:  teamSide?.name  ?? null,
    teamCrest: teamSide?.crest ?? null,
    matchesPlayed:    matches.length,
    wins,
    losses,
    draws,
    form:             formString.join(""),
    goalsFor,
    goalsAgainst,
    form_pts:         parseFloat(formPts.toFixed(3)),
    form_gf:          parseFloat((goalsFor / n).toFixed(3)),
    form_ga:          parseFloat((goalsAgainst / n).toFixed(3)),
    form_pts_home,
    form_pts_away,
    form_pts_exp,
    goals_trend,
    days_since_last,
    recentMatches:    matches.slice(0, 5).map((m) => ({
      id:       m.id,
      utcDate:  m.utcDate,
      home:     m.homeTeam.name,
      away:     m.awayTeam.name,
      score:    `${m.score.fullTime.home}-${m.score.fullTime.away}`,
    })),
  };
}

async function getStandings(competition) {
  if (!VALID_COMPETITIONS.includes(competition)) {
    throw new Error(`Competitie invalida: ${competition}. Valide: ${VALID_COMPETITIONS.join(", ")}`);
  }

  return withCache(`standings_${competition}`, TTL_STANDINGS, () => _fetchStandings(competition));
}

async function _fetchStandings(competition) {
  const { data } = await client().get(`/competitions/${competition}/standings`);

  const table = data.standings?.find((s) => s.type === "TOTAL")?.table ?? [];

  return {
    competition,
    season:    data.season?.startDate?.slice(0, 4) ?? "N/A",
    updatedAt: data.season?.currentMatchday ?? null,
    table: table.map((row) => ({
      position:    row.position,
      team:        row.team.name,
      teamId:      row.team.id,
      crest:       row.team.crest ?? null,
      playedGames: row.playedGames,
      won:         row.won,
      draw:        row.draw,
      lost:        row.lost,
      goalsFor:    row.goalsFor,
      goalsAgainst: row.goalsAgainst,
      goalDifference: row.goalDifference,
      points:      row.points,
    })),
  };
}

async function getH2H(homeTeamId, awayTeamId) {
  const key = `h2h_${homeTeamId}_${awayTeamId}`;
  return withCache(key, TTL_H2H, () => _fetchH2H(homeTeamId, awayTeamId));
}

async function _fetchH2H(homeTeamId, awayTeamId) {
  const { data } = await client().get(`/teams/${homeTeamId}/matches`, {
    params: { status: "FINISHED", limit: 30 },
  });

  const h2hMatches = (data.matches || []).filter((m) =>
    (m.homeTeam.id === homeTeamId && m.awayTeam.id === awayTeamId) ||
    (m.homeTeam.id === awayTeamId && m.awayTeam.id === homeTeamId)
  );

  let homeWins = 0, awayWins = 0, draws = 0;

  for (const m of h2hMatches) {
    const hs = m.score.fullTime.home;
    const as = m.score.fullTime.away;
    if (hs == null || as == null) continue;

    const homeTeamIsHome = m.homeTeam.id === homeTeamId;
    const homeTeamScore  = homeTeamIsHome ? hs : as;
    const awayTeamScore  = homeTeamIsHome ? as : hs;

    if (homeTeamScore > awayTeamScore)      homeWins++;
    else if (awayTeamScore > homeTeamScore) awayWins++;
    else                                    draws++;
  }

  const total = homeWins + awayWins + draws;

  console.log(`[h2h] ${homeTeamId} vs ${awayTeamId}: ${homeWins}W ${draws}D ${awayWins}L (${total} meciuri)`);

  return {
    h2h_home_wins: homeWins || 1,
    h2h_away_wins: awayWins || 1,
    h2h_draws:     draws,
    h2h_total:     total,
    recentH2H: h2hMatches.slice(0, 5).map((m) => ({
      date:  m.utcDate,
      home:  m.homeTeam.name,
      away:  m.awayTeam.name,
      score: `${m.score.fullTime.home}-${m.score.fullTime.away}`,
    })),
  };
}

async function getFinishedMatches(competition) {
  const key = `finished_${competition}`;
  return withCache(key, TTL_FINISHED, () => _fetchFinishedMatches(competition));
}

async function _fetchFinishedMatches(competition) {
  const dateTo = new Date();
  const dateFrom = new Date();
  dateFrom.setDate(dateFrom.getDate() - 30);

  const fmt = (d) => d.toISOString().slice(0, 10);

  let matches = [];

  try {
    const { data } = await client().get(`/competitions/${competition}/matches`, {
      params: {
        status: "FINISHED",
        dateFrom: fmt(dateFrom),
        dateTo: fmt(dateTo),
      },
    });
    matches = data.matches || [];
  } catch (err) {
    console.warn(`[footballApi] finished with date filter failed for ${competition}, falling back:`, err.response?.status);
    const { data } = await client().get(`/competitions/${competition}/matches`, {
      params: { status: "FINISHED" },
    });
    matches = data.matches || [];
  }

  matches.sort((a, b) => new Date(b.utcDate) - new Date(a.utcDate));

  return matches.slice(0, 100).map((m) => ({
    utcDate:   m.utcDate,
    homeTeam:  { id: m.homeTeam.id, name: m.homeTeam.name },
    awayTeam:  { id: m.awayTeam.id, name: m.awayTeam.name },
    homeScore: m.score.fullTime.home,
    awayScore: m.score.fullTime.away,
  }));
}

module.exports = { getUpcomingMatches, getTeamStats, getStandings, getH2H, getFinishedMatches };
