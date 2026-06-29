const request = require('supertest');

jest.mock('../services/footballApi', () => ({
  getTeamStats: jest.fn(),
  getStandings: jest.fn(),
}));

jest.mock('../services/mlService', () => ({
  predict: jest.fn(),
  getModelStats: jest.fn(),
  getPredictionHistory: jest.fn(),
  savePrediction: jest.fn(),
  healthCheck: jest.fn(),
}));

const express = require('express');
const predictionsRouter = require('../routes/predictions');
const { getTeamStats, getStandings } = require('../services/footballApi');
const { predict, getModelStats, getPredictionHistory, savePrediction } = require('../services/mlService');

const app = express();
app.use(express.json());
app.use('/api/predictions', predictionsRouter);

const mockTeamStats = {
  teamId: 57,
  teamName: "Arsenal",
  teamCrest: "https://example.com/arsenal.png",
  form_pts: 2.4,
  form_gf: 2.0,
  form_ga: 0.8,
  wins: 4,
  draws: 0,
  losses: 1,
  form: "WWWLW",
};

const mockStandings = {
  competition: "PL",
  table: [
    { teamId: 57, position: 1, team: "Arsenal", points: 75 },
    { teamId: 65, position: 2, team: "Man City", points: 73 },
    { teamId: 64, position: 3, team: "Liverpool", points: 70 },
  ],
};

const mockPrediction = {
  prediction: 1,
  label: "Home Win",
  probability_home_win: 0.72,
  probability_away_win: 0.28,
  confidence: "High",
  confidencePercent: 72,
};

describe('POST /api/predictions/predict', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    getTeamStats.mockResolvedValue(mockTeamStats);
    getStandings.mockResolvedValue(mockStandings);
    predict.mockResolvedValue(mockPrediction);
    savePrediction.mockResolvedValue({});
  });

  test('should return prediction for valid request', async () => {
    const response = await request(app)
      .post('/api/predictions/predict')
      .send({
        homeTeamId: 57,
        awayTeamId: 65,
        competition: 'PL',
      });

    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('prediction');
    expect(response.body.prediction).toHaveProperty('label', 'Home Win');
    expect(response.body.prediction).toHaveProperty('confidence', 'High');
    expect(response.body).toHaveProperty('homeTeam');
    expect(response.body).toHaveProperty('awayTeam');
  });

  test('should return 400 for missing homeTeamId', async () => {
    const response = await request(app)
      .post('/api/predictions/predict')
      .send({
        awayTeamId: 65,
        competition: 'PL',
      });

    expect(response.status).toBe(400);
    expect(response.body).toHaveProperty('error');
  });

  test('should return 400 for missing awayTeamId', async () => {
    const response = await request(app)
      .post('/api/predictions/predict')
      .send({
        homeTeamId: 57,
        competition: 'PL',
      });

    expect(response.status).toBe(400);
    expect(response.body).toHaveProperty('error');
  });

  test('should return 400 for missing competition', async () => {
    const response = await request(app)
      .post('/api/predictions/predict')
      .send({
        homeTeamId: 57,
        awayTeamId: 65,
      });

    expect(response.status).toBe(400);
    expect(response.body).toHaveProperty('error');
  });

  test('should return 400 for invalid competition', async () => {
    const response = await request(app)
      .post('/api/predictions/predict')
      .send({
        homeTeamId: 57,
        awayTeamId: 65,
        competition: 'INVALID',
      });

    expect(response.status).toBe(400);
    expect(response.body).toHaveProperty('error');
  });

  test('should accept uppercase competition codes', async () => {
    const response = await request(app)
      .post('/api/predictions/predict')
      .send({
        homeTeamId: 57,
        awayTeamId: 65,
        competition: 'pl',
      });

    expect(response.status).toBe(200);
  });

  test('should include rank in response', async () => {
    const response = await request(app)
      .post('/api/predictions/predict')
      .send({
        homeTeamId: 57,
        awayTeamId: 65,
        competition: 'PL',
      });

    expect(response.status).toBe(200);
    expect(response.body.homeTeam).toHaveProperty('rank');
    expect(response.body.awayTeam).toHaveProperty('rank');
  });

  test('should call ML service with all features', async () => {
    await request(app)
      .post('/api/predictions/predict')
      .send({
        homeTeamId: 57,
        awayTeamId: 65,
        competition: 'PL',
      });

    expect(predict).toHaveBeenCalledWith(
      expect.objectContaining({
        home_form_pts: expect.any(Number),
        away_form_pts: expect.any(Number),
        form_pts_diff: expect.any(Number),
        home_rank: expect.any(Number),
        away_rank: expect.any(Number),
        rank_diff: expect.any(Number),
        h2h_home_wins: expect.any(Number),
        h2h_away_wins: expect.any(Number),
      })
    );
  });
});

describe('GET /api/predictions/stats', () => {
  test('should return model stats', async () => {
    const mockStats = {
      model: 'XGBoost',
      test_metrics: {
        accuracy: 0.6883,
        precision: 0.6884,
        recall: 0.8444,
        f1_score: 0.7585,
      },
    };

    getModelStats.mockResolvedValue(mockStats);

    const response = await request(app)
      .get('/api/predictions/stats');

    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('model', 'XGBoost');
    expect(response.body).toHaveProperty('test_metrics');
  });

  test('should return 502 when ML service is unavailable', async () => {
    getModelStats.mockRejectedValue(new Error('ML service error: Connection refused'));

    const response = await request(app)
      .get('/api/predictions/stats');

    expect(response.status).toBe(502);
    expect(response.body).toHaveProperty('error');
  });
});

describe('GET /api/predictions/history', () => {
  test('should return prediction history', async () => {
    const mockHistory = [
      {
        id: 1234567890,
        timestamp: '2026-04-30T10:00:00.000Z',
        homeTeam: { name: 'Arsenal' },
        awayTeam: { name: 'Chelsea' },
        prediction: { label: 'Home Win' },
      },
    ];

    getPredictionHistory.mockResolvedValue(mockHistory);

    const response = await request(app)
      .get('/api/predictions/history');

    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('predictions');
    expect(response.body).toHaveProperty('count');
    expect(Array.isArray(response.body.predictions)).toBe(true);
  });

  test('should respect limit parameter', async () => {
    getPredictionHistory.mockResolvedValue([]);

    await request(app)
      .get('/api/predictions/history?limit=5');

    expect(getPredictionHistory).toHaveBeenCalledWith(5);
  });

  test('should use default limit of 20', async () => {
    getPredictionHistory.mockResolvedValue([]);

    await request(app)
      .get('/api/predictions/history');

    expect(getPredictionHistory).toHaveBeenCalledWith(20);
  });
});