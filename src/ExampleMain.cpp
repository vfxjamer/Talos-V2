// ============================================================================
// TALON V2 — Full 10-Phase Progressive Training System (GigaLearnCPP)
// Target: Champ/GC Level Bot | 48B Total Steps | 1024x4 LEAKY_RELU + LayerNorm
// ============================================================================

#include <GigaLearnCPP/Learner.h>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/OBSBuilders/DefaultObsPadded.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include <filesystem>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <unordered_map>
#include <ctime>

using namespace GGL;
using namespace RLGC;

// ============================================================================
// PHASE BOUNDARIES (in steps)
// ============================================================================

static constexpr int64_t PHASE_1_END   = 1000000000LL;      // 1B
static constexpr int64_t PHASE_2_END   = 3000000000LL;      // 3B
static constexpr int64_t PHASE_3_END   = 7000000000LL;      // 7B
static constexpr int64_t PHASE_4_END   = 16000000000LL;     // 16B
static constexpr int64_t PHASE_5_END   = 26000000000LL;     // 26B
static constexpr int64_t PHASE_6_END   = 32000000000LL;     // 32B
static constexpr int64_t PHASE_7_END   = 40000000000LL;     // 40B
static constexpr int64_t PHASE_8_END   = 44000000000LL;     // 44B
static constexpr int64_t PHASE_9_END   = 47000000000LL;     // 47B
static constexpr int64_t PHASE_10_END  = 48000000000LL;     // 48B

// ============================================================================
// TRAINING STATS
// ============================================================================

struct TrainingStats {
    int64_t totalSteps     = 0;
    int64_t phaseSteps    = 0;
    int     currentPhase  = 1;
    int64_t phaseStartStep = 0;
    int     episodes      = 0;

    float avgGoalsFor     = 0.0f;
    float avgGoalsAgainst = 0.0f;
    float avgTouches      = 0.0f;
    float avgSaves        = 0.0f;
    float avgShots        = 0.0f;
    float avgDemos        = 0.0f;
};

static TrainingStats g_stats;

// ============================================================================
// PHASE CONFIGURATION
// ============================================================================

struct PhaseConfig {
    float       gamma;
    float       entropy;
    const char* name;
    const char* description;
    int64_t     startStep;
    int64_t     endStep;
};

static PhaseConfig PHASES[] = {
    { 0.0f, 0.0f, "INVALID", "INVALID", 0LL, 0LL },

    { 0.990f, 0.05f,
      "Phase 1: Kickoff Mastery",
      "Kickoffs, first touches, speed flips, positioning",
      0LL, PHASE_1_END },

    { 0.991f, 0.06f,
      "Phase 2: Ball Control",
      "Dribbling, ball touches, ground play",
      PHASE_1_END, PHASE_2_END },

    { 0.992f, 0.08f,
      "Phase 3: Basic Gameplay",
      "Goals, shots, saves, demos, boost management",
      PHASE_2_END, PHASE_3_END },

    { 0.993f, 0.10f,
      "Phase 4: Aerial Introduction",
      "Aerials, jump rewards, air control",
      PHASE_3_END, PHASE_4_END },

    { 0.994f, 0.10f,
      "Phase 5: Aerial Mastery",
      "Musty flicks, double taps, aerial shots",
      PHASE_4_END, PHASE_5_END },

    { 0.995f, 0.08f,
      "Phase 6: Game Sense Introduction",
      "Rotation, shadow defense, challenge timing",
      PHASE_5_END, PHASE_6_END },

    { 0.996f, 0.07f,
      "Phase 7: Game Sense Mastery",
      "Pressure, counter attacks, boost stealing",
      PHASE_6_END, PHASE_7_END },

    { 0.997f, 0.06f,
      "Phase 8: Mechanical Ceiling",
      "Speed flips, wave dashes, double taps",
      PHASE_7_END, PHASE_8_END },

    { 0.998f, 0.05f,
      "Phase 9: Full Games",
      "Wins matter most now",
      PHASE_8_END, PHASE_9_END },

    { 0.999f, 0.03f,
      "Phase 10: Optimization",
      "Pure win optimization",
      PHASE_9_END, PHASE_10_END },
};

// ============================================================================
// AUTO PHASE DETECTION
// ============================================================================

int DetectPhaseFromSteps(int64_t totalSteps) {
    if (totalSteps >= PHASE_10_END) return 10;
    if (totalSteps >= PHASE_9_END)  return 9;
    if (totalSteps >= PHASE_8_END)  return 8;
    if (totalSteps >= PHASE_7_END)  return 7;
    if (totalSteps >= PHASE_6_END)  return 6;
    if (totalSteps >= PHASE_5_END)  return 5;
    if (totalSteps >= PHASE_4_END)  return 4;
    if (totalSteps >= PHASE_3_END)  return 3;
    if (totalSteps >= PHASE_2_END)  return 2;
    return 1;
}

// ============================================================================
// CHECKPOINT LOADING
// ============================================================================

int64_t LoadCheckpointSteps() {
    std::filesystem::path checkpointPath = "checkpoints";
    if (!std::filesystem::exists(checkpointPath)) {
        return 0;
    }

    int64_t maxStep = 0;
    for (const auto& entry : std::filesystem::directory_iterator(checkpointPath)) {
        if (entry.is_directory()) {
            std::string name = entry.path().filename().string();
            try {
                int64_t step = std::stoll(name);
                if (step > maxStep) {
                    maxStep = step;
                }
            } catch (...) {
                // Ignore non-numeric folder names
            }
        }
    }

    return maxStep;
}

// ============================================================================
// CUSTOM REWARD CLASSES
// ============================================================================

// ----------------------------------------------------------------------------
// KickoffSpeedReward - Phase 1
// Reward for speed at first ball touch
// ----------------------------------------------------------------------------
class KickoffSpeedReward : public Reward {
public:
    static constexpr float MAX_SPEED = 2300.0f;

    void Reset(const GameState& initialState) override {}

    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        // Only active during kickoff (ball not yet moving fast)
        if (state.ball.vel.Length() > 400.0f) return 0.0f;

        // Check if this player touched the ball
        if (!player.ballTouchedStep) return 0.0f;

        float speed = player.vel.Length();
        return std::min(speed / MAX_SPEED, 1.0f);
    }
};

// ----------------------------------------------------------------------------
// KickoffFirstTouchReward - Phase 1
// Big reward for touching ball before opponent
// ----------------------------------------------------------------------------
class KickoffFirstTouchReward : public Reward {
public:
    struct PlayerState {
        bool touchedBall = false;
        bool opponentTouchedFirst = false;
    };
    std::unordered_map<uint32_t, PlayerState> playerStates;

    void Reset(const GameState& initialState) override {
        playerStates.clear();
        for (auto& p : initialState.players) {
            playerStates[p.carId] = {};
        }
    }

    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        if (state.ball.vel.Length() > 400.0f) return 0.0f;

        auto& ps = playerStates[player.carId];

        // Check if opponent touched first
        for (auto& other : state.players) {
            if (other.carId == player.carId) continue;
            if (other.team == player.team) continue;
            if (other.ballTouchedStep && !ps.touchedBall) {
                ps.opponentTouchedFirst = true;
            }
        }

        if (player.ballTouchedStep && !ps.touchedBall) {
            ps.touchedBall = true;
            if (!ps.opponentTouchedFirst) {
                return 1.0f;  // We got first touch!
            } else {
                return -0.5f; // Opponent got it first
            }
        }

        return 0.0f;
    }
};

// ----------------------------------------------------------------------------
// KickoffDirectionReward - Phase 1
// Reward for hitting ball toward opponent goal
// ----------------------------------------------------------------------------
class KickoffDirectionReward : public Reward {
public:
    void Reset(const GameState& initialState) override {}

    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        if (state.ball.vel.Length() > 400.0f) return 0.0f;
        if (!player.ballTouchedStep) return 0.0f;

        float ballSpeed = state.ball.vel.Length();
        if (ballSpeed < 100.0f) return 0.0f;

        // Blue wants +y, Orange wants -y
        float direction = (player.team == Team::BLUE) ? state.ball.vel.y : -state.ball.vel.y;
        return std::max(0.0f, direction / 6000.0f);
    }
};

// ----------------------------------------------------------------------------
// SpeedFlipDetectionReward - Phase 1, 8, 10
// Detect diagonal flip cancel during kickoff (speed flip)
// ----------------------------------------------------------------------------
class SpeedFlipDetectionReward : public Reward {
public:
    struct PlayerState {
        bool wasInAir = false;
        bool doingSpeedFlip = false;
    };
    std::unordered_map<uint32_t, PlayerState> playerStates;

    void Reset(const GameState& initialState) override {
        playerStates.clear();
        for (auto& p : initialState.players) {
            playerStates[p.carId] = {};
        }
    }

    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        if (state.ball.vel.Length() > 400.0f) return 0.0f;

        auto& ps = playerStates[player.carId];
        float speed = player.vel.Length();

        // Detect diagonal flip + forward momentum + high speed
        bool isDiagonalFlip = (player.prevAction.jump > 0 && player.prevAction.handbrake > 0);
        bool justLeftGround = !ps.wasInAir && !player.isOnGround;
        bool hasSpeed = speed > 2000.0f;

        if (isDiagonalFlip && justLeftGround && hasSpeed) {
            ps.doingSpeedFlip = true;
            return 2.0f;
        }

        ps.wasInAir = !player.isOnGround;

        if (ps.doingSpeedFlip && player.isOnGround) {
            ps.doingSpeedFlip = false;
            return 1.0f; // Completed speed flip
        }

        return 0.0f;
    }
};

// ----------------------------------------------------------------------------
// AerialTouchReward - Phase 4, 5
// Reward touching ball while both car and ball are in the air
// ----------------------------------------------------------------------------
class AerialTouchReward : public Reward {
public:
    void Reset(const GameState& initialState) override {}

    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        // Both car and ball must be airborne (height > 300uu)
        if (player.pos.z < 300.0f) return 0.0f;
        if (player.isOnGround) return 0.0f;

        if (!player.ballTouchedStep) return 0.0f;

        // Scale by height of touch (higher = more reward)
        float height = player.pos.z;
        return (height - 300.0f) / 1000.0f + 0.5f;
    }
};

// ============================================================================
// REWARD GETTERS PER PHASE (as specified in requirements)
// ============================================================================

std::vector<WeightedReward> GetPhase1Rewards() {
    // Phase 1: Kickoff Mastery (0-1B steps)
    // StateSetter: KickoffState
    return std::vector<WeightedReward>{
        WeightedReward(new KickoffSpeedReward(),          3.0f),
        WeightedReward(new KickoffFirstTouchReward(),     5.0f),
        WeightedReward(new KickoffDirectionReward(),      4.0f),
        WeightedReward(new SpeedFlipDetectionReward(),    6.0f),
        WeightedReward(new VelocityPlayerToBallReward(),  2.0f),
        WeightedReward(new SaveBoostReward(0.5f),         0.2f),
    };
}

std::vector<WeightedReward> GetPhase2Rewards() {
    // Phase 2: Ball Control (1B-3B steps)
    // StateSetter: CombinedState: KickoffState 40%, RandomState 60%
    return std::vector<WeightedReward>{
        WeightedReward(new TouchBallReward(),              2.0f),
        WeightedReward(new VelocityPlayerToBallReward(),  4.0f),
        WeightedReward(new TouchAccelReward(),             3.0f),
        WeightedReward(new StrongTouchReward(20, 100),    60.0f),
        WeightedReward(new VelocityBallToGoalReward(),    2.0f),
        WeightedReward(new PickupBoostReward(),            10.0f),
        WeightedReward(new SaveBoostReward(0.5f),         0.2f),
        WeightedReward(new GoalReward(),                  100.0f),
    };
}

std::vector<WeightedReward> GetPhase3Rewards() {
    // Phase 3: Basic Gameplay (3B-7B steps)
    // StateSetter: CombinedState: KickoffState 20%, RandomState 80%
    return std::vector<WeightedReward>{
        WeightedReward(new GoalReward(),                   150.0f),
        WeightedReward(new ZeroSumReward(new SaveReward(), 0.5f), 80.0f),
        WeightedReward(new ZeroSumReward(new DemoReward(), 0.5f), 60.0f),
        WeightedReward(new StrongTouchReward(20, 100),    60.0f),
        WeightedReward(new VelocityBallToGoalReward(),    2.0f),
        WeightedReward(new VelocityPlayerToBallReward(),  4.0f),
        WeightedReward(new PickupBoostReward(),            10.0f),
        WeightedReward(new SaveBoostReward(0.5f),         0.2f),
        WeightedReward(new AirReward(),                   0.25f),
    };
}

std::vector<WeightedReward> GetPhase4Rewards() {
    // Phase 4: Aerial Introduction (7B-16B steps)
    // StateSetter: CombinedState: KickoffState 20%, RandomState 80%
    return std::vector<WeightedReward>{
        WeightedReward(new GoalReward(),                   150.0f),
        WeightedReward(new AirReward(),                    2.0f),
        WeightedReward(new AerialTouchReward(),           6.0f),
        WeightedReward(new StrongTouchReward(20, 100),    60.0f),
        WeightedReward(new VelocityBallToGoalReward(),    2.0f),
        WeightedReward(new VelocityPlayerToBallReward(),  4.0f),
        WeightedReward(new PickupBoostReward(),            10.0f),
        WeightedReward(new SaveBoostReward(0.5f),         0.5f),
        WeightedReward(new ZeroSumReward(new SaveReward(), 0.5f), 80.0f),
    };
}

std::vector<WeightedReward> GetPhase5Rewards() {
    // Phase 5: Aerial Mastery (16B-26B steps)
    // StateSetter: CombinedState: KickoffState 15%, RandomState 85%
    return std::vector<WeightedReward>{
        WeightedReward(new GoalReward(),                   150.0f),
        WeightedReward(new AerialTouchReward(),           8.0f),
        WeightedReward(new StrongTouchReward(20, 130),    80.0f),
        WeightedReward(new VelocityBallToGoalReward(),    3.0f),
        WeightedReward(new ZeroSumReward(new SaveReward(), 0.5f), 80.0f),
        WeightedReward(new ZeroSumReward(new DemoReward(), 0.5f), 60.0f),
        WeightedReward(new PickupBoostReward(),            10.0f),
        WeightedReward(new SaveBoostReward(0.5f),         0.5f),
        WeightedReward(new AirReward(),                    1.0f),
    };
}

std::vector<WeightedReward> GetPhase6Rewards() {
    // Phase 6: Game Sense Introduction (26B-32B steps)
    // StateSetter: CombinedState: KickoffState 30%, RandomState 70%
    return std::vector<WeightedReward>{
        WeightedReward(new GoalReward(),                   150.0f),
        WeightedReward(new ZeroSumReward(new SaveReward(), 0.5f), 100.0f),
        WeightedReward(new ZeroSumReward(new DemoReward(), 0.5f), 60.0f),
        WeightedReward(new StrongTouchReward(20, 130),    60.0f),
        WeightedReward(new VelocityBallToGoalReward(),    3.0f),
        WeightedReward(new VelocityPlayerToBallReward(),  4.0f),
        WeightedReward(new PickupBoostReward(),            10.0f),
        WeightedReward(new SaveBoostReward(0.5f),         1.0f),
        WeightedReward(new FaceBallReward(),               0.25f),
    };
}

std::vector<WeightedReward> GetPhase7Rewards() {
    // Phase 7: Game Sense Mastery (32B-40B steps)
    // StateSetter: CombinedState: KickoffState 40%, RandomState 60%
    return std::vector<WeightedReward>{
        WeightedReward(new GoalReward(),                   150.0f),
        WeightedReward(new ZeroSumReward(new SaveReward(), 0.5f), 100.0f),
        WeightedReward(new ZeroSumReward(new DemoReward(), 0.5f), 80.0f),
        WeightedReward(new StrongTouchReward(20, 130),    60.0f),
        WeightedReward(new VelocityBallToGoalReward(),    3.0f),
        WeightedReward(new PickupBoostReward(),            10.0f),
        WeightedReward(new SaveBoostReward(0.5f),         1.0f),
        WeightedReward(new AirReward(),                    0.5f),
        WeightedReward(new WavedashReward(),               4.0f),
    };
}

std::vector<WeightedReward> GetPhase8Rewards() {
    // Phase 8: Mechanical Ceiling (40B-44B steps)
    // StateSetter: CombinedState: KickoffState 30%, RandomState 70%
    return std::vector<WeightedReward>{
        WeightedReward(new GoalReward(),                   150.0f),
        WeightedReward(new ZeroSumReward(new SaveReward(), 0.5f), 100.0f),
        WeightedReward(new ZeroSumReward(new DemoReward(), 0.5f), 80.0f),
        WeightedReward(new StrongTouchReward(20, 130),    80.0f),
        WeightedReward(new WavedashReward(),               4.0f),
        WeightedReward(new SpeedFlipDetectionReward(),    4.0f),
        WeightedReward(new AirReward(),                    1.0f),
        WeightedReward(new PickupBoostReward(),            10.0f),
        WeightedReward(new VelocityBallToGoalReward(),    3.0f),
    };
}

std::vector<WeightedReward> GetPhase9Rewards() {
    // Phase 9: Full Games (44B-47B steps)
    // StateSetter: CombinedState: KickoffState 50%, RandomState 50%
    return std::vector<WeightedReward>{
        WeightedReward(new GoalReward(),                   200.0f),
        WeightedReward(new ZeroSumReward(new SaveReward(), 0.5f), 120.0f),
        WeightedReward(new ZeroSumReward(new DemoReward(), 0.5f), 80.0f),
        WeightedReward(new StrongTouchReward(20, 130),    60.0f),
        WeightedReward(new VelocityBallToGoalReward(),    3.0f),
        WeightedReward(new PickupBoostReward(),            10.0f),
        WeightedReward(new SaveBoostReward(0.5f),         1.0f),
        WeightedReward(new AirReward(),                    0.5f),
    };
}

std::vector<WeightedReward> GetPhase10Rewards() {
    // Phase 10: Optimization (47B-48B steps)
    // StateSetter: CombinedState: KickoffState 60%, RandomState 40%
    return std::vector<WeightedReward>{
        WeightedReward(new GoalReward(),                   250.0f),
        WeightedReward(new ZeroSumReward(new SaveReward(), 0.5f), 150.0f),
        WeightedReward(new ZeroSumReward(new DemoReward(), 0.5f), 100.0f),
        WeightedReward(new StrongTouchReward(20, 130),    60.0f),
        WeightedReward(new VelocityBallToGoalReward(),    3.0f),
        WeightedReward(new PickupBoostReward(),            10.0f),
        WeightedReward(new SaveBoostReward(0.5f),         2.0f),
        WeightedReward(new SpeedFlipDetectionReward(),    3.0f),
    };
}

// ============================================================================
// STATE SETTERS PER PHASE
// ============================================================================

StateSetter* GetPhaseStateSetter(int phase) {
    switch (phase) {
        case 1:
            return new KickoffState();

        case 2: {
            // CombinedState: KickoffState 40%, RandomState 60%
            std::vector<std::pair<StateSetter*, float>> s = {
                { new KickoffState(), 0.4f },
                { new RandomState(true, true, true), 0.6f },
            };
            return new CombinedState(s);
        }

        case 3: {
            // CombinedState: KickoffState 20%, RandomState 80%
            std::vector<std::pair<StateSetter*, float>> s = {
                { new KickoffState(), 0.2f },
                { new RandomState(true, true, true), 0.8f },
            };
            return new CombinedState(s);
        }

        case 4: {
            // CombinedState: KickoffState 20%, RandomState 80%
            std::vector<std::pair<StateSetter*, float>> s = {
                { new KickoffState(), 0.2f },
                { new RandomState(true, true, true), 0.8f },
            };
            return new CombinedState(s);
        }

        case 5: {
            // CombinedState: KickoffState 15%, RandomState 85%
            std::vector<std::pair<StateSetter*, float>> s = {
                { new KickoffState(), 0.15f },
                { new RandomState(true, true, true), 0.85f },
            };
            return new CombinedState(s);
        }

        case 6: {
            // CombinedState: KickoffState 30%, RandomState 70%
            std::vector<std::pair<StateSetter*, float>> s = {
                { new KickoffState(), 0.3f },
                { new RandomState(true, true, true), 0.7f },
            };
            return new CombinedState(s);
        }

        case 7: {
            // CombinedState: KickoffState 40%, RandomState 60%
            std::vector<std::pair<StateSetter*, float>> s = {
                { new KickoffState(), 0.4f },
                { new RandomState(true, true, true), 0.6f },
            };
            return new CombinedState(s);
        }

        case 8: {
            // CombinedState: KickoffState 30%, RandomState 70%
            std::vector<std::pair<StateSetter*, float>> s = {
                { new KickoffState(), 0.3f },
                { new RandomState(true, true, true), 0.7f },
            };
            return new CombinedState(s);
        }

        case 9: {
            // CombinedState: KickoffState 50%, RandomState 50%
            std::vector<std::pair<StateSetter*, float>> s = {
                { new KickoffState(), 0.5f },
                { new RandomState(true, true, true), 0.5f },
            };
            return new CombinedState(s);
        }

        case 10: {
            // CombinedState: KickoffState 60%, RandomState 40%
            std::vector<std::pair<StateSetter*, float>> s = {
                { new KickoffState(), 0.6f },
                { new RandomState(true, true, true), 0.4f },
            };
            return new CombinedState(s);
        }

        default:
            return new KickoffState();
    }
}

// ============================================================================
// TERMINAL CONDITIONS PER PHASE
// ============================================================================

std::vector<TerminalCondition*> GetPhaseTerminals(int phase) {
    std::vector<TerminalCondition*> conds;
    switch (phase) {
        case 1:  conds.push_back(new NoTouchCondition(10)); break;
        case 2:  conds.push_back(new NoTouchCondition(12)); break;
        case 3:  conds.push_back(new NoTouchCondition(15)); break;
        case 4:  conds.push_back(new NoTouchCondition(18)); break;
        case 5:  conds.push_back(new NoTouchCondition(20)); break;
        case 6:  conds.push_back(new NoTouchCondition(25)); break;
        case 7:  conds.push_back(new NoTouchCondition(30)); break;
        case 8:  conds.push_back(new NoTouchCondition(35)); break;
        case 9:  conds.push_back(new NoTouchCondition(40)); break;
        case 10: conds.push_back(new NoTouchCondition(45)); break;
        default: conds.push_back(new NoTouchCondition(15)); break;
    }
    conds.push_back(new GoalScoreCondition());
    return conds;
}

// ============================================================================
// ENVIRONMENT CREATION
// ============================================================================

std::vector<WeightedReward> GetRewardsForPhase(int phase) {
    switch (phase) {
        case 1:  return GetPhase1Rewards();
        case 2:  return GetPhase2Rewards();
        case 3:  return GetPhase3Rewards();
        case 4:  return GetPhase4Rewards();
        case 5:  return GetPhase5Rewards();
        case 6:  return GetPhase6Rewards();
        case 7:  return GetPhase7Rewards();
        case 8:  return GetPhase8Rewards();
        case 9:  return GetPhase9Rewards();
        case 10: return GetPhase10Rewards();
        default: return GetPhase1Rewards();
    }
}

EnvCreateResult EnvCreateFunc(int index) {
    int phase = g_stats.currentPhase;

    auto rewards       = GetRewardsForPhase(phase);
    auto terminals     = GetPhaseTerminals(phase);
    auto stateSetter   = GetPhaseStateSetter(phase);

    auto arena = Arena::Create(GameMode::SOCCAR);

    // Strict 1v1 — Octane hitbox
    arena->AddCar(Team::BLUE,   CAR_CONFIG_OCTANE);
    arena->AddCar(Team::ORANGE, CAR_CONFIG_OCTANE);

    auto obsBuilder   = new DefaultObsPadded(2);
    auto actionParser = new DefaultAction();

    EnvCreateResult result    = {};
    result.actionParser       = actionParser;
    result.obsBuilder         = obsBuilder;
    result.stateSetter        = stateSetter;
    result.terminalConditions = terminals;
    result.rewards            = rewards;
    result.arena              = arena;

    return result;
}

// ============================================================================
// PHASE TRANSITION CHECK
// ============================================================================

void CheckPhaseTransition() {
    int detectedPhase = DetectPhaseFromSteps(g_stats.totalSteps);

    if (detectedPhase != g_stats.currentPhase) {
        printf("\n");
        printf("============================================================\n");
        printf("  >>> PHASE TRANSITION: %d -> %d <<<\n", g_stats.currentPhase, detectedPhase);
        printf("  Total Steps: %lld\n", g_stats.totalSteps);
        printf("============================================================\n\n");

        g_stats.currentPhase = detectedPhase;
        g_stats.phaseStartStep = g_stats.totalSteps;
    }
}

// ============================================================================
// STEP CALLBACK
// ============================================================================

void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
    g_stats.totalSteps += (int64_t)states.size();
    g_stats.phaseSteps += (int64_t)states.size();

    // Check for phase transition
    CheckPhaseTransition();

    int goalsFor     = 0;
    int goalsAgainst = 0;
    int touches      = 0;
    int saves        = 0;
    int shots        = 0;
    int demos        = 0;

    for (auto& state : states) {
        if (state.goalScored) {
            if (state.ball.pos.y > 0) goalsFor++;
            else                       goalsAgainst++;
        }

        for (auto& player : state.players) {
            if (player.team == Team::BLUE) {
                if (player.ballTouchedStep) {
                    touches++;
                }

                // Check for saves (ball near our goal, we touched it)
                float goalY = -5120.0f;
                if (std::abs(state.ball.pos.y - goalY) < 500.0f && player.ballTouchedStep) {
                    saves++;
                }

                // Check for shots on goal
                float oppGoalY = 5120.0f;
                if (std::abs(state.ball.pos.y - oppGoalY) < 1000.0f && player.ballTouchedStep) {
                    shots++;
                }

                demos += player.eventState.demo ? 1 : 0;
            }
        }
    }

    g_stats.episodes++;
    float n = (float)g_stats.episodes;
    g_stats.avgGoalsFor      = (g_stats.avgGoalsFor      * (n-1) + goalsFor)      / n;
    g_stats.avgGoalsAgainst = (g_stats.avgGoalsAgainst * (n-1) + goalsAgainst) / n;
    g_stats.avgTouches      = (g_stats.avgTouches      * (n-1) + touches)      / n;
    g_stats.avgSaves        = (g_stats.avgSaves        * (n-1) + saves)        / n;
    g_stats.avgShots        = (g_stats.avgShots        * (n-1) + shots)        / n;
    g_stats.avgDemos        = (g_stats.avgDemos        * (n-1) + demos)        / n;

    report.AddAvg("Bot/GoalsFor",       g_stats.avgGoalsFor);
    report.AddAvg("Bot/GoalsAgainst",   g_stats.avgGoalsAgainst);
    report.AddAvg("Bot/Touches",        g_stats.avgTouches);
    report.AddAvg("Bot/Saves",          g_stats.avgSaves);
    report.AddAvg("Bot/Shots",          g_stats.avgShots);
    report.AddAvg("Bot/Demos",          g_stats.avgDemos);
    report.AddAvg("Train/TotalSteps",  (float)g_stats.totalSteps);
    report.AddAvg("Train/PhaseSteps",   (float)g_stats.phaseSteps);
    report.AddAvg("Train/Phase",        (float)g_stats.currentPhase);
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    // Initialize RocketSim with collision_meshes path
    // Default "collision_meshes" works if running from Talos V2 dir
    // Override with --mesh-path <path> if needed
    std::string meshPath = "collision_meshes";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--mesh-path" && i + 1 < argc) {
            meshPath = argv[i + 1];
            break;
        }
    }
    RocketSim::Init(meshPath.c_str());

    // Load checkpoint to detect starting phase
    int64_t loadedSteps = LoadCheckpointSteps();
    if (loadedSteps > 0) {
        g_stats.totalSteps = loadedSteps;
        g_stats.currentPhase = DetectPhaseFromSteps(loadedSteps);
        printf("\n>>> Loaded checkpoint with %lld steps, starting at Phase %d\n\n",
               loadedSteps, g_stats.currentPhase);
    } else {
        g_stats.currentPhase = 1;
        g_stats.totalSteps = 0;
    }

    g_stats.phaseStartStep = g_stats.totalSteps;

    PhaseConfig& phase = PHASES[g_stats.currentPhase];

    printf("\n");
    printf("============================================================\n");
    printf("  TALON V2 - 10-Phase Progressive Training (GigaLearnCPP)\n");
    printf("============================================================\n");
    printf("  Phase:       %s\n",       phase.name);
    printf("  Goal:        %s\n",       phase.description);
    printf("  Target:      %lld steps\n", phase.endStep - phase.startStep);
    printf("  Gamma:       %.3f\n",     phase.gamma);
    printf("  Entropy:     %.2f\n",     phase.entropy);
    printf("  Network:     1024x4 LEAKY_RELU + LayerNorm\n");
    printf("  Mode:        1v1 only | Octane hitbox\n");
    printf("============================================================\n\n");

    LearnerConfig cfg = {};

    // GPU CUDA Training - P100 has CUDA, use GPU_CUDA explicitly for Kaggle
    // Use --cpu flag to force CPU mode explicitly
    bool forceCPU = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--cpu") {
            forceCPU = true;
            break;
        }
    }

    // Use GPU_CUDA for P100 (explicit for Kaggle)
    // If --cpu flag is passed, use CPU instead
    cfg.deviceType = forceCPU ? LearnerDeviceType::CPU : LearnerDeviceType::GPU_CUDA;
    cfg.numGames    = 64;
    cfg.tickSkip    = 8;
    cfg.actionDelay = 7;

    // PPO Configuration (as specified)
    cfg.ppo.tsPerItr      = 200000;
    cfg.ppo.batchSize     = 200000;
    cfg.ppo.miniBatchSize = 50000;
    cfg.ppo.epochs        = 3;
    cfg.ppo.gaeGamma      = phase.gamma;
    cfg.ppo.gaeLambda     = 0.95f;
    cfg.ppo.entropyScale  = phase.entropy;
    cfg.ppo.policyLR      = 3e-4f;
    cfg.ppo.criticLR      = 3e-4f;

    // Network: 1024x4 LEAKY_RELU + LayerNorm (as specified)
    cfg.ppo.sharedHead.layerSizes = { 1024, 1024, 1024, 1024 };
    cfg.ppo.policy.layerSizes     = { 1024, 1024, 1024, 1024 };
    cfg.ppo.critic.layerSizes     = { 1024, 1024, 1024, 1024 };

    cfg.ppo.sharedHead.activationType = ModelActivationType::LEAKY_RELU;
    cfg.ppo.policy.activationType      = ModelActivationType::LEAKY_RELU;
    cfg.ppo.critic.activationType      = ModelActivationType::LEAKY_RELU;

    cfg.ppo.sharedHead.addLayerNorm   = true;
    cfg.ppo.policy.addLayerNorm       = true;
    cfg.ppo.critic.addLayerNorm       = true;

    cfg.ppo.sharedHead.optimType      = ModelOptimType::ADAM;
    cfg.ppo.policy.optimType          = ModelOptimType::ADAM;
    cfg.ppo.critic.optimType          = ModelOptimType::ADAM;

    // Checkpoints (save more frequently for Kaggle 12hr sessions)
    cfg.checkpointFolder = "checkpoints";
    cfg.tsPerSave        = 5000000;   // 5M steps per save (more frequent for session limits)

    // Skill tracker off
    cfg.skillTracker.enabled = false;

    // Metrics off
    cfg.sendMetrics = false;

    // Random seed
    cfg.randomSeed = 456;

    Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);
    learner->Start();

    printf("\n");
    printf("============================================================\n");
    printf("  TRAINING COMPLETE\n");
    printf("============================================================\n");
    printf("  Total Steps:      %lld\n",    g_stats.totalSteps);
    printf("  Final Phase:      %d\n",      g_stats.currentPhase);
    printf("  Goals For:        %.2f/ep\n", g_stats.avgGoalsFor);
    printf("  Goals Against:    %.2f/ep\n", g_stats.avgGoalsAgainst);
    printf("  Touches:          %.2f/ep\n", g_stats.avgTouches);
    printf("  Saves:            %.2f/ep\n", g_stats.avgSaves);
    printf("  Demos:            %.2f/ep\n", g_stats.avgDemos);
    printf("============================================================\n\n");

    return EXIT_SUCCESS;
}