from bionic_mind.core.adaptive_emotion import AdaptiveEmotionLearner, EmotionWeights
from bionic_mind.core.meta_action import MetaActionSystem, MetaAction


class TestEmotionWeights:
    def test_default_weights(self):
        w = EmotionWeights()
        assert w.prediction_error == -0.30
        assert w.goal_progress == 0.20

    def test_to_list_and_from_list(self):
        w = EmotionWeights()
        values = w.to_list()
        assert len(values) == 9
        w2 = EmotionWeights.from_list(values)
        assert abs(w2.prediction_error - w.prediction_error) < 0.001


class TestAdaptiveEmotionLearner:
    def test_update_weights_positive_feedback(self):
        learner = AdaptiveEmotionLearner(learning_rate=0.1)
        inputs = {
            "prediction_error": 0.2,
            "goal_progress": 0.8,
            "consistency": 0.7,
            "social_feedback": 0.8,
            "novelty": 0.3,
        }
        result = learner.update_weights(inputs, valence_target=0.8, arousal_target=0.5)
        assert result["update_count"] == 1

    def test_update_weights_negative_feedback(self):
        learner = AdaptiveEmotionLearner(learning_rate=0.1)
        inputs = {
            "prediction_error": 0.8,
            "goal_progress": 0.1,
            "consistency": 0.2,
            "social_feedback": -0.5,
            "novelty": 0.3,
        }
        result = learner.update_weights(inputs, valence_target=-0.5, arousal_target=0.7)
        assert result["update_count"] == 1

    def test_multiple_updates(self):
        learner = AdaptiveEmotionLearner(learning_rate=0.05)
        for i in range(10):
            inputs = {
                "prediction_error": 0.3,
                "goal_progress": 0.6,
                "consistency": 0.7,
                "social_feedback": 0.5,
                "novelty": 0.4,
            }
            learner.update_weights(inputs, valence_target=0.5, arousal_target=0.4)
        assert learner.get_stats()["update_count"] == 10

    def test_get_stats(self):
        learner = AdaptiveEmotionLearner()
        stats = learner.get_stats()
        assert "update_count" in stats
        assert "weights" in stats


class TestMetaActionSystem:
    def test_evaluate_no_action_cooldown(self):
        system = MetaActionSystem(cooldown_cycles=10)
        actions = system.evaluate_and_propose(
            emotion_state={"valence": 0.0, "arousal": 0.3},
            drive_state={"curiosity": 0.5, "consistency": 0.5},
            performance_metrics={"avg_prediction_error": 0.5},
        )
        assert actions == []

    def test_evaluate_high_error(self):
        system = MetaActionSystem(cooldown_cycles=0)
        system._cycles_since_last_action = 15
        actions = system.evaluate_and_propose(
            emotion_state={"valence": 0.0, "arousal": 0.3},
            drive_state={"curiosity": 0.5, "consistency": 0.5},
            performance_metrics={"avg_prediction_error": 0.8},
        )
        assert len(actions) > 0
        assert any(a.action_type == "adjust_learning_rate" for a in actions)

    def test_apply_actions(self):
        system = MetaActionSystem()
        actions = [MetaAction(
            action_type="adjust_learning_rate",
            target="learning_rate",
            value=0.2,
            reason="test",
        )]
        applied = system.apply(actions)
        assert "learning_rate" in applied
        assert system.current_learning_rate == 0.2

    def test_get_state(self):
        system = MetaActionSystem()
        state = system.get_state()
        assert "learning_rate" in state
        assert "drive_weights" in state
        assert "emotion_sensitivity" in state
