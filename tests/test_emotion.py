from bionic_mind.core.emotion import EmotionSystem, EmotionState, EmotionMode


class TestEmotionState:
    def test_default_state(self):
        state = EmotionState()
        assert state.valence == 0.0
        assert state.arousal == 0.3
        assert state.mode == EmotionMode.NEUTRAL

    def test_focused_mode(self):
        state = EmotionState(valence=-0.5, arousal=0.8)
        assert state.mode == EmotionMode.FOCUSED

    def test_exploratory_mode(self):
        state = EmotionState(valence=0.5, arousal=0.8)
        assert state.mode == EmotionMode.EXPLORATORY

    def test_satisfied_mode(self):
        state = EmotionState(valence=0.5, arousal=0.2)
        assert state.mode == EmotionMode.SATISFIED

    def test_depressed_mode(self):
        state = EmotionState(valence=-0.5, arousal=0.2)
        assert state.mode == EmotionMode.DEPRESSED

    def test_describe(self):
        state = EmotionState(valence=0.5, arousal=0.8)
        desc = state.describe()
        assert "兴奋" in desc or "好奇" in desc

    def test_to_dict(self):
        state = EmotionState(valence=0.3, arousal=0.6)
        d = state.to_dict()
        assert d["valence"] == 0.3
        assert d["arousal"] == 0.6


class TestEmotionSystem:
    def test_evaluate_default(self):
        system = EmotionSystem()
        state = system.evaluate()
        assert -1 <= state.valence <= 1
        assert 0 <= state.arousal <= 1

    def test_evaluate_positive_feedback(self):
        system = EmotionSystem()
        state = system.evaluate(social_feedback=0.8, goal_progress=0.7)
        assert state.valence > 0

    def test_evaluate_negative_feedback(self):
        system = EmotionSystem()
        state = system.evaluate(prediction_error=0.9, consistency=0.1)
        assert state.valence < 0

    def test_evaluate_high_arousal(self):
        system = EmotionSystem(smoothing_factor=1.0)
        state = system.evaluate(prediction_error=0.9, novelty=0.8)
        assert state.arousal > 0.5

    def test_smoothing(self):
        system = EmotionSystem(smoothing_factor=0.3)
        system.evaluate(social_feedback=0.9, goal_progress=0.9)
        v1 = system.state.valence
        system.evaluate(prediction_error=0.9, consistency=0.1)
        v2 = system.state.valence
        assert v2 < v1

    def test_get_modulation_focused(self):
        system = EmotionSystem(smoothing_factor=1.0)
        system.state = EmotionState(valence=-0.5, arousal=0.8)
        mod = system.get_modulation()
        assert mod.temperature_offset < 0
        assert mod.search_breadth == "narrow"

    def test_get_modulation_exploratory(self):
        system = EmotionSystem(smoothing_factor=1.0)
        system.state = EmotionState(valence=0.5, arousal=0.8)
        mod = system.get_modulation()
        assert mod.temperature_offset > 0
        assert mod.search_breadth == "wide"

    def test_history_tracking(self):
        system = EmotionSystem()
        for _ in range(10):
            system.evaluate(social_feedback=0.5)
        assert len(system.valence_history) == 10
        assert len(system.arousal_history) == 10

    def test_trend(self):
        system = EmotionSystem()
        for _ in range(5):
            system.evaluate(social_feedback=0.8, goal_progress=0.8)
        trend = system.get_trend()
        assert "valence_trend" in trend
        assert "arousal_trend" in trend

    def test_reset(self):
        system = EmotionSystem()
        system.evaluate(social_feedback=0.9)
        system.reset()
        assert system.state.valence == 0.0
        assert len(system.valence_history) == 0
