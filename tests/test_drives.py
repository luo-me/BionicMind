from bionic_mind.core.drives import DriveSystem, DriveState


class TestDriveState:
    def test_default(self):
        state = DriveState()
        assert state.curiosity == 0.5
        assert state.consistency == 0.5

    def test_dominant(self):
        state = DriveState(curiosity=0.9, consistency=0.2)
        name, value = state.dominant()
        assert name == "curiosity"
        assert value == 0.9

    def test_to_dict(self):
        state = DriveState()
        d = state.to_dict()
        assert "curiosity" in d
        assert "consistency" in d
        assert len(d) == 5


class TestDriveSystem:
    def test_update_default(self):
        system = DriveSystem()
        state = system.update()
        assert 0 <= state.curiosity <= 1

    def test_update_high_prediction_error(self):
        system = DriveSystem()
        state = system.update(prediction_error=0.9)
        assert state.curiosity > 0.5

    def test_update_low_consistency(self):
        system = DriveSystem()
        state = system.update(consistency=0.1)
        assert state.consistency > 0.5

    def test_social_feedback_resets_timer(self):
        system = DriveSystem()
        system.update(social_feedback=0.5)
        assert system.drives.social_connection < 0.5

    def test_should_act_spontaneously(self):
        system = DriveSystem(spontaneous_action_threshold=0.5)
        system.update(prediction_error=0.9, consistency=0.1)
        assert system.should_act_spontaneously() or system.drives.dominant()[1] < 0.5

    def test_get_spontaneous_prompt(self):
        system = DriveSystem(spontaneous_action_threshold=0.3)
        system.update(prediction_error=0.9)
        prompt = system.get_spontaneous_prompt()
        if system.should_act_spontaneously():
            assert prompt is not None
            assert "好奇心" in prompt or "驱力" in prompt

    def test_drive_gradient(self):
        system = DriveSystem()
        system.update(prediction_error=0.5)
        system.update(prediction_error=0.9)
        gradient = system.get_drive_gradient()
        assert "curiosity" in gradient

    def test_decay_curiosity(self):
        system = DriveSystem()
        system.drives.curiosity = 0.8
        system.decay_curiosity(0.1)
        assert abs(system.drives.curiosity - 0.7) < 0.01

    def test_reset(self):
        system = DriveSystem()
        system.update(prediction_error=0.9)
        system.reset()
        assert system.drives.curiosity == 0.5
        assert len(system._drive_history) == 0

    def test_to_dict(self):
        system = DriveSystem()
        system.update()
        d = system.to_dict()
        assert "drives" in d
        assert "dominant" in d
        assert "should_act_spontaneously" in d
