from src.preprocessing.keyframe_select import select_keyframes


def test_select_keyframes_returns_input_paths(tmp_path):
    frames = [tmp_path / "0001.jpg", tmp_path / "0002.jpg"]
    result = select_keyframes(frames, tmp_path / "keyframes")
    assert result == frames
