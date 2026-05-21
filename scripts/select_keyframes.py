"""Select keyframes from extracted frames."""

from src.preprocessing.keyframe_select import select_keyframes


if __name__ == "__main__":
    select_keyframes([], "data/keyframes")
