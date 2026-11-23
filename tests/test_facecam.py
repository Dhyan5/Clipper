from clipforge.facecam import expand_box, facecam_crop, gameplay_crop


def test_expand_box_within_frame():
    box = {"x": 10, "y": 20, "w": 200, "h": 100}
    out = expand_box(box, 1280, 720)
    assert out["x"] >= 0 and out["y"] >= 0
    assert out["x"] + out["w"] <= 1280
    assert out["y"] + out["h"] <= 720
    assert out["w"] % 2 == 0 and out["h"] % 2 == 0


def test_facecam_crop_none():
    assert facecam_crop(None, 1280, 720) is None


def test_gameplay_crop_avoids_facecam():
    box = {"x": 400, "y": 40, "w": 320, "h": 240}
    x, y, w, h = gameplay_crop(box, 1280, 720)
    assert x == 0 and w > 0 and h > 0
    assert y >= box["y"] + box["h"] - 4
    assert y + h <= 720


def test_gameplay_crop_without_facecam():
    assert gameplay_crop(None, 1280, 720) == (0, 0, 1280, 720)
