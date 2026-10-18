from html.parser import HTMLParser

import numpy as np

import ui


class Balance(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.ok = [], True

    def handle_starttag(self, tag, attrs):
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack.pop() != tag:
            self.ok = False


def balanced(s):
    b = Balance()
    b.feed(s)
    return b.ok and not b.stack


CL = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
P = np.array([0.05, 0.02, 0.03, 0.1, 0.1, 0.6, 0.1])
PARTS = [ui.hero("T", "s", ["a"]), ui.kpi_row([("L", "v", "s")]), ui.panel("P", "<b>x</b>", "sub"), ui.pill("ok", "x"),
         ui.emotion_card("sad", 0.6, ("happy", 0.1)), ui.emotion_card("sad", 0.3, None, low=True), ui.prob_bars(P, CL),
         ui.timeline_strip([1.0, 1.5], ["sad", "angry"]), ui.steps([("a", "b"), ("c", "d")])]


def test_components_single_line_and_balanced():
    for part in PARTS:
        assert "\n" not in part and balanced(part), part[:70]


def test_html_escaped_and_bars_sorted():
    assert "<script>" not in ui.hero("<script>x</script>", "<b>", ["<i>"])
    bars = ui.prob_bars(P, CL)
    assert bars.index("sad") < bars.index("angry")                 # sabse zyada probability upar
    assert "60%" in bars and "width:60.0%" in bars


def test_low_confidence_warning_shows():
    assert "Low confidence" in ui.emotion_card("sad", 0.3, None, low=True) and "Low confidence" not in ui.emotion_card("sad", 0.9)
