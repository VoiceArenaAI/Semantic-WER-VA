import pytest

from va_semantic_wer import SemanticWER, available_languages, normalize, segments, substitution_prompt
from va_semantic_wer.backends import Backend, parse_verdict
from va_semantic_wer.prompting import deletion_template, insertion_template, mark_deletion, mark_insertion, substitution_template


def test_normalize_keeps_spoken_symbols():
    assert normalize("Okay, so I'm going!") == "okay so i'm going"
    assert normalize("It costs $5 at 3:30.") == "it costs $5 at 3:30"
    assert normalize("150.000 đồng, 1.5 km") == "150.000 đồng 1.5 km"
    assert normalize("Guarda-chuva") == "guarda chuva"
    assert normalize("50%") == "50%"
    assert normalize("50 %") == "50 %"
    assert normalize("٥٠٪ و 3٫5") == "٥٠٪ و 3٫5"
    assert normalize("10‰ at 20°") == "10‰ at 20°"
    assert normalize("₺100 or ₩500") == "₺100 or ₩500"


def test_normalize_removes_only_punctuation_and_tags():
    assert normalize("♪ la la ♪") == "♪ la la ♪"
    assert normalize("great 😀 job") == "great 😀 job"
    assert normalize("**bold** #tag") == "bold tag"
    assert normalize("मैं घर गया। फिर?") == "मैं घर गया फिर"
    assert normalize("Tôi đã đến, nhé!") == "tôi đã đến nhé"
    assert normalize("rock ’n’ roll") == "rock 'n' roll"
    assert normalize("It’s ' quoted 'cause") == "it's quoted 'cause"
    assert normalize("[laughing] Ha ha. Oh, okay") == "ha ha oh okay"
    assert normalize("O que [com um tom de tristeza] foi") == "o que foi"
    assert normalize("a de romance [incompleto") == "a de romance"
    assert normalize("O que hÃ¡ de Ãºnico") == "o que há de único"
    assert normalize("<|speaker:0|> Hello there <|speaker:1|> hi") == "hello there hi"
    assert normalize("<|speaker:0> Torna melhor?") == "torna melhor"


def test_segments_merge_adjacent_edits():
    ss = segments("a thimble full of water", "a thimbleful of water")
    subs = [s for s in ss if s.kind != "equal"]
    assert len(subs) == 1 and subs[0].kind == "substitution"
    assert subs[0].ref == ("thimble", "full") and subs[0].hyp == ("thimbleful",)


def test_segments_insertion_deletion_and_empty():
    assert [s.kind for s in segments("i think", "i i think") if s.kind != "equal"] == ["insertion"]
    assert [s.kind for s in segments("you know we went", "we went") if s.kind != "equal"] == ["deletion"]
    assert segments("", "") == []
    assert segments("", "hello")[0].kind == "insertion"


@pytest.mark.parametrize("lang", ["en", "pt", "ro", "vi"])
def test_prompts_render(lang):
    assert lang in available_languages()
    t = substitution_template(lang)
    assert "{" not in t.replace("{ref}", "").replace("{pred}", "")
    for i in range(1, 9):
        assert f"\n{i}. " in t
    p = substitution_prompt(lang, "x y", "z")
    assert p.rstrip().endswith('Reference: x y\nPrediction: z')


def test_vietnamese_tone_note():
    assert "different tone usually makes a different word" in substitution_template("vi")
    assert "different tone" not in substitution_template("en")


def test_parse_verdict():
    assert parse_verdict("True") is True and parse_verdict(" false.\n") is False and parse_verdict("maybe") is None


class FakeJudge(Backend):
    model = "fake-judge"

    def __init__(self, equal_pairs=(), disfluent=()):
        self.equal_pairs, self.disfluent, self.calls = set(equal_pairs), set(disfluent), []

    def judge(self, prompts):
        self.calls += prompts
        out = []
        for p in prompts:
            last = p.rstrip().rsplit("\n", 2)
            if last[-1].startswith("Prediction: "):
                out.append((last[-2].removeprefix("Reference: "), last[-1].removeprefix("Prediction: ")) in self.equal_pairs)
            else:
                out.append(last[-1].removeprefix("Reference: ") in self.disfluent)
        return out


def test_scorer_forgives_only_judged_equal(tmp_path):
    judge = FakeJudge({("thimble full", "thimbleful")})
    s = SemanticWER("en", backend=judge, cache_path=tmp_path / "c.jsonl")
    res = s.score(["a thimble full of water", "the cat sat"], ["a thimbleful of water", "a cat sat"])
    assert res.wer == pytest.approx(3 / 8)
    assert res.semantic_wer == pytest.approx(1 / 8)
    assert res.unique_substitutions == 2 and res.substitutions_equal == 1 and res.newly_judged == 2
    n = len(judge.calls)
    res2 = SemanticWER("en", backend=judge, cache_path=tmp_path / "c.jsonl").score(
        ["a thimble full of water"], ["a thimbleful of water"])
    assert len(judge.calls) == n and res2.from_cache == 1 and res2.semantic_wer == 0


def test_unanswered_counts_as_error(tmp_path):
    class Silent(FakeJudge):
        def judge(self, prompts):
            return [None] * len(prompts)
    res = SemanticWER("en", backend=Silent(set()), cache_path=tmp_path / "c.jsonl").score(["the cat"], ["a cat"])
    assert res.semantic_wer == res.wer == 0.5 and res.unanswered == 1


def test_cache_is_jsonl(tmp_path):
    import json
    judge = FakeJudge({("colour", "color")})
    SemanticWER("en", backend=judge, cache_path=tmp_path / "v.jsonl").score(["the colour"], ["the color"])
    rows = [json.loads(l) for l in (tmp_path / "v.jsonl").read_text().splitlines()]
    assert rows == [{"task": "substitution", "language": "en", "model": "fake-judge",
                     "ref": "colour", "hyp": "color", "equal": True}]


@pytest.mark.parametrize("lang", ["en", "pt", "ro", "vi"])
def test_insertion_and_deletion_prompts_render(lang):
    ins, dele = insertion_template(lang), deletion_template(lang)
    for t in (ins, dele):
        assert "{" not in t.replace("{sentence}", "")
        for i in range(1, 6):
            assert f"\n{i}. " in t
    assert "⟨+" in ins and "⟨−" not in ins
    assert "⟨−" in dele and "⟨+" not in dele


def test_marking():
    assert mark_insertion(["i", "think", "so"], 1, ["i"]) == "i ⟨+ i ⟩ think so"
    assert mark_deletion(["you", "know", "we", "went"], 0, 2) == "⟨− you know ⟩ we went"


def test_insertions_and_deletions_forgiven_only_when_disfluent(tmp_path):
    judge = FakeJudge(disfluent={"i think ⟨+ uh ⟩ we should go", "⟨− mhm ⟩ that makes sense"})
    s = SemanticWER("en", backend=judge, cache_path=tmp_path / "c.jsonl")
    res = s.score(["i think we should go", "mhm that makes sense", "it was really good"],
                  ["i think uh we should go", "that makes sense", "it was like really good"])
    assert res.wer == pytest.approx(3 / 13)
    assert res.semantic_wer == pytest.approx(1 / 13)       # only the inserted "like" still counts
    assert res.unique_insertions_deletions == 3 and res.insertions_deletions_disfluent == 2
    assert [u.corrected for u in res.utterances] == ["i think we should go", "mhm that makes sense", "it was like really good"]
