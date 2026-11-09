import json

import hotpot_eval


def test_eval_runs_after_json_fix(tmp_path, capsys):
    pred = {"answer": {"1": "Paris"}, "sp": {"1": []}}
    gold = [{"_id": "1", "answer": "Paris", "supporting_facts": []}]
    (tmp_path / "p.json").write_text(json.dumps(pred))
    (tmp_path / "g.json").write_text(json.dumps(gold))
    hotpot_eval.eval(str(tmp_path / "p.json"), str(tmp_path / "g.json"))
    assert "'em': 1.0" in capsys.readouterr().out
