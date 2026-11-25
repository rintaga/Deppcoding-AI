import pytest
from agent.agent import Agent, make_plan, parse_tool_call
from agent.tools import SecurityError, Workspace, check_python_file, diagnose


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "w", log_path=tmp_path / "log.jsonl", cmd_timeout=5)


def test_path_escape_blocked(ws, tmp_path):
    (tmp_path / "secret.txt").write_text("x")
    for bad in ["../secret.txt", "/etc/passwd", "a/../../secret.txt"]:
        with pytest.raises(SecurityError):
            ws.read_file(bad)
    (ws.root / "link").symlink_to(tmp_path)
    with pytest.raises(SecurityError):
        ws.read_file("link/secret.txt")


@pytest.mark.parametrize("cmd", ["rm -rf x", "ls; id", "cat /etc/passwd", "ls ../", "python -c 'print(1)'", "python",
                                 "pip install x", "git -c core.x=y log", "git push", "curl http://x", "ls $(id)", "cat a | sh",
                                 "git diff --output=x", "ls -z", "python -m http.server"])
def test_dangerous_commands_rejected(ws, cmd):
    with pytest.raises(SecurityError):
        ws.validate_command(cmd)


def test_allowed_commands_work(ws):
    ws.write_file("a.py", "print('halo')")
    assert ws.run_command("python a.py")["stdout"].strip() == "halo"
    assert "a.py" in ws.run_command("ls")["stdout"]
    assert ws.run_command("grep -n halo a.py")["ok"]


def test_run_python_limits(ws):
    assert ws.run_python("print(2+3)")["stdout"].strip() == "5"
    assert ws.run_python("while True: pass", timeout=2)["timed_out"]
    for bad in ["import subprocess", "import os\nos.system('id')", "eval('1')", "from shutil import rmtree"]:
        with pytest.raises(SecurityError):
            ws.run_python(bad)


def test_diagnose_and_check(ws):
    assert diagnose("ModuleNotFoundError: No module named 'torch'")["kind"] == "modul_tidak_ada"
    assert diagnose("torch.OutOfMemoryError: CUDA out of memory")["kind"] == "oom_gpu"
    ws.write_file("bad.py", "def f(:\n  pass\n")
    r = check_python_file(ws, "bad.py")
    assert not r["ok"] and r["context"]
    assert "SyntaxError" in ws.inspect_project()["python_syntax_errors"][0] or ws.inspect_project()["python_syntax_errors"]


def test_actions_need_approval_and_model_cannot_bypass(ws):
    calls = iter(['<tool>{"name":"write_file","args":{"rel":"x.txt","content":"hi"}}</tool>', "selesai"])
    denied = Agent(ws, lambda d: False, lambda p: next(calls))
    denied.ask("tulis x")
    assert not (ws.root / "x.txt").exists()
    calls2 = iter(['<tool>{"name":"run_command","args":{"command":"rm -rf /"}}</tool>', "ok"])
    asked = []
    a = Agent(ws, lambda d: asked.append(d) or True, lambda p: next(calls2))
    a.ask("hapus")
    assert asked == []  # perintah terlarang ditolak sebelum sempat minta persetujuan


def test_approved_write_and_read_tools_need_no_approval(ws):
    asked = []
    a = Agent(ws, lambda d: asked.append(d) or True)
    assert a.execute("write_file", {"rel": "ok.txt", "content": "1"})[0] and len(asked) == 1
    assert a.execute("read_file", {"rel": "ok.txt"}) == (True, "1") and len(asked) == 1


def test_parse_tool_call_validation():
    assert parse_tool_call("tanpa alat") is None
    assert parse_tool_call('<tool>{"name":"list_dir","args":{"rel":"."}}</tool>') == ("list_dir", {"rel": "."})
    for bad in ['<tool>{"name":"hapus_semua"}</tool>', '<tool>bukan json</tool>',
                '<tool>{"name":"read_file","args":{"rel":5}}</tool>', '<tool>{"name":"read_file","args":{"x":"y"}}</tool>']:
        with pytest.raises(ValueError):
            parse_tool_call(bad)


def test_plan_and_log(ws):
    assert len(make_plan("perbaiki bug import")) >= 5
    Agent(ws, lambda d: True).execute("list_dir", {})
    assert (ws.log_path).read_text().count("\n") >= 1
