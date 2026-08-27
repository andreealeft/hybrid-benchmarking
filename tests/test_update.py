"""Delivering a change to somebody who is not us.

This is the only machinery in the library whose job is to run on a stranger's
machine, and it was the only machinery with no test.  What it got wrong is
worth stating plainly, because both halves looked like working code:

* it compared the **version number** in ``pyproject.toml``, so a push that did
  not bump that number reached nobody.  Six consecutive pushes did not bump it,
  five of them changed what a reader sees, and the updater reported success by
  saying nothing at all.
* it read that number through ``raw.githubusercontent.com``, which serves
  ``max-age=300``, so even a bumped version could be five minutes stale --
  long enough for an app installed minutes ago to miss what was already on
  ``main``.

Nothing here touches the network: the two questions the updater asks are
replaced, and what is tested is which answer makes it act.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hybrid_benchmarking import update as up


@pytest.fixture
def elsewhere(tmp_path, monkeypatch):
    """A stamp that is not the real one, so a test cannot disturb a real
    install and cannot be fooled by one."""
    monkeypatch.setattr(up, "data_dir", lambda: tmp_path)
    return tmp_path


def _answers(monkeypatch, commit=None, version=None):
    monkeypatch.setattr(up, "latest_commit", lambda: commit)
    monkeypatch.setattr(up, "latest", lambda: version)


class TestWhatMakesItAct:
    def test_a_new_commit_moves_it_even_though_the_version_is_unchanged(
            self, elsewhere, monkeypatch):
        """The bug itself.  Everything about this case says nothing to do,
        except the one thing that matters: the code is different."""
        up.write_stamp("a" * 40)
        _answers(monkeypatch, commit="b" * 40, version=up.installed())
        assert up.what_to_install() == "b" * 40

    def test_the_same_commit_is_left_alone(self, elsewhere, monkeypatch):
        up.write_stamp("a" * 40)
        _answers(monkeypatch, commit="a" * 40)
        assert up.what_to_install() is None

    def test_a_copy_that_has_never_been_stamped_updates_once(
            self, elsewhere, monkeypatch):
        """Every install made before the stamp existed is in this state, and
        each has to right itself without anybody being told to reinstall."""
        _answers(monkeypatch, commit="c" * 40)
        assert up.what_to_install() == "c" * 40
        up.write_stamp("c" * 40)
        assert up.what_to_install() is None

    def test_it_falls_back_to_the_version_when_the_commit_cannot_be_had(
            self, elsewhere, monkeypatch):
        """An outage should cost the improvement, not the mechanism."""
        _answers(monkeypatch, commit=None, version="999.0.0")
        assert up.what_to_install() == "999.0.0"

    def test_offline_it_does_nothing_rather_than_guessing(
            self, elsewhere, monkeypatch):
        _answers(monkeypatch, commit=None, version=None)
        assert up.what_to_install() is None

    def test_the_fallback_still_refuses_to_go_backwards(
            self, elsewhere, monkeypatch):
        _answers(monkeypatch, commit=None, version="0.0.1")
        assert up.what_to_install() is None


class TestTheStampSurvivesTheInstall:
    def test_it_is_not_kept_inside_the_package(self):
        """Installing over the package is exactly what would erase it, so a
        stamp that lived there would report 'never installed' every launch and
        the tool would reinstall itself forever."""
        package = Path(up.__file__).resolve().parent
        assert package not in up._stamp_file().resolve().parents

    def test_it_reads_back_what_was_written(self, elsewhere):
        assert up.stamp() is None
        up.write_stamp("d" * 40)
        assert up.stamp() == "d" * 40

    def test_an_unwritable_home_costs_a_redundant_update_not_a_crash(
            self, monkeypatch, tmp_path):
        """Being unable to remember is survivable; failing to open is not."""
        monkeypatch.setattr(up, "data_dir", lambda: tmp_path / "no" / "such")
        monkeypatch.setattr(Path, "mkdir", _refuse)
        up.write_stamp("e" * 40)
        assert up.stamp() is None


def _refuse(*args, **kwargs):
    raise OSError("read-only")


class TestTheQuestionItAsks:
    def test_it_reads_the_head_commit_out_of_the_feed(self, monkeypatch):
        """The shape GitHub actually serves, first entry first."""
        feed = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<feed xmlns="http://www.w3.org/2005/Atom">\n'
            "  <id>tag:github.com,2008:/o/r/commits/main</id>\n"
            "  <entry>\n"
            "    <id>tag:github.com,2008:Grit::Commit/"
            "2b2cfb2b2a15df86a62bae1f640ed8ff43bd1055</id>\n"
            "  </entry>\n"
            "  <entry>\n"
            "    <id>tag:github.com,2008:Grit::Commit/"
            "ed90ebb0000000000000000000000000000000ff</id>\n"
            "  </entry>\n"
            "</feed>\n"
        )
        monkeypatch.setattr(up, "_fetch", lambda url, limit=4096: feed)
        assert up.latest_commit() == "2b2cfb2b2a15df86a62bae1f640ed8ff43bd1055"

    def test_the_source_it_asks_is_one_that_is_not_served_stale(self):
        """``raw.githubusercontent.com`` caches for five minutes, which is the
        second half of the bug.  The commit feed does not."""
        assert "commits/main.atom" in up.HEAD
        assert "raw.githubusercontent.com" not in up.HEAD

    def test_it_asks_one_question_when_there_is_nothing_to_do(
            self, elsewhere, monkeypatch):
        """The ordinary launch is the one where nothing needs installing, and
        it is the one that must stay bounded."""
        asked = []
        monkeypatch.setattr(up, "_fetch",
                            lambda url, limit=4096: asked.append(url) or None)
        up.write_stamp("f" * 40)
        monkeypatch.setattr(up, "latest_commit", lambda: "f" * 40)
        up.what_to_install()
        assert asked == []


class TestItInstallsWhatItDecidedOn:
    """The third bug in this chain, and the one that would have hidden the
    fix for the other two.

    pip keeps an HTTP cache keyed on the URL, and the branch archive's URL
    never changes.  So pip answered "Using cached", installed a zip from the
    first time it had ever run, and reported success -- after which the stamp
    was written and the copy never asked again.  Every visible sign said the
    update had worked and nothing on disk had moved.
    """

    @staticmethod
    def _calls_of_an_update(monkeypatch, target, codes=(0, 0)):
        calls = []
        codes = list(codes)

        def watch(argv, **kwargs):
            calls.append(argv)

            class Done:
                returncode = codes[len(calls) - 1] if len(calls) <= len(codes) else 0

            return Done()

        monkeypatch.setattr(up, "from_a_checkout", lambda: False)
        monkeypatch.setattr(up, "what_to_install", lambda: target)
        monkeypatch.setattr(up, "latest", lambda: "0.0.1")
        monkeypatch.setattr(up.subprocess, "run", watch)
        up.check_and_update()
        return calls

    def _argv_of_an_update(self, monkeypatch, target):
        return self._calls_of_an_update(monkeypatch, target)[-1]

    def test_the_package_files_are_forced_not_merely_upgraded(
            self, elsewhere, monkeypatch):
        """``--upgrade`` against a URL does not reliably replace a package
        whose version has not moved: in one environment here it did and in
        another it did not, and the one where it did not exited zero and left
        the old files in place.  Every version this ships from now on has the
        same version number as the last, so that is the ordinary case."""
        calls = self._calls_of_an_update(monkeypatch, "b" * 40)
        assert len(calls) == 2, calls
        assert "--force-reinstall" in calls[-1], calls[-1]

    def test_the_first_pass_still_resolves_dependencies(
            self, elsewhere, monkeypatch):
        """The forcing pass carries ``--no-deps``, so on its own a release
        that adds a dependency would install cleanly and fail to import."""
        calls = self._calls_of_an_update(monkeypatch, "b" * 40)
        assert "--no-deps" not in calls[0], calls[0]
        assert "--no-deps" in calls[-1], calls[-1]

    def test_a_failure_in_either_pass_stops_it(
            self, elsewhere, monkeypatch):
        for codes in ((1, 0), (0, 1)):
            up._stamp_file().unlink(missing_ok=True)
            self._calls_of_an_update(monkeypatch, "b" * 40, codes=codes)
            assert up.stamp() is None, codes

    def test_the_archive_it_fetches_names_the_commit(
            self, elsewhere, monkeypatch):
        commit = "b" * 40
        argv = self._argv_of_an_update(monkeypatch, commit)
        assert any(part.endswith("/{}.zip".format(commit)) for part in argv), argv
        assert not any("refs/heads/main" in part for part in argv), argv

    def test_it_records_the_commit_it_actually_asked_for(
            self, elsewhere, monkeypatch):
        commit = "c" * 40
        self._argv_of_an_update(monkeypatch, commit)
        assert up.stamp() == commit

    def test_the_branch_fallback_refuses_the_cache(
            self, elsewhere, monkeypatch):
        """No commit is known here, so the URL is the branch one and cannot be
        made unique.  Then the cache has to be defeated rather than avoided."""
        argv = self._argv_of_an_update(monkeypatch, "0.9.9")
        assert "--no-cache-dir" in argv, argv
        assert any("refs/heads/main" in part for part in argv), argv

    def test_a_failed_install_is_not_recorded_as_done(
            self, elsewhere, monkeypatch):
        """Otherwise one failure is permanent: the stamp would say the commit
        had arrived and nothing would ever ask again."""
        class Failed:
            returncode = 1

        monkeypatch.setattr(up, "from_a_checkout", lambda: False)
        monkeypatch.setattr(up, "what_to_install", lambda: "d" * 40)
        monkeypatch.setattr(up, "latest", lambda: "0.0.1")
        monkeypatch.setattr(up.subprocess, "run", lambda *a, **k: Failed())
        assert up.check_and_update() is None
        assert up.stamp() is None


class TestWhatItRefusesToTouch:
    def test_a_working_copy_is_never_installed_over(self, monkeypatch):
        """Somebody developing the library has it installed from a source
        tree, and a release written over that would destroy work."""
        monkeypatch.setattr(up, "from_a_checkout", lambda: True)
        monkeypatch.setattr(up, "what_to_install", lambda: "b" * 40)
        assert up.check_and_update() is None
