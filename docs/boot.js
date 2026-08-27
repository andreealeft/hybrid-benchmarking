/* Run the whole tool inside the browser tab.

   The interface talks to a handful of /api addresses.  On a machine where the
   tool is installed those are served by a small local server; here there is no
   server, so this file starts Python inside the tab and answers the same
   addresses from the same functions.  The page itself is unchanged: it still
   calls fetch, and fetch is what has been replaced.

   Nothing about the privacy story changes, and one thing about the offline
   story does.  Your data still never leaves the machine: the numbers are
   computed in this tab, and no instance, no log and no answer is sent
   anywhere.  But the first visit does download Python and its libraries from a
   public mirror, which the installed version never does, so this build works
   offline only after that first load. */

(function () {
  "use strict";

  var PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.6/full/";

  /* Written in by pages/build.sh from the wheel it has just built, because the
     name carries the version and the version moves.  Hardcoded, it went stale
     the first time the version was bumped and the whole build would have
     answered 404 to its own package: a name that has to track a build is not a
     constant, and this is the second place in this repository where treating
     one as a constant broke delivery silently. */
  var WHEEL = "hybrid_benchmarking-0.2.3-py3-none-any.whl";

  var realFetch = window.fetch.bind(window);
  var starting = null;

  function say(text, done) {
    var box = document.getElementById("booting");
    if (!box) {
      box = document.createElement("div");
      box.id = "booting";
      box.setAttribute("style", "position:fixed;left:0;right:0;bottom:0;" +
        "z-index:99;padding:.5rem .9rem;font:13px/1.5 ui-sans-serif,system-ui," +
        "sans-serif;background:var(--accent);color:var(--ground);" +
        "text-align:center");
      document.body.appendChild(box);
    }
    box.textContent = text;
    if (done) { setTimeout(function () { box.remove(); }, 1500); }
  }

  async function start() {
    say("Starting Python in this tab. The first visit takes a moment.");
    var script = document.createElement("script");
    script.src = PYODIDE + "pyodide.js";
    await new Promise(function (ok, fail) {
      script.onload = ok; script.onerror = fail; document.head.appendChild(script);
    });

    var py = await loadPyodide({ indexURL: PYODIDE });
    say("Loading the mathematics libraries.");
    await py.loadPackage(["numpy", "sympy", "micropip"]);

    say("Installing the tool.");
    var micropip = py.pyimport("micropip");
    await micropip.install(new URL(WHEEL, window.location.href).href);

    var dispatcher = await (await realFetch("api.py")).text();
    py.FS.mkdirTree("/tool");
    py.FS.writeFile("/tool/api.py", dispatcher);
    py.FS.mkdirTree("/uploads");
    py.runPython("import sys; sys.path.insert(0, '/tool'); import api");

    say("Ready.", true);
    return py;
  }

  /* Every /api call goes to Python instead of to a server.  Anything else,
     the page's own files, still goes to the network as it did. */
  window.fetch = function (url, options) {
    var path = String(url && url.url ? url.url : url);
    if (path.indexOf("/api/") !== 0) { return realFetch(url, options); }
    if (!starting) { starting = start(); }

    return starting.then(function (py) {
      var method = (options && options.method) || "GET";
      var body = (options && options.body) || "";
      py.globals.set("REQ_PATH", path);
      py.globals.set("REQ_METHOD", method);
      py.globals.set("REQ_BODY", typeof body === "string" ? body : "");
      var answer = py.runPython("api.handle(REQ_PATH, REQ_METHOD, REQ_BODY)");
      return new Response(answer,
        { status: 200, headers: { "Content-Type": "application/json" } });
    }).catch(function (error) {
      return new Response(JSON.stringify(
        { error: "the tool could not start in this browser: " + error }),
        { status: 200, headers: { "Content-Type": "application/json" } });
    });
  };

  /* A browser tab has no paths, so the two fields that ask for one get a file
     picker beside them.  The file is copied into Python's own filesystem and
     the field is filled with the name it was given there, after which the
     ordinary route reads it exactly as it reads a file on a disk. */
  function attach(field) {
    if (!field || field.dataset.picker) { return; }
    field.dataset.picker = "yes";
    field.placeholder = "choose a file";
    field.readOnly = true;

    var picker = document.createElement("input");
    picker.type = "file";
    picker.style.display = "none";
    picker.onchange = async function () {
      var file = picker.files[0];
      if (!file) { return; }
      var py = await (starting || (starting = start()));
      var bytes = new Uint8Array(await file.arrayBuffer());
      var where = "/uploads/" + file.name.replace(/[^\w.\-]/g, "_");
      py.FS.writeFile(where, bytes);
      field.value = where;
      field.dispatchEvent(new Event("change", { bubbles: true }));
    };

    var button = document.createElement("button");
    button.type = "button";
    button.className = "ghost";
    button.textContent = "Choose a file";
    button.onclick = function () { picker.click(); };

    field.after(picker, button);
    field.onclick = function () { picker.click(); };
  }

  new MutationObserver(function () {
    attach(document.getElementById("genpath"));
    attach(document.getElementById("logpath"));
  }).observe(document.documentElement, { childList: true, subtree: true });
})();
