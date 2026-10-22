# Briefs for monitors

The tasks in `SKILL.md` start monitors with the skill
`monitoring-jupylet-work`, and name a brief of this file. Give the brief to
the monitor as it is.

## The pane brief

```
A person is about to show the browser pane of the Claude app, with the
globe icon in the upper right corner. Watch the pane, and report when it
is displayed.

1. Do "Start the clock".
2. Do "Wait N seconds", with N = 5. It returns the minutes since the
   clock started.
3. Check the pane. Call mcp__Claude_Browser__tabs_context. Its last line
   says "The Browser pane is currently displayed." or "The Browser pane
   is currently hidden."
4. Act on the result:
   - **If the pane is displayed,** do "Report" with: "The pane is
     displayed." Stop here.
   - **If the minutes from step 2 reached 5,** do "Report" with: "The
     pane is still hidden after 5 minutes." Stop here.
   - **If the minutes from step 2 went up,** do "Send a note" with, for
     example: "The pane is still hidden after 2 minutes."

   Then continue to step 2.
```

## The sign-in brief

```
A person is signing in to a Jupyter web page in the browser pane of the
Claude app, with a token that I gave them. Watch the sign-in, and report
how it ends.

1. Do "Start the clock".
2. Do "Wait N seconds", with N = 60. The wait ends early when the web
   page loads again, which happens on each sign-in try.
3. Check the sign-in. Call mcp__Claude_Browser__javascript_tool, with
   action "javascript_exec" and this text:
   ({status: (await fetch('/api/status', {credentials: 'same-origin'})).status,
     failed: document.body.innerText.includes('Invalid credentials'),
     loaded: performance.timeOrigin,
     minutes: Math.floor((Date.now() - sessionStorage['jupylet-monitor-start']) / 60000)})
   The check returns four values:
   - `status`: 200 when the web page is signed in, 403 when it is not.
   - `failed`: true when the web page says that the token was wrong.
   - `loaded`: the time when the web page last loaded.
   - `minutes`: the minutes since the clock started.
4. Act on the result:
   - **If `status` is 200,** do "Report" with: "The web page is signed
     in." Stop here.
   - **If `failed` is true, and `loaded` changed since your last note
     about a failed try,** do "Send a note" with: "The sign-in failed:
     the token did not match."
   - **If `minutes` reached 5,** do "Report" with: "The web page is
     still not signed in after 5 minutes." Stop here.
   - **If `minutes` went up,** do "Send a note" with, for example: "The
     web page is still not signed in after 2 minutes."

   Then continue to step 2.
```
