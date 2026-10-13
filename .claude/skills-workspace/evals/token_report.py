import json, sys, os
F = ('inputTokens', 'cacheCreationInputTokens', 'cacheReadInputTokens', 'outputTokens', 'thinkingTokens', 'costUSD')
def rows(run, n):
    p = os.path.join(run, 'agents', n + '.jsonl')
    if not os.path.exists(p):
        return []
    out, prev = [], {f: 0 for f in F}
    for line in open(p):
        m = json.loads(line)['meta']
        u = list(m.get('modelUsage', {}).values())[0] if m.get('modelUsage') else {}
        cur = {f: u.get(f, 0) for f in F}
        d = {f: cur[f] - prev[f] for f in F}
        d['requests'] = m.get('num_turns', 0)
        out.append(d); prev = cur
    return out
mode = sys.argv[1]
if mode == 'summary':
    print('| Run | Turns | Calls w/s/h | Model requests | New input | Cache write | Cache read | Output (thinking) | Total tokens | Cost |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for run in sys.argv[2:]:
        tot = {f: 0 for f in F}; calls = []; req = 0
        for n in ('worker', 'system', 'human'):
            r = rows(run, n); calls.append(len(r)); req += sum(x['requests'] for x in r)
            for x in r:
                for f in F: tot[f] += x[f]
        t = sum(tot[f] for f in F[:4]); turns = json.load(open(os.path.join(run, 'run.json')))['turns']
        print('| %s | %d | %s | %d | %.1fk | %.1fk | %.1fk | %.1fk (%.1fk) | %.0fk | $%.4f |' % (
            run.split('-')[-1], turns, '/'.join(map(str, calls)), req, tot['inputTokens']/1e3, tot['cacheCreationInputTokens']/1e3,
            tot['cacheReadInputTokens']/1e3, tot['outputTokens']/1e3, tot['thinkingTokens']/1e3, t/1e3, tot['costUSD']))
else:
    run = sys.argv[2]
    for n in ('worker', 'system', 'human'):
        print('\n%s' % n)
        print('| Call | Requests | New input | Cache write | Cache read | Output | Thinking | Cost |')
        print('|---|---|---|---|---|---|---|---|')
        for i, x in enumerate(rows(run, n), 1):
            print('| %d | %d | %d | %d | %d | %d | %d | $%.4f |' % (i, x['requests'], x['inputTokens'], x['cacheCreationInputTokens'],
                  x['cacheReadInputTokens'], x['outputTokens'], x['thinkingTokens'], x['costUSD']))
