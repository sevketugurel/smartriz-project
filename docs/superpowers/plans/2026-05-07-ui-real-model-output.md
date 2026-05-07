# SmarTRIZ UI — Wire to Real Model Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every mock/static value in the UI with real model output from the Ollama 7B TRIZ pipeline, and enrich the contradiction detector's output with TRIZ-39 parameter IDs so the matrix renders correctly.

**Architecture:** Backend `contradiction_detector` gets a richer JSON schema (adds `improving_id`/`worsening_id` 1-39 integers) while keeping `state["contradictions"]` as `List[str]` for backwards compat. Frontend derives agent log lines from model output in the SSE hook, and App.jsx pulls `contradiction_details` for matrix pair resolution. ReasoningChain and SolutionOutput switch from static mocks to live `result` data.

**Tech Stack:** Python 3.11, LangGraph, FastAPI SSE, React 19 + Vite, Tailwind utility classes

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `src/smartriz/agents/state.py` | Modify | Add `contradiction_details` field |
| `src/smartriz/agents/prompts.py` | Modify | Update `DETECTOR_SYSTEM` with structured schema + TRIZ-39 param list |
| `src/smartriz/agents/graph.py` | Modify | Parse new detector schema, populate `contradiction_details`, keep `contradictions` as `List[str]` |
| `ui/src/hooks/useTrizStream.js` | Modify | Add `partialResult` state; derive `logLine` from `updates` not `payload.log_line` |
| `ui/src/App.jsx` | Modify | Use `contradiction_details[0]` for pair; parse `selected_principles` for cards; heuristic confidence; remove `reasoningMock` import |
| `ui/src/screens/ReasoningChain.jsx` | Modify | Accept `state` prop, build dynamic 5-section markdown |
| `ui/src/screens/SolutionOutput.jsx` | Modify | Conditional cases section; dynamic `ConfidenceBar` values via props |

---

### Task 1: Add `contradiction_details` to TRIZState

**Files:**
- Modify: `src/smartriz/agents/state.py`

- [ ] **Step 1: Read current state file**

Current content:
```python
from typing import Dict, List, Optional, TypedDict

class TRIZState(TypedDict, total=False):
    original_problem: str
    analysis: Optional[str]
    contradictions: List[str]
    selected_principles: List[str]
    final_solution: Optional[str]
    critic_feedback: Optional[str]
    iterations: int
    meta: Optional[Dict[str, object]]
```

- [ ] **Step 2: Add the new field**

Replace the file content with:
```python
from typing import Any, Dict, List, Optional, TypedDict


class TRIZState(TypedDict, total=False):
    original_problem: str
    analysis: Optional[str]
    contradictions: List[str]
    contradiction_details: Optional[List[Dict[str, Any]]]
    selected_principles: List[str]
    final_solution: Optional[str]
    critic_feedback: Optional[str]
    iterations: int
    meta: Optional[Dict[str, object]]
```

- [ ] **Step 3: Verify no import errors**

```bash
cd /Users/sevketugurel/Desktop/LIFTUP/smartriz-project && .venv/bin/python -c "from smartriz.agents.state import TRIZState; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/smartriz/agents/state.py
git commit -m "feat(state): add contradiction_details field to TRIZState"
```

---

### Task 2: Update DETECTOR_SYSTEM Prompt with Structured Schema

**Files:**
- Modify: `src/smartriz/agents/prompts.py` (lines 17-26)

- [ ] **Step 1: Replace DETECTOR_SYSTEM with structured schema + TRIZ-39 list**

Replace the `DETECTOR_SYSTEM` constant (lines 17-26) with:
```python
DETECTOR_SYSTEM = """You are a TRIZ contradiction specialist. Given a problem analysis, identify the technical contradictions.

A technical contradiction has the form: "Improving X worsens Y".

TRIZ-39 Engineering Parameters (use ONLY these names and IDs):
1: Weight of moving object, 2: Weight of stationary object, 3: Length of moving object,
4: Length of stationary object, 5: Area of moving object, 6: Area of stationary object,
7: Volume of moving object, 8: Volume of stationary object, 9: Speed, 10: Force,
11: Stress or pressure, 12: Shape, 13: Stability of object composition, 14: Strength,
15: Duration of action of moving object, 16: Duration of action of stationary object,
17: Temperature, 18: Illumination intensity, 19: Use of energy by moving object,
20: Use of energy by stationary object, 21: Power, 22: Loss of energy, 23: Loss of substance,
24: Loss of information, 25: Loss of time, 26: Quantity of substance, 27: Reliability,
28: Measurement accuracy, 29: Manufacturing precision, 30: External harm affects the object,
31: Object-generated harmful factors, 32: Ease of manufacture, 33: Ease of operation,
34: Ease of repair, 35: Adaptability or versatility, 36: Device complexity,
37: Difficulty of detecting and measuring, 38: Extent of automation, 39: Productivity

Respond with a JSON object containing exactly this field:
{
  "contradictions": [
    {
      "description": "Improving X worsens Y",
      "improving_parameter": "<exact name from list above>",
      "worsening_parameter": "<exact name from list above>",
      "improving_id": <integer 1-39>,
      "worsening_id": <integer 1-39>
    }
  ]
}

IMPORTANT: improving_id and worsening_id MUST be integers between 1 and 39 inclusive.
Return 1 to 3 contradictions."""
```

- [ ] **Step 2: Verify syntax**

```bash
cd /Users/sevketugurel/Desktop/LIFTUP/smartriz-project && .venv/bin/python -c "from smartriz.agents.prompts import DETECTOR_SYSTEM; print(len(DETECTOR_SYSTEM), 'chars OK')"
```
Expected: prints char count with `OK`

- [ ] **Step 3: Commit**

```bash
git add src/smartriz/agents/prompts.py
git commit -m "feat(prompts): enrich DETECTOR_SYSTEM with structured contradiction schema + TRIZ-39 list"
```

---

### Task 3: Update `contradiction_detector` in graph.py

**Files:**
- Modify: `src/smartriz/agents/graph.py` (lines 33-46)

- [ ] **Step 1: Replace `contradiction_detector` function**

Replace lines 33-46 with:
```python
def contradiction_detector(state: TRIZState) -> dict:
    print("Agent: Detecting Contradictions...")
    result = chat_json(
        system=DETECTOR_SYSTEM,
        user=DETECTOR_USER.format(
            problem=state["original_problem"],
            analysis=state.get("analysis", ""),
        ),
        schema_hint='{"contradictions": [{"description": "...", "improving_parameter": "...", "worsening_parameter": "...", "improving_id": 1, "worsening_id": 14}]}',
    )
    raw_items = result.get("contradictions", [])
    if not raw_items:
        raw_items = [{"description": "Improving strength worsens weight"}]

    contradictions: list[str] = []
    contradiction_details: list[dict] = []

    for item in raw_items[:3]:
        if isinstance(item, str):
            contradictions.append(item)
            continue
        desc = item.get("description", "")
        if desc:
            contradictions.append(desc)
        imp_id = item.get("improving_id")
        wors_id = item.get("worsening_id")
        if (
            isinstance(imp_id, int)
            and isinstance(wors_id, int)
            and 1 <= imp_id <= 39
            and 1 <= wors_id <= 39
            and desc
        ):
            contradiction_details.append({
                "description": desc,
                "improving_parameter": item.get("improving_parameter", ""),
                "worsening_parameter": item.get("worsening_parameter", ""),
                "improving_id": imp_id,
                "worsening_id": wors_id,
            })

    if not contradictions:
        contradictions = ["Improving strength worsens weight"]

    return {
        "contradictions": contradictions,
        "contradiction_details": contradiction_details if contradiction_details else None,
    }
```

- [ ] **Step 2: Verify graph imports and compiles**

```bash
cd /Users/sevketugurel/Desktop/LIFTUP/smartriz-project && .venv/bin/python -c "from smartriz.agents.graph import triz_app; print('graph OK')"
```
Expected: `graph OK`

- [ ] **Step 3: Run lint**

```bash
cd /Users/sevketugurel/Desktop/LIFTUP/smartriz-project && ruff check src/smartriz/agents/
```
Expected: no errors (fix any reported)

- [ ] **Step 4: Run existing tests**

```bash
cd /Users/sevketugurel/Desktop/LIFTUP/smartriz-project && .venv/bin/python -m pytest tests/ -q 2>&1 | tail -20
```
Expected: all pass or same failures as before this change

- [ ] **Step 5: Commit**

```bash
git add src/smartriz/agents/graph.py
git commit -m "feat(graph): parse structured contradiction schema, emit contradiction_details"
```

---

### Task 4: Update SSE Hook — partialResult + Dynamic logLine

**Files:**
- Modify: `ui/src/hooks/useTrizStream.js`

- [ ] **Step 1: Replace useTrizStream.js with updated version**

The key changes: add `partialResult` state; accumulate it on every `agent_done`; derive `logLine` from `updates` content instead of `payload.log_line`.

```javascript
import { useCallback, useMemo, useRef, useState } from 'react'

const STEP_ORDER = ['analyst', 'detector', 'solver', 'critic']

const initialAgentState = {
  analyst: { status: 'waiting', logLine: 'Waiting for input.' },
  detector: { status: 'waiting', logLine: 'Waiting for upstream output.' },
  solver: { status: 'waiting', logLine: 'Waiting for contradiction mapping.' },
  critic: { status: 'waiting', logLine: 'Waiting for candidate solution.' },
}

function deriveLogLine(agent, updates) {
  const truncate = (str, max) => (str && str.length > max ? str.slice(0, max - 1) + '…' : str || '')
  const firstSentence = (str) => {
    if (!str) return ''
    const m = str.match(/[^.!?]+[.!?]/)
    return truncate(m ? m[0].trim() : str.trim(), 110)
  }

  switch (agent) {
    case 'analyst': {
      return firstSentence(updates.analysis) || 'Analysis complete.'
    }
    case 'detector': {
      return updates.contradictions?.[0] || 'No contradiction parsed.'
    }
    case 'solver': {
      const principles = updates.selected_principles ?? []
      const names = principles.slice(0, 2).join(', ')
      return `${principles.length} principle${principles.length !== 1 ? 's' : ''} selected${names ? ': ' + names : ''}`
    }
    case 'critic': {
      return firstSentence(updates.critic_feedback) || 'Evaluation complete.'
    }
    default:
      return 'Step complete.'
  }
}

export function useTrizStream() {
  const [status, setStatus] = useState('idle')
  const [result, setResult] = useState(null)
  const [partialResult, setPartialResult] = useState(null)
  const [error, setError] = useState('')
  const [agentStates, setAgentStates] = useState(initialAgentState)
  const sourceRef = useRef(null)

  const reset = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close()
      sourceRef.current = null
    }
    setStatus('idle')
    setResult(null)
    setPartialResult(null)
    setError('')
    setAgentStates(initialAgentState)
  }, [])

  const start = useCallback((problem) => {
    const trimmed = problem.trim()
    if (!trimmed) return

    if (sourceRef.current) {
      sourceRef.current.close()
    }

    setStatus('running')
    setError('')
    setResult(null)
    setPartialResult(null)
    setAgentStates(initialAgentState)

    const streamUrl = `http://localhost:8000/api/stream?problem=${encodeURIComponent(trimmed)}`
    const source = new EventSource(streamUrl)
    sourceRef.current = source

    source.addEventListener('agent_start', (event) => {
      const payload = JSON.parse(event.data)
      setAgentStates((prev) => ({
        ...prev,
        [payload.agent]: {
          ...prev[payload.agent],
          status: 'active',
          logLine: 'Running…',
        },
      }))
    })

    source.addEventListener('agent_done', (event) => {
      const payload = JSON.parse(event.data)
      const updates = payload.updates || {}
      const logLine = deriveLogLine(payload.agent, updates)

      setAgentStates((prev) => ({
        ...prev,
        [payload.agent]: {
          ...prev[payload.agent],
          status: 'done',
          logLine,
        },
      }))

      setPartialResult((prev) => ({ ...(prev || {}), ...updates }))
    })

    source.addEventListener('complete', (event) => {
      const payload = JSON.parse(event.data)
      setResult(payload.result)
      setStatus('complete')
      source.close()
      sourceRef.current = null
    })

    source.addEventListener('error', (event) => {
      const payload = event?.data ? JSON.parse(event.data) : null
      setError(payload?.message || 'Connection lost during analysis.')
      setStatus('error')
      source.close()
      sourceRef.current = null
    })
  }, [])

  const currentStep = useMemo(() => {
    if (status === 'idle') return 'Problem'
    const active = STEP_ORDER.find((step) => agentStates[step].status === 'active')
    if (!active) {
      if (status === 'complete') return 'Solution'
      return 'Problem'
    }
    if (active === 'analyst') return 'Problem'
    if (active === 'detector') return 'Contradiction'
    if (active === 'solver') return 'Principles'
    return 'Solution'
  }, [agentStates, status])

  return {
    status,
    result,
    partialResult,
    error,
    agentStates,
    currentStep,
    start,
    reset,
  }
}
```

- [ ] **Step 2: Lint check**

```bash
cd /Users/sevketugurel/Desktop/LIFTUP/smartriz-project/ui && npm run lint 2>&1 | grep -E "error|warning" | head -20
```
Fix any errors reported in `useTrizStream.js` specifically.

- [ ] **Step 3: Commit**

```bash
git add ui/src/hooks/useTrizStream.js
git commit -m "feat(hook): derive agent log lines from model output, add partialResult accumulator"
```

---

### Task 5: Update App.jsx — pair resolution, principle cards, confidence, remove reasoningMock

**Files:**
- Modify: `ui/src/App.jsx`

- [ ] **Step 1: Replace App.jsx with updated version**

```jsx
import { useMemo, useState } from 'react'
import Breadcrumb from './components/Breadcrumb'
import { casesMock } from './data/casesMock'
import { matrixMock } from './data/matrixMock'
import { parameters } from './data/parameters'
import { principles } from './data/principles'
import { useTrizStream } from './hooks/useTrizStream'
import AgentPipeline from './screens/AgentPipeline'
import ContradictionMatrix from './screens/ContradictionMatrix'
import PrincipleDetailPanel from './screens/PrincipleDetailPanel'
import ProblemInput from './screens/ProblemInput'
import ReasoningChain from './screens/ReasoningChain'
import SolutionOutput from './screens/SolutionOutput'
import './App.css'

function resolvePair(result, contradictionText) {
  // Primary: use structured contradiction_details from model
  const detail = result?.contradiction_details?.[0]
  if (detail) {
    const imp = detail.improving_id
    const wors = detail.worsening_id
    if (Number.isInteger(imp) && Number.isInteger(wors) && imp >= 1 && imp <= 39 && wors >= 1 && wors <= 39) {
      return { improvingId: imp, worseningId: wors }
    }
  }
  // Fallback: text heuristic
  console.warn('[SmarTRIZ] Using heuristic pair resolution — contradiction_details not available')
  const source = (contradictionText || '').toLowerCase()
  if (source.includes('weight') && source.includes('strength')) {
    return { improvingId: 1, worseningId: 14 }
  }
  if (source.includes('reliability') && source.includes('strength')) {
    return { improvingId: 27, worseningId: 14 }
  }
  return { improvingId: 27, worseningId: 14 }
}

function parsePrincipleEntry(entry) {
  // Parses "1: Segmentation" or "Segmentation" into { id, name }
  const match = entry.match(/^(\d+):\s*(.+)$/)
  if (match) {
    return { id: parseInt(match[1], 10), name: match[2].trim() }
  }
  return null
}

export default function App() {
  const [problem, setProblem] = useState('')
  const [domain, setDomain] = useState('')
  const [improvementParameter, setImprovementParameter] = useState('')
  const [expandedPrinciple, setExpandedPrinciple] = useState(null)
  const { status, result, error, agentStates, currentStep, start, reset } = useTrizStream()

  const pair = useMemo(
    () => resolvePair(result, result?.contradictions?.[0] || improvementParameter),
    [result, improvementParameter],
  )
  const matrixKey = `${pair.improvingId}-${pair.worseningId}`

  const recommendedPrinciples = useMemo(() => {
    const modelPrinciples = result?.selected_principles
    if (modelPrinciples && modelPrinciples.length > 0) {
      const cards = modelPrinciples.slice(0, 4).map((entry) => {
        const parsed = parsePrincipleEntry(entry)
        if (!parsed) return null
        const found = principles.find((p) => p.id === parsed.id)
        return found || { id: parsed.id, name: parsed.name, description: parsed.name }
      }).filter(Boolean)
      if (cards.length > 0) return cards
    }
    // Fallback to matrix mock
    console.warn('[SmarTRIZ] Using matrixMock fallback for principles — selected_principles not available')
    const ids = matrixMock[matrixKey] || [15, 35, 1]
    return principles.filter((item) => ids.includes(item.id))
  }, [result, matrixKey])

  const domainCases = useMemo(() => {
    if (!domain) return []
    const selected = casesMock.filter((item) => item.domain === domain)
    return selected.length ? selected : casesMock
  }, [domain])

  const confidence = useMemo(() => {
    if (!result) return null
    const contradictions = result.contradictions || []
    const selectedPrinciples = result.selected_principles || []
    return {
      contradictionClarity: contradictions.length === 0 ? 0 : Math.min(1, contradictions.length / 2),
      principleRelevance: selectedPrinciples.length >= 2 ? 0.85 : selectedPrinciples.length === 1 ? 0.6 : 0.3,
      caseSimilarity: domain ? 0.7 : null,
    }
  }, [result, domain])

  const handleAnalyze = () => {
    if (!problem.trim()) return
    start(problem)
  }

  const handleRefine = () => {
    reset()
    setExpandedPrinciple(null)
  }

  return (
    <div className="app-shell">
      {status !== 'idle' ? <Breadcrumb currentStep={currentStep} /> : null}

      {status === 'idle' || status === 'error' ? (
        <>
          <ProblemInput
            problem={problem}
            domain={domain}
            improvementParameter={improvementParameter}
            onProblemChange={setProblem}
            onDomainChange={setDomain}
            onImprovementParameterChange={setImprovementParameter}
            onAnalyze={handleAnalyze}
          />
          {error ? <p className="inline-error">{error}</p> : null}
        </>
      ) : null}

      {status === 'running' ? <AgentPipeline agentStates={agentStates} /> : null}

      {status === 'complete' ? (
        <main className="results-layout">
          <ContradictionMatrix
            parameters={parameters}
            improvingId={pair.improvingId}
            worseningId={pair.worseningId}
            matrixMap={matrixMock}
            recommendedPrinciples={recommendedPrinciples}
            onExpandPrinciple={setExpandedPrinciple}
          />

          <ReasoningChain state={result} />

          <SolutionOutput
            improvingLabel={parameters[pair.improvingId - 1].name}
            worseningLabel={parameters[pair.worseningId - 1].name}
            principles={recommendedPrinciples}
            cases={domainCases}
            finalSolution={result?.final_solution}
            confidence={confidence}
            onRefine={handleRefine}
          />

          <PrincipleDetailPanel
            principle={expandedPrinciple}
            cases={domainCases}
            onApply={() => setExpandedPrinciple(null)}
            onClose={() => setExpandedPrinciple(null)}
          />
        </main>
      ) : null}
    </div>
  )
}
```

- [ ] **Step 2: Lint check**

```bash
cd /Users/sevketugurel/Desktop/LIFTUP/smartriz-project/ui && npm run lint 2>&1 | grep "error" | head -20
```
Fix any errors in `App.jsx`.

- [ ] **Step 3: Commit**

```bash
git add ui/src/App.jsx
git commit -m "feat(app): wire real model output to pair resolution, principles, confidence; remove reasoningMock"
```

---

### Task 6: Update ReasoningChain — accept `state` prop, dynamic markdown

**Files:**
- Modify: `ui/src/screens/ReasoningChain.jsx`

- [ ] **Step 1: Replace ReasoningChain.jsx**

```jsx
import { useMemo, useState } from 'react'

function buildMarkdown(state) {
  if (!state) return ''
  const sections = []

  if (state.analysis) {
    sections.push(`## 1. Problem Analysis\n\n${state.analysis}`)
  }

  const contradictions = state.contradictions || []
  if (contradictions.length > 0) {
    const items = contradictions.map((c) => `- ${c}`).join('\n')
    sections.push(`## 2. Identified Contradictions\n\n${items}`)
  }

  const principles = state.selected_principles || []
  if (principles.length > 0) {
    const items = principles.map((p) => `- ${p}`).join('\n')
    sections.push(`## 3. Selected TRIZ Principles\n\n${items}`)
  }

  if (state.final_solution) {
    sections.push(`## 4. Proposed Solution\n\n${state.final_solution}`)
  }

  if (state.critic_feedback) {
    sections.push(`## 5. Critic Feedback\n\n${state.critic_feedback}`)
  }

  return sections.join('\n\n')
}

function renderMarkdown(markdown) {
  return markdown
    .split('\n')
    .map((line) => {
      if (line.startsWith('## ')) return `<h3>${line.slice(3)}</h3>`
      if (line.startsWith('- ')) return `<li>${line.slice(2)}</li>`
      if (line === '') return '<br/>'
      return `<p>${line}</p>`
    })
    .join('')
}

export default function ReasoningChain({ state }) {
  const [open, setOpen] = useState(false)
  const rendered = useMemo(() => renderMarkdown(buildMarkdown(state)), [state])

  return (
    <section className="reasoning-screen">
      <button className="text-button reasoning-toggle" onClick={() => setOpen((v) => !v)} type="button">
        See full reasoning chain {open ? '↑' : '↓'}
      </button>
      {open ? <div className="reasoning-content" dangerouslySetInnerHTML={{ __html: rendered }} /> : null}
    </section>
  )
}
```

- [ ] **Step 2: Lint check**

```bash
cd /Users/sevketugurel/Desktop/LIFTUP/smartriz-project/ui && npm run lint 2>&1 | grep "error" | head -10
```

- [ ] **Step 3: Commit**

```bash
git add ui/src/screens/ReasoningChain.jsx
git commit -m "feat(ReasoningChain): replace static mock with dynamic 5-section markdown from model state"
```

---

### Task 7: Update SolutionOutput — conditional cases, dynamic confidence

**Files:**
- Modify: `ui/src/screens/SolutionOutput.jsx`

- [ ] **Step 1: Replace SolutionOutput.jsx**

Note: `confidence` is now a prop with shape `{ contradictionClarity, principleRelevance, caseSimilarity }`. `cases` is empty array when no domain selected.

```jsx
import CaseCard from '../components/CaseCard'
import ConfidenceBar from '../components/ConfidenceBar'
import ParameterChip from '../components/ParameterChip'

export default function SolutionOutput({
  improvingLabel,
  worseningLabel,
  principles,
  cases,
  finalSolution,
  confidence,
  onRefine,
}) {
  const bullets = finalSolution
    ? finalSolution.split('.').map((item) => item.trim()).filter(Boolean).slice(0, 4)
    : []

  const showCases = cases && cases.length > 0

  return (
    <section className="solution-card">
      <div className="solution-section">
        <h3>Identified contradiction</h3>
        <div className="chip-row">
          <ParameterChip label={improvingLabel} trend="up" />
          <ParameterChip label={worseningLabel} trend="down" />
        </div>
      </div>

      <div className="solution-section">
        <h3>Recommended principles</h3>
        <div className="recommended-row">
          {principles.slice(0, 4).map((item) => (
            <span className="recommend-pill" key={item.id}>
              <span className="numeric">P{item.id}</span> {item.name}
            </span>
          ))}
        </div>
      </div>

      <div className="solution-section">
        <h3>Proposed solution</h3>
        <ul className="solution-list">
          {bullets.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </div>

      {showCases ? (
        <div className="solution-section">
          <h3>Reference cases (offline knowledge base)</h3>
          <div className="cases-grid">
            {cases.slice(0, 2).map((item) => (
              <CaseCard key={item.id} item={item} />
            ))}
          </div>
        </div>
      ) : null}

      {confidence ? (
        <div className="solution-section">
          <h3>Confidence indicators</h3>
          <ConfidenceBar label="Contradiction clarity" value={confidence.contradictionClarity} />
          <ConfidenceBar label="Principle relevance" value={confidence.principleRelevance} />
          {confidence.caseSimilarity !== null ? (
            <ConfidenceBar label="Case similarity" value={confidence.caseSimilarity} />
          ) : null}
        </div>
      ) : null}

      <div className="footer-actions">
        <button className="outline-button" type="button">
          Export as PDF
        </button>
        <button className="primary-button" type="button" onClick={onRefine}>
          Refine problem →
        </button>
      </div>
    </section>
  )
}
```

- [ ] **Step 2: Lint check**

```bash
cd /Users/sevketugurel/Desktop/LIFTUP/smartriz-project/ui && npm run lint 2>&1 | grep "error" | head -10
```

- [ ] **Step 3: Commit**

```bash
git add ui/src/screens/SolutionOutput.jsx
git commit -m "feat(SolutionOutput): conditional cases section, dynamic confidence props, remove static fills"
```

---

### Task 8: Backend Validation

**Files:** None — run-only verification

- [ ] **Step 1: Run graph test**

```bash
cd /Users/sevketugurel/Desktop/LIFTUP/smartriz-project && .venv/bin/python scripts/run_graph_test.py 2>&1 | tail -40
```
Expected: `contradiction_details` key present in final state output. Each item has `improving_id` and `worsening_id` as integers between 1-39.

- [ ] **Step 2: Run lint**

```bash
cd /Users/sevketugurel/Desktop/LIFTUP/smartriz-project && ruff check src/
```
Expected: no errors

- [ ] **Step 3: (Optional, if backend running) Curl SSE stream**

```bash
curl -N "http://localhost:8000/api/stream?problem=A%20heavy%20bridge%20component%20must%20remain%20strong%20while%20reducing%20weight." 2>&1 | head -60
```
Expected: `agent_done` events for `detector` contain `"contradiction_details"` in `updates`.

---

### Task 9: Frontend End-to-End Validation

**Files:** None — run-only verification

- [ ] **Step 1: Start dev server**

```bash
cd /Users/sevketugurel/Desktop/LIFTUP/smartriz-project/ui && npm run dev
```
Open `http://localhost:5173` in browser.

- [ ] **Step 2: Run full lint**

```bash
cd /Users/sevketugurel/Desktop/LIFTUP/smartriz-project/ui && npm run lint 2>&1
```
Fix any errors in files modified during this plan.

- [ ] **Step 3: Manual checklist — enter problem "A heavy bridge component must remain strong while reducing weight."**

- [ ] AgentPipeline: 4 agent log lines are dynamic (analyst shows first sentence of analysis, detector shows first contradiction string, solver shows principle count + names, critic shows first sentence of feedback). No "Running..." or static AGENT_LOG_LINES text after completion.
- [ ] ContradictionMatrix: column/row highlights correspond to IDs from `contradiction_details` (e.g., 1 and 14 for weight/strength). Heuristic fallback only fires if model skips IDs (check console for warn).
- [ ] Recommended Principles grid: card count equals `selected_principles` array length (not mock P15/P35).
- [ ] "See full reasoning chain" panel: opens and shows 5 `##` sections with real model text.
- [ ] Domain field left blank → "cases" section invisible in SolutionOutput.
- [ ] Confidence bars are not `0.78`/`0.72`/`0.66` static values — they vary based on model output.
- [ ] Browser console: no errors; mock fallback `console.warn` lines are acceptable.

- [ ] **Step 4: Final commit**

```bash
git add -A
git status  # review — should be empty or only lock file changes
git commit -m "chore: post-integration cleanup" --allow-empty
```
