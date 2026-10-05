import React, { useEffect, useMemo, useState } from "react";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8001";

const ragExamples = [
  "What diagnosis is mentioned in the claim documents?",
  "What is the total claim amount?",
  "What is the patient's passport number?",
  "Who is Bill Gates?"
];

const agentExamples = [
  "Show pending claims",
  "What is the policy number?",
  "What is the policy number for claim CLM2024002193?",
  "Show rejected claims where total claim amount is greater than 100000",
  "Show top 3 claims by total claim amount",
  "Show claims where approved amount is less than total claim amount",
  "Delete rejected claims from the dataset"
];

// Read the saved API URL so students do not need to type it every time.
function getInitialApiBaseUrl() {
  return localStorage.getItem("docIntelApiBaseUrl") || DEFAULT_API_BASE_URL;
}

// Convert API values into readable table text.
function formatValue(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }

  if (typeof value === "number") {
    return value.toLocaleString("en-IN");
  }

  return String(value);
}

// Build a compact status label from a boolean flag.
function statusLabel(value) {
  return value ? "Available" : "Missing";
}

// Keep fetch handling consistent across all API calls.
async function requestJson(apiBaseUrl, path, options = {}) {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    },
    ...options
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "API request failed");
  }

  return data;
}

// Render a reusable section header for dashboard panels.
function SectionHeader({ eyebrow, title, action }) {
  return (
    <div className="section-header">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h2>{title}</h2>
      </div>
      {action}
    </div>
  );
}

// Render a small metric tile with stable dimensions.
function Metric({ label, value, tone = "neutral" }) {
  return (
    <div className={`metric metric-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

// Render JSON-like API output in a readable block.
function JsonBlock({ data }) {
  if (!data) {
    return <div className="empty-state">No response yet</div>;
  }

  return <pre className="json-block">{JSON.stringify(data, null, 2)}</pre>;
}

// Render the claims dataset as a dense table.
function ClaimsTable({ claims }) {
  if (!claims.length) {
    return <div className="empty-state">No claims loaded</div>;
  }

  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Claim ID</th>
            <th>Patient</th>
            <th>Policy</th>
            <th>Status</th>
            <th>Total</th>
            <th>Approved</th>
          </tr>
        </thead>
        <tbody>
          {claims.map((claim) => (
            <tr key={claim.claim_id}>
              <td>{claim.claim_id}</td>
              <td>{claim.patient_name}</td>
              <td>{claim.policy_number}</td>
              <td>
                <span className={`status-pill ${String(claim.claim_status).toLowerCase()}`}>
                  {claim.claim_status}
                </span>
              </td>
              <td>{formatValue(claim.total_claim_amount)}</td>
              <td>{formatValue(claim.approved_amount)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Render recent audit trail entries.
function AuditTrail({ entries }) {
  if (!entries.length) {
    return <div className="empty-state">No audit entries loaded</div>;
  }

  return (
    <div className="audit-list">
      {entries.slice(0, 8).map((entry) => (
        <article className="audit-entry" key={entry.request_id}>
          <div className="audit-topline">
            <strong>{entry.tool_used}</strong>
            <span className={`safety ${entry.safety_status}`}>{entry.safety_status}</span>
          </div>
          <p>{entry.user_request}</p>
          <span>{entry.final_answer}</span>
        </article>
      ))}
    </div>
  );
}

// Render clickable example prompts.
function PromptButtons({ examples, onPick }) {
  return (
    <div className="prompt-grid">
      {examples.map((example) => (
        <button className="ghost-button" key={example} onClick={() => onPick(example)}>
          {example}
        </button>
      ))}
    </div>
  );
}

// Main application shell for the provided document intelligence UI.
export default function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState(getInitialApiBaseUrl);
  const [health, setHealth] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState(null);
  const [claims, setClaims] = useState([]);
  const [auditEntries, setAuditEntries] = useState([]);
  const [ragQuestion, setRagQuestion] = useState(ragExamples[0]);
  const [agentQuestion, setAgentQuestion] = useState(agentExamples[3]);
  const [ragResponse, setRagResponse] = useState(null);
  const [agentResponse, setAgentResponse] = useState(null);
  const [selectedClaimId, setSelectedClaimId] = useState("CLM2024002193");
  const [selectedClaim, setSelectedClaim] = useState(null);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState("");

  const artifactCount = useMemo(() => {
    if (!pipelineStatus?.output_files) {
      return 0;
    }

    return Object.values(pipelineStatus.output_files).filter((item) => item.exists).length;
  }, [pipelineStatus]);

  // Persist API URL edits for local browser sessions.
  useEffect(() => {
    localStorage.setItem("docIntelApiBaseUrl", apiBaseUrl);
  }, [apiBaseUrl]);

  // Load dashboard data when the API URL changes.
  useEffect(() => {
    refreshDashboard();
  }, []);

  // Refresh all read-only dashboard data.
  async function refreshDashboard() {
    setError("");

    try {
      const [healthData, statusData, claimsData, auditData] = await Promise.all([
        requestJson(apiBaseUrl, "/health"),
        requestJson(apiBaseUrl, "/pipeline/status"),
        requestJson(apiBaseUrl, "/claims"),
        requestJson(apiBaseUrl, "/audit-trail")
      ]);

      setHealth(healthData);
      setPipelineStatus(statusData);
      setClaims(claimsData.records || []);
      setAuditEntries((auditData.entries || []).slice().reverse());
    } catch (err) {
      setError(err.message);
    }
  }

  // Trigger the backend pipeline through FastAPI.
  async function runPipeline() {
    setIsBusy(true);
    setError("");

    try {
      await requestJson(apiBaseUrl, "/pipeline/run", { method: "POST" });
      await refreshDashboard();
    } catch (err) {
      setError(err.message);
    } finally {
      setIsBusy(false);
    }
  }

  // Ask the RAG endpoint using the current prompt.
  async function askRag() {
    setIsBusy(true);
    setError("");

    try {
      const response = await requestJson(apiBaseUrl, "/rag/ask", {
        method: "POST",
        body: JSON.stringify({
          question: ragQuestion,
          top_k: 3,
          use_cache: true
        })
      });
      setRagResponse(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsBusy(false);
    }
  }

  // Ask the LLM agent endpoint using the current prompt.
  async function askAgent() {
    setIsBusy(true);
    setError("");

    try {
      const response = await requestJson(apiBaseUrl, "/agent/ask", {
        method: "POST",
        body: JSON.stringify({ question: agentQuestion })
      });
      setAgentResponse(response);
      await refreshDashboard();
    } catch (err) {
      setError(err.message);
    } finally {
      setIsBusy(false);
    }
  }

  // Load one claim detail record from the API.
  async function loadClaimDetail() {
    setIsBusy(true);
    setError("");

    try {
      const response = await requestJson(apiBaseUrl, `/claims/${selectedClaimId}`);
      setSelectedClaim(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <span className="eyebrow">AI Document Intelligence</span>
          <h1>PDFs to ML-Ready Data</h1>
        </div>
        <div className="api-control">
          <label htmlFor="apiBaseUrl">API Base URL</label>
          <input
            id="apiBaseUrl"
            value={apiBaseUrl}
            onChange={(event) => setApiBaseUrl(event.target.value)}
          />
          <button onClick={refreshDashboard} disabled={isBusy}>
            Refresh
          </button>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <section className="metrics-grid">
        <Metric label="API" value={health?.status || "Unknown"} tone={health ? "green" : "neutral"} />
        <Metric label="Artifacts" value={artifactCount} tone="blue" />
        <Metric label="Claims" value={claims.length} tone="green" />
        <Metric label="Audit Entries" value={auditEntries.length} tone="amber" />
      </section>

      <section className="panel">
        <SectionHeader
          eyebrow="Pipeline"
          title="Processing Status"
          action={
            <button className="primary-button" onClick={runPipeline} disabled={isBusy}>
              Run Pipeline
            </button>
          }
        />
        <div className="artifact-grid">
          {Object.entries(pipelineStatus?.output_files || {}).map(([name, artifact]) => (
            <div className="artifact" key={name}>
              <strong>{name.replaceAll("_", " ")}</strong>
              <span className={artifact.exists ? "ok-text" : "muted-text"}>
                {statusLabel(artifact.exists)}
              </span>
            </div>
          ))}
        </div>
      </section>

      <div className="two-column">
        <section className="panel">
          <SectionHeader eyebrow="RAG" title="Document Q&A" />
          <PromptButtons examples={ragExamples} onPick={setRagQuestion} />
          <textarea value={ragQuestion} onChange={(event) => setRagQuestion(event.target.value)} />
          <button className="primary-button" onClick={askRag} disabled={isBusy}>
            Ask RAG
          </button>
          <JsonBlock data={ragResponse} />
        </section>

        <section className="panel">
          <SectionHeader eyebrow="Agent" title="Dynamic Query Assistant" />
          <PromptButtons examples={agentExamples} onPick={setAgentQuestion} />
          <textarea value={agentQuestion} onChange={(event) => setAgentQuestion(event.target.value)} />
          <button className="primary-button" onClick={askAgent} disabled={isBusy}>
            Ask Agent
          </button>
          {agentResponse?.final_answer && (
            <div className="answer-box">
              <strong>Final Answer</strong>
              <p>{agentResponse.final_answer}</p>
            </div>
          )}
          <JsonBlock data={agentResponse} />
        </section>
      </div>

      <section className="panel">
        <SectionHeader
          eyebrow="Dataset"
          title="Claim Records"
          action={
            <div className="inline-form">
              <input value={selectedClaimId} onChange={(event) => setSelectedClaimId(event.target.value)} />
              <button onClick={loadClaimDetail} disabled={isBusy}>
                Load Claim
              </button>
            </div>
          }
        />
        <ClaimsTable claims={claims} />
        {selectedClaim && <JsonBlock data={selectedClaim} />}
      </section>

      <section className="panel">
        <SectionHeader eyebrow="Observability" title="Agent Audit Trail" />
        <AuditTrail entries={auditEntries} />
      </section>
    </main>
  );
}
