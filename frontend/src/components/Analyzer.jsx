import { useMemo, useState } from "react";
import EmotionOrb from "./EmotionOrb";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const examples = [
  "I failed my test and I feel like I disappointed everyone.",
  "I finally got the internship I wanted and I cannot stop smiling.",
  "I am nervous about tomorrow but I still believe I can do it.",
];

export default function Analyzer({ onBack }) {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const example = useMemo(
    () => examples[Math.floor(Math.random() * examples.length)],
    []
  );

  async function analyze() {
    if (!text.trim()) return;

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error("The emotion engine could not process that text.");
      }

      setResult(await response.json());
    } catch (err) {
      setError(
        `${err.message} Make sure the FastAPI backend is running on port 8000.`
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="analyzer-page">
      <header className="topbar">
        <button className="back-btn" onClick={onBack}>← smh cheerful</button>
        <div className="status-pill"><span /> AI engine online</div>
      </header>

      <div className="analyzer-grid">
        <div className="input-panel">
          <p className="eyebrow">YOUR WORDS</p>
          <h2>Tell me what's going on.</h2>
          <p className="subcopy">
            Don't worry about writing perfectly. Describe the moment exactly
            how it feels.
          </p>

          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={example}
            maxLength={2000}
          />

          <div className="composer-row">
            <span>{text.length}/2000</span>
            <button
              className="primary-btn"
              onClick={analyze}
              disabled={loading || !text.trim()}
            >
              {loading ? "Reading..." : "Reveal emotions →"}
            </button>
          </div>

          {error && <p className="error-message">{error}</p>}

          <div className="model-card">
            <span className="model-icon">◈</span>
            <div>
              <strong>Unsupervised emotion discovery</strong>
              <p>Embeddings → deep latent space → UMAP → HDBSCAN → semantic alignment</p>
            </div>
          </div>
        </div>

        <div className="result-panel">
          {!result ? (
            <div className="empty-state">
              <EmotionOrb emotion="hope" />
              <h3>Your emotional map will appear here.</h3>
              <p>
                We look for patterns behind the words — not just whether the
                sentence sounds positive or negative.
              </p>
            </div>
          ) : (
            <div className="result-state">
              <EmotionOrb emotion={result.primary_emotion} />
              <p className="eyebrow">PRIMARY SIGNAL</p>
              <h3 className="primary-emotion">
                {result.primary_emotion}
              </h3>
              <div className="confidence">
                <div className="confidence-track">
                  <div
                    className="confidence-fill"
                    style={{ width: `${result.confidence * 100}%` }}
                  />
                </div>
                <span>{Math.round(result.confidence * 100)}% semantic match</span>
              </div>

              <div className="emotion-list">
                {result.emotions.map((item) => (
                  <div className="emotion-row" key={item.emotion}>
                    <span>{item.emotion}</span>
                    <div className="bar">
                      <div
                        className="bar-fill"
                        style={{ width: `${item.score * 100}%` }}
                      />
                    </div>
                    <b>{Math.round(item.score * 100)}%</b>
                  </div>
                ))}
              </div>

              <div className="explanation">
                <span>✦</span>
                <p>{result.explanation}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
